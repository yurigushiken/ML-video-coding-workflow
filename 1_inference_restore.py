import os
import cv2
import json
import time
import base64
import torch
import requests
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from torch.utils.data import DataLoader, Dataset
from itertools import islice
from collections import deque

# Retry / Adapter for robust HTTP requests
import requests.packages.urllib3.util.retry
import requests.adapters

# SAM2 imports
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

##############################################################################
#                                  SETTINGS                                  #
##############################################################################

# Base Input Directory containing all participant directories
INPUT_BASE_DIR = r"/path/to/input/base/dir"

# YOLO / Roboflow Inference settings
API_KEY = "YOUR_API_KEY"  
MODEL_ID = "YOUR_MODEL_ID"
MODEL_VERSION = "8"
INFERENCE_SERVER_URL = "http://localhost:9001/infer/object_detection"

# SAM2 settings
SAM2_CONFIG_FILE = r"/path/to/sam2_config.yaml"
SAM2_CHECKPOINT = r"/path/to/checkpoint-b4.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Processing settings
YOLO_CONFIDENCE_THRESHOLD = 0.5
MAX_WORKERS = 16
BATCH_SIZE = 16

##############################################################################
#                                  DATASET                                   #
##############################################################################

class FrameDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        frame = cv2.imread(str(img_path))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()
        return {
            'path': img_path,
            'frame': frame_tensor.pin_memory() if torch.cuda.is_available() else frame_tensor
        }

##############################################################################
#                              MAIN LOGIC                                    #
##############################################################################

class OptimizedYOLOSAM2Runner:
    def __init__(
        self,
        input_frames_dir,
        output_base_dir,
        api_key=API_KEY,
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        inference_url=INFERENCE_SERVER_URL,
        confidence_thresh=YOLO_CONFIDENCE_THRESHOLD,
        sam2_config_file=SAM2_CONFIG_FILE,
        sam2_checkpoint_file=SAM2_CHECKPOINT,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        max_workers=MAX_WORKERS
    ):
        self.input_frames_dir = Path(input_frames_dir)
        self.output_base_dir = Path(output_base_dir)
        self.api_key = api_key
        self.model_id = model_id
        self.model_version = model_version
        self.inference_url = inference_url
        self.confidence_thresh = confidence_thresh
        self.sam2_config_file = sam2_config_file
        self.sam2_checkpoint_file = sam2_checkpoint_file
        self.device = device
        self.batch_size = batch_size
        self.max_workers = max_workers

        # Initialize CUDA settings
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.cuda.set_per_process_memory_fraction(0.85)
            os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
            torch.cuda.empty_cache()

        # Initialize streams if CUDA is available
        self.streams = [torch.cuda.Stream() for _ in range(3)] if torch.cuda.is_available() else []

        # Initialize threading lock for SAM2 access
        self.sam2_lock = Lock()

        # Request session with retry strategy
        self.session = requests.Session()
        retries = requests.packages.urllib3.util.retry.Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[502, 503, 504]
        )
        adapter = requests.adapters.HTTPAdapter(max_retries=retries)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

        # Create output directory structure
        self.output_dir = self.create_output_dirs()
        self.json_dir = self.output_dir / "detections"
        self.visual_dir = self.output_dir / "visual_outputs"

        # Load font and model
        self.font = self.load_font()
        self.sam2_predictor = self.load_sam2_model()

        # We'll store random pastel colors by class,
        # but also have specific colors for some class labels.
        self.class_color_map = {}
        self.special_colors = {
            # BGR format
            "blue_dot":    (203, 192, 255),  
            "man_body":    (230, 216, 173),  
            "man_hands":   (0, 165, 255),    
            "man_face":    (0, 140, 255),    
            "woman_body":  (97, 105, 255),   
            "woman_face":  (0, 128, 0),      
            "woman_hands": (0, 255, 255),    
            "toy":         (128, 0, 128)     
        }

    def create_output_dirs(self):
        """
        Creates an 'inference_output' folder in the base output directory
        and subfolders for detections and visuals.
        """
        final_output_dir = self.output_base_dir / "inference_output"
        final_output_dir.mkdir(parents=True, exist_ok=True)

        (final_output_dir / "detections").mkdir(parents=True, exist_ok=True)
        (final_output_dir / "visual_outputs").mkdir(parents=True, exist_ok=True)

        return final_output_dir

    def load_font(self, font_path=None, size=15):
        """
        Loads a default font or a TTF if you provide a path.
        """
        try:
            if font_path and Path(font_path).exists():
                return ImageFont.truetype(font_path, size)
            else:
                return ImageFont.load_default()
        except Exception as e:
            print(f"Error loading font: {e}. Using default font.")
            return ImageFont.load_default()

    def load_sam2_model(self):
        """
        Builds and loads SAM2 with the given config/checkpoint.
        """
        try:
            sam2_model = build_sam2(self.sam2_config_file, self.sam2_checkpoint_file, device=self.device)
            predictor = SAM2ImagePredictor(sam2_model, device=self.device)
            print("SAM2.1 model is loaded.")
            return predictor
        except Exception as e:
            print(f"Error loading SAM2 model: {e}")
            return None

    def get_class_color(self, class_name):
        """
        If class_name has a specified pastel color, use it.
        Otherwise generate a random darker pastel color for variety.
        """
        if class_name in self.special_colors:
            self.class_color_map[class_name] = self.special_colors[class_name]
        elif class_name not in self.class_color_map:
            color = np.random.randint(120, 201, size=3, dtype=np.uint8)
            self.class_color_map[class_name] = tuple(int(x) for x in color)
        return self.class_color_map[class_name]

    def run_inference_on_image(self, image_path):
        """
        Performs YOLO object detection and SAM2 mask generation on a single image.
        Saves JSON and annotated image outputs unless they both already exist.
        """
        start_local = time.time()

        json_path = self.json_dir / f"{Path(image_path).stem}_detections.json"
        annotated_save_path = self.visual_dir / f"{Path(image_path).stem}_annotated.jpg"

        # If both the JSON file and annotated image exist, skip reprocessing
        if json_path.exists() and annotated_save_path.exists():
            duration = time.time() - start_local
            return True, f"Skipping {image_path}, already processed.", duration

        ##################
        # 1) YOLO DETECT #
        ##################
        try:
            with open(image_path, "rb") as f:
                img_bytes = f.read()
            b64_image = base64.b64encode(img_bytes).decode('utf-8')

            payload = {
                "api_key": self.api_key,
                "model_id": f"{self.model_id}/{self.model_version}",
                "model_type": "object-detection",
                "image": [
                    {
                        "type": "base64",
                        "value": b64_image
                    }
                ],
                "confidence": self.confidence_thresh,
                "iou_threshold": 0.5,
                "max_detections": 300
            }

            resp = self.session.post(self.inference_url, json=payload, timeout=120)
            if resp.status_code != 200:
                duration = time.time() - start_local
                return False, f"Error with YOLO detection: {resp.status_code} - {resp.text}", duration

            result = resp.json()
            if not isinstance(result, list) or len(result) == 0:
                duration = time.time() - start_local
                return False, "No YOLO result returned.", duration

            predictions = result[0].get("predictions", [])

        except Exception as e:
            duration = time.time() - start_local
            return False, f"Exception in YOLO step: {e}", duration

        #####################
        # 2) SAM SEGMENT(S) #
        #####################
        frame_bgr = cv2.imread(str(image_path))
        if frame_bgr is None:
            duration = time.time() - start_local
            return False, f"Unable to load {image_path}", duration

        annotated_overlay = frame_bgr.copy()
        final_predictions = []

        for pred in predictions:
            class_name = pred.get("class", "N/A")
            confidence = pred.get("confidence", 0)
            x_center   = pred.get("x", 0)
            y_center   = pred.get("y", 0)
            bbox_w     = pred.get("width", 0)
            bbox_h     = pred.get("height", 0)

            x0 = int(x_center - bbox_w / 2)
            y0 = int(y_center - bbox_h / 2)
            x1 = int(x_center + bbox_w / 2)
            y1 = int(y_center + bbox_h / 2)

            padding = 10
            x0 = max(x0 + padding, 0)
            y0 = max(y0 + padding, 0)
            x1 = min(x1 - padding, frame_bgr.shape[1] - 1)
            y1 = min(y1 - padding, frame_bgr.shape[0] - 1)

            if x1 <= x0 or y1 <= y0 or self.sam2_predictor is None:
                final_predictions.append({
                    "class": class_name,
                    "confidence": confidence,
                    "mask_pixels": []
                })
                continue

            try:
                input_point = np.array([[x_center, y_center]])
                input_label = np.array([1])

                with self.sam2_lock:
                    self.sam2_predictor.set_image(frame_bgr)
                    with torch.inference_mode():
                        with torch.cuda.amp.autocast():
                            masks, scores, _ = self.sam2_predictor.predict(
                                point_coords=input_point,
                                point_labels=input_label,
                                box=[x0, y0, x1, y1],
                                multimask_output=True
                            )

                if isinstance(masks, torch.Tensor):
                    masks = masks.cpu().numpy()

                if len(masks) > 0:
                    best_mask_index = np.argmax(scores)
                    segmask = masks[best_mask_index].astype(np.uint8)

                    binary_mask = (segmask > 0).astype(np.uint8) * 255
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    refined_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
                    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel, iterations=2)

                    refined_mask = cv2.GaussianBlur(refined_mask, (3, 3), 0)
                    refined_mask = (refined_mask > 127).astype(np.uint8)

                    mask_positions = np.argwhere(refined_mask > 0)
                    mask_positions = mask_positions.tolist()

                    final_predictions.append({
                        "class": class_name,
                        "confidence": confidence,
                        "mask_pixels": mask_positions
                    })

                    color_bgr = self.get_class_color(class_name)
                    obj_overlay = annotated_overlay.copy()
                    obj_overlay[refined_mask > 0] = color_bgr

                    alpha = 0.6
                    annotated_overlay = cv2.addWeighted(obj_overlay, alpha, annotated_overlay, 1 - alpha, 0)
                else:
                    final_predictions.append({
                        "class": class_name,
                        "confidence": confidence,
                        "mask_pixels": []
                    })
            except Exception as e:
                print(f"SAM2 error for {image_path}: {e}")
                final_predictions.append({
                    "class": class_name,
                    "confidence": confidence,
                    "mask_pixels": []
                })

        ##################
        # 3) SAVE OUTPUT #
        ##################
        try:
            with open(json_path, "w") as jf:
                json.dump(final_predictions, jf, indent=4)
        except Exception as e:
            print(f"Error saving JSON for {image_path}: {e}")

        try:
            annotated_pil = Image.fromarray(cv2.cvtColor(annotated_overlay, cv2.COLOR_BGR2RGB))
            annotated_pil.save(str(annotated_save_path), quality=95)
        except Exception as e:
            print(f"Error saving annotated image for {image_path}: {e}")

        duration = time.time() - start_local
        return True, str(image_path), duration

    def process_images(self):
        """
        - Gathers all frames in the input_frames_dir
        - Processes every frame that lacks *either* a JSON file or an annotated image.
        - YOLO -> SAM -> JSON (mask only) + annotated image
        """
        all_files = sorted([
            p for p in self.input_frames_dir.glob("*.*")
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        ])

        total_frames = len(all_files)
        if total_frames == 0:
            print(f"No frames found in {self.input_frames_dir}")
            return

        print(f"Total frames in directory: {total_frames}")

        selected_files = []
        for img_path in all_files:
            json_path = self.json_dir / f"{img_path.stem}_detections.json"
            annotated_path = self.visual_dir / f"{img_path.stem}_annotated.jpg"
            if not (json_path.exists() and annotated_path.exists()):
                selected_files.append(img_path)

        if len(selected_files) == 0:
            print(f"All {total_frames} frames already have JSON & annotated images; skipping.")
            return

        print(f"Processing {len(selected_files)} out of {total_frames} frames (missing JSON or annotated image).")

        last_messages = deque(maxlen=10)
        start_time = time.time()
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_image = {executor.submit(self.run_inference_on_image, img): img for img in selected_files}

            processed_count = 0
            for future in as_completed(future_to_image):
                img_path = future_to_image[future]
                processed_count += 1

                elapsed_total = time.time() - start_time
                rate = processed_count / elapsed_total if elapsed_total > 0 else 0
                est_time_left = (len(selected_files) - processed_count) / rate if rate > 0 else 0
                total_est_time = elapsed_total + est_time_left

                elapsed_min = int(elapsed_total / 60)
                total_est_min = int(total_est_time / 60)

                try:
                    result_tuple = future.result()
                    if len(result_tuple) == 3:
                        success, msg, file_duration = result_tuple
                    else:
                        success, msg = result_tuple
                        file_duration = 0.0
                    short_info = f"✓ {Path(msg).name}" if success else f"✗ {msg}"
                except Exception as e:
                    success = False
                    msg = f"Exception: {img_path} -> {e}"
                    file_duration = 0.0
                    short_info = f"✗ {msg}"

                line_message = (
                    f"[{processed_count}/{len(selected_files)}] "
                    f"[{rate:.2f} it/s] Rate: {rate:.2f} imgs/s, "
                    f"took {file_duration:.2f}s -- {short_info}"
                )
                last_messages.append(line_message)
                results.append((success, msg))

                os.system('cls' if os.name == 'nt' else 'clear')
                print(f"Elapsed time: {elapsed_min} min")
                print(f"Estimated total time: {total_est_min} min")
                print(f"Directory: {self.input_frames_dir}")
                percent_done = (processed_count / len(selected_files)) * 100
                print(f"Progress: {percent_done:0.1f}% ({processed_count}/{len(selected_files)})")

                print("\n--- Last 10 processed frames ---")
                for msg_line in last_messages:
                    print(msg_line)

        successes = sum(1 for r in results if r[0])
        print(f"\nDone. Successfully processed {successes}/{len(results)} frames.")
        print(f"Results are in {self.output_dir}")

    def monitor_performance(self):
        """Monitor GPU performance metrics."""
        if torch.cuda.is_available():
            print(f"Current GPU Memory Usage: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            print(f"Max GPU Memory Usage: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

            try:
                gpu_util = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]
                ).decode("utf-8").strip()
                print(f"GPU Utilization: {gpu_util}%")
            except Exception as e:
                print(f"Failed to get GPU utilization: {e}")


def main():
    base_input_dir = Path(INPUT_BASE_DIR)
    if not base_input_dir.exists():
        print(f"Base input directory does not exist: {base_input_dir}")
        return

    frames_dirs = list(base_input_dir.rglob("1024x1024_frames"))
    if not frames_dirs:
        print(f"No directories named '1024x1024_frames' found under {base_input_dir}")
        return

    print(f"Found {len(frames_dirs)} '1024x1024_frames' directories to process.")

    for frames_dir in frames_dirs:
        participant_dir = frames_dir.parent
        print(f"\nProcessing participant directory: {participant_dir}")
        output_base_dir = participant_dir

        runner = OptimizedYOLOSAM2Runner(
            input_frames_dir=frames_dir,
            output_base_dir=output_base_dir
        )

        runner.process_images()
        runner.monitor_performance()


if __name__ == "__main__":
    main()
