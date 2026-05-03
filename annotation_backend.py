import argparse  # Import argparse so the backend can build one namespace object for the shared tile runner.
import csv  # Import csv so the backend can read and write detection tables.
from datetime import datetime  # Import datetime so each semi-automatic run gets a unique timestamp.
import json  # Import json so the backend can persist manual state and write COCO annotation files.
from pathlib import Path  # Import Path so file and folder handling stays readable and safe.
import shutil  # Import shutil so the backend can copy images and package zip archives.
import sys  # Import sys so the backend can expose the local Segment Anything repository on sys.path.

import cv2  # Import OpenCV so the backend can draw overlays, labels, and masks.
import numpy as np  # Import NumPy so the backend can work with masks, boxes, and images.
from PIL import Image  # Import PIL Image so the backend can read image sizes and create image outputs.
import torch  # Import torch so the backend can build tensors for NMS and resolve devices through shared helpers.
import torchvision  # Import torchvision so the backend can run non-maximum suppression on detections.

REPO_ROOT = Path(__file__).resolve().parent  # Resolve the repository root from the current backend file location.
SEGMENT_ANYTHING_REPO = REPO_ROOT / "segment_anything"  # Build the outer Segment Anything repository path.
if str(SEGMENT_ANYTHING_REPO) not in sys.path:  # Add the local Segment Anything repository to Python's import search path when needed.
    sys.path.insert(0, str(SEGMENT_ANYTHING_REPO))  # Prepend the local Segment Anything repository path so imports resolve to the checked-in code first.

from segment_anything import SamPredictor, sam_model_registry  # Import the classic SAM predictor pieces used by the desktop manual workflow.
from tile_grounded_sam_runner import build_models  # Import the shared Grounded-SAM model builder so the desktop tool reuses the verified pipeline.
from tile_grounded_sam_runner import crop_tiles  # Import the shared tile cropper so the desktop tool matches the verified tiling behavior.
from tile_grounded_sam_runner import ensure_paths_exist  # Import the shared path validator so missing config or checkpoint files fail early.
from tile_grounded_sam_runner import resolve_device  # Import the shared device resolver so auto, CPU, and GPU mode stay consistent.
from tile_grounded_sam_runner import run_grounded_sam_on_tiles  # Import the shared tile inference runner so the desktop tool stays aligned with the CLI flow.

APP_ROOT = REPO_ROOT  # Point the application root at the repository root.
DEFAULT_CONFIG = APP_ROOT / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"  # Point to the verified GroundingDINO config file.
DEFAULT_GROUNDING_CHECKPOINT = APP_ROOT / "weights" / "groundingdino_swint_ogc.pth"  # Point to the verified GroundingDINO checkpoint.
DEFAULT_SAM_CHECKPOINT = APP_ROOT / "weights" / "sam_vit_b_01ec64.pth"  # Point to the verified SAM checkpoint.
DEFAULT_SAM_ENCODER = "vit_b"  # Use the lighter SAM encoder that already works on the current machine.
DEFAULT_PROMPT = "car_plate . plate ."  # Keep the current default prompt that the desktop tool already expects.
DEFAULT_CLASS_NAME = "car_plate"  # Keep the current default YOLO class name that the desktop tool already expects.
DEFAULT_OUTPUT_ROOT = APP_ROOT / "outputs" / "dataset_ui_runs"  # Point to the default shared output root.
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}  # Define the supported input-image suffix set.
GROUNDED_MODEL_CACHE = {}  # Create a cache so GroundingDINO and SAM only load once per configuration.
MANUAL_SAM_CACHE = {}  # Create a cache so the manual SAM predictor only loads once per configuration.
EXPORT_FORMAT_YOLO_BOX = "YOLO Detection (v8-v12 compatible)"  # Define the shared YOLO detection export label so one txt layout can serve modern YOLO detection models.
EXPORT_FORMAT_YOLO_SEG = "YOLO Segmentation (v8-v12 compatible)"  # Define the shared YOLO segmentation export label so one polygon txt layout can serve modern YOLO segmentation models.
EXPORT_FORMAT_COCO_BOX = "COCO Detection JSON"  # Define the COCO detection export label for bbox-style COCO datasets.
EXPORT_FORMAT_COCO_SEG = "COCO Instance Segmentation JSON"  # Define the COCO instance-segmentation export label for polygon-style COCO datasets.
EXPORT_FORMAT_OPTIONS = [EXPORT_FORMAT_YOLO_BOX, EXPORT_FORMAT_YOLO_SEG, EXPORT_FORMAT_COCO_BOX, EXPORT_FORMAT_COCO_SEG]  # Collect every supported export format into one ordered list for desktop and web selectors.


def normalize_output_root(output_root_text):  # Define a helper that resolves the selected output root into one safe absolute path.
    requested_text = str(output_root_text).strip()  # Normalize the incoming output-root text by stripping surrounding whitespace.
    base_path = Path(requested_text) if requested_text else DEFAULT_OUTPUT_ROOT  # Use the requested output-root path when present and otherwise fall back to the default output root.
    if not base_path.is_absolute():  # Resolve relative output-root paths against the repository root so behavior stays predictable.
        base_path = APP_ROOT / base_path  # Convert the relative output-root path into a repository-root-relative absolute path.
    return base_path.resolve()  # Return the normalized absolute output-root path to the caller.


def build_args(image_path, prompt, rows, cols, box_threshold, text_threshold, nms_threshold, max_detections, min_box_area, output_root, device_mode):  # Define a helper that builds the shared runner-style namespace from desktop settings.
    return argparse.Namespace(image=str(image_path), rows=int(rows), cols=int(cols), output_root=str(output_root), config=str(DEFAULT_CONFIG), grounding_checkpoint=str(DEFAULT_GROUNDING_CHECKPOINT), sam_checkpoint=str(DEFAULT_SAM_CHECKPOINT), sam_encoder_version=DEFAULT_SAM_ENCODER, prompt=str(prompt), box_threshold=float(box_threshold), text_threshold=float(text_threshold), nms_threshold=float(nms_threshold), max_detections=int(max_detections), min_box_area=float(min_box_area), device_mode=str(device_mode), dry_run=False)  # Return one namespace object that matches the shared tile runner interface.


def build_grounded_cache_key(args):  # Define a helper that builds one stable cache key for Grounded-SAM model reuse.
    return (str(args.config), str(args.grounding_checkpoint), str(args.sam_checkpoint), str(args.sam_encoder_version), str(getattr(args, "device_mode", "auto")))  # Return the tuple that uniquely identifies the current Grounded-SAM model bundle.


def get_cached_grounded_models(args):  # Define a helper that returns cached Grounded-SAM models or builds them on first use.
    cache_key = build_grounded_cache_key(args)  # Build the cache key for the current Grounded-SAM configuration.
    if cache_key not in GROUNDED_MODEL_CACHE:  # Build the models only when this exact configuration has not been seen yet.
        _, config_path, grounding_checkpoint_path, sam_checkpoint_path = ensure_paths_exist(args)  # Validate the key paths before building any heavy model objects.
        GROUNDED_MODEL_CACHE[cache_key] = build_models(args, config_path, grounding_checkpoint_path, sam_checkpoint_path)  # Build and cache the GroundingDINO model, SAM predictor, and resolved device tuple.
    return GROUNDED_MODEL_CACHE[cache_key]  # Return the cached Grounded-SAM model bundle to the caller.


def build_manual_sam_cache_key(checkpoint_path, encoder_version, resolved_device):  # Define a helper that builds one stable cache key for classic SAM reuse.
    return (str(checkpoint_path), str(encoder_version), str(resolved_device))  # Return the tuple that uniquely identifies one manual SAM bundle.


def get_shared_bundle_from_grounded_cache(checkpoint_path, encoder_version, resolved_device):  # Define a helper that reuses the SAM predictor already loaded by Grounded-SAM when possible.
    for cache_key, cache_value in GROUNDED_MODEL_CACHE.items():  # Loop over every cached Grounded-SAM bundle currently in memory.
        same_checkpoint = cache_key[2] == str(checkpoint_path)  # Check whether the current Grounded-SAM bundle uses the same SAM checkpoint.
        same_encoder = cache_key[3] == str(encoder_version)  # Check whether the current Grounded-SAM bundle uses the same SAM encoder version.
        same_device = cache_value[2] == str(resolved_device)  # Check whether the current Grounded-SAM bundle already runs on the same resolved device.
        if same_checkpoint and same_encoder and same_device:  # Reuse the current Grounded-SAM SAM predictor only when all key settings match.
            return cache_value[1].model, cache_value[1]  # Return the shared SAM model and predictor from the current Grounded-SAM bundle.
    return None  # Return nothing when no reusable Grounded-SAM SAM bundle currently matches.


def get_manual_sam_bundle(checkpoint_path, encoder_version, device_mode):  # Define a helper that returns one cached classic SAM model and predictor pair.
    resolved_device = resolve_device(device_mode)  # Resolve the requested manual-SAM device mode into one real runtime device string.
    cache_key = build_manual_sam_cache_key(checkpoint_path, encoder_version, resolved_device)  # Build the cache key for the current manual SAM configuration.
    if cache_key not in MANUAL_SAM_CACHE:  # Build or reuse the manual SAM bundle only when this exact configuration has not been cached yet.
        shared_bundle = get_shared_bundle_from_grounded_cache(checkpoint_path, encoder_version, resolved_device)  # Try to reuse the SAM bundle already loaded by Grounded-SAM.
        if shared_bundle is not None:  # Reuse the Grounded-SAM SAM bundle when an exact match already exists.
            MANUAL_SAM_CACHE[cache_key] = shared_bundle  # Save the reused shared SAM bundle into the manual cache.
        else:  # Build a dedicated classic SAM bundle only when no shared one is available.
            sam_model = sam_model_registry[encoder_version](checkpoint=str(checkpoint_path))  # Build the requested SAM model from the selected checkpoint.
            sam_model.to(device=resolved_device)  # Move the current SAM model onto the selected runtime device.
            MANUAL_SAM_CACHE[cache_key] = (sam_model, SamPredictor(sam_model))  # Cache the built SAM model and predictor for later reuse.
    return MANUAL_SAM_CACHE[cache_key]  # Return the cached classic SAM bundle to the caller.


def load_detection_rows(detections_csv_path):  # Define a helper that reads the generated detections CSV into normalized dictionaries.
    detection_path = Path(str(detections_csv_path))  # Convert the incoming detections CSV path into a Path object.
    if not detection_path.exists():  # Return an empty list when the detections CSV file does not exist.
        return []  # Exit early because there are no detections to load.
    normalized_rows = []  # Create a list that will collect the normalized detection rows.
    with detection_path.open("r", encoding="utf-8", newline="") as csv_file:  # Open the detections CSV file for reading.
        for row in csv.DictReader(csv_file):  # Loop over every row in the detections CSV file.
            normalized_rows.append({"tile_name": row["tile_name"], "row_index": int(row["row_index"]), "col_index": int(row["col_index"]), "tile_left": int(float(row["tile_left"])), "tile_top": int(float(row["tile_top"])), "tile_right": int(float(row["tile_right"])), "tile_bottom": int(float(row["tile_bottom"])), "detection_index": int(row["detection_index"]), "phrase": row["phrase"], "confidence": float(row["confidence"]), "tile_x1": float(row["tile_x1"]), "tile_y1": float(row["tile_y1"]), "tile_x2": float(row["tile_x2"]), "tile_y2": float(row["tile_y2"]), "global_x1": float(row["global_x1"]), "global_y1": float(row["global_y1"]), "global_x2": float(row["global_x2"]), "global_y2": float(row["global_y2"])})  # Append one normalized detection row into the output list.
    return normalized_rows  # Return the normalized detection rows to the caller.


def apply_global_nms(detection_rows, global_nms_threshold):  # Define a helper that reduces duplicate detections after tile boxes are mapped back onto the full image.
    if not detection_rows:  # Return an empty list when there are no detections to filter.
        return []  # Exit early because global NMS has nothing to process.
    box_array = np.array([[row["global_x1"], row["global_y1"], row["global_x2"], row["global_y2"]] for row in detection_rows], dtype=np.float32)  # Build the global xyxy box array from the detection rows.
    score_array = np.array([row["confidence"] for row in detection_rows], dtype=np.float32)  # Build the confidence array from the detection rows.
    keep_indices = torchvision.ops.nms(torch.from_numpy(box_array), torch.from_numpy(score_array), float(global_nms_threshold)).cpu().numpy().tolist()  # Run global NMS and convert the kept indices back into a Python list.
    keep_indices = sorted(keep_indices)  # Sort the kept indices so the final row order stays stable.
    return [detection_rows[index] for index in keep_indices]  # Return the globally filtered detection rows.


def parse_prompt_to_class_name(prompt_text, fallback_name):  # Define a helper that converts the current prompt into one YOLO-safe class name.
    prompt_parts = [part.strip() for part in str(prompt_text).split(".") if part.strip()]  # Split the prompt into non-empty dot-separated phrases.
    base_name = fallback_name if str(fallback_name).strip() else (prompt_parts[0] if prompt_parts else "target_object")  # Use the explicit class name first and otherwise fall back to the first prompt phrase.
    safe_name = str(base_name).lower().replace(" ", "_").replace("/", "_").replace("\\", "_")  # Normalize the selected class name into one filesystem-safe token.
    while "__" in safe_name:  # Collapse repeated underscores so the final class name stays readable.
        safe_name = safe_name.replace("__", "_")  # Remove one level of repeated underscores from the selected class name.
    return safe_name.strip("_") or "target_object"  # Return the normalized class name and fall back again when the result becomes empty.


def get_annotation_color(index):  # Define a helper that gives each detection or annotation one stable display color.
    rng = np.random.default_rng(seed=index + 20260429)  # Seed one local random generator from the current annotation index.
    return rng.integers(80, 256, size=3, dtype=np.uint8)  # Return one bright RGB color for the current annotation index.


def build_mask_path(output_root, row):  # Define a helper that reconstructs one per-detection binary mask path from the saved folder layout.
    tile_stem = Path(row["tile_name"]).stem  # Read the current tile stem from the saved tile image name.
    return Path(str(output_root)) / "masks" / tile_stem / f"{tile_stem}_mask_{row['detection_index']:02d}.png"  # Return the deterministic mask path for the current detection.


def draw_multilingual_label(image_rgb, text, left, top, color):  # Define a helper that draws one readable label with a solid background on an overlay image.
    font = cv2.FONT_HERSHEY_SIMPLEX  # Pick one compact OpenCV font for overlay labels.
    scale = 0.55  # Set the font scale to one compact readable value.
    thickness = 2  # Set the font thickness so labels remain visible after resizing.
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)  # Measure the current label so the background box can be sized correctly.
    box_left = max(0, int(left))  # Clamp the label left coordinate so it stays inside the image.
    box_top = max(text_height + baseline + 6, int(top))  # Clamp the label top coordinate so the background box stays visible.
    cv2.rectangle(image_rgb, (box_left, box_top - text_height - baseline - 6), (box_left + text_width + 8, box_top + 2), tuple(int(channel) for channel in color.tolist()), -1)  # Draw the filled label background box.
    cv2.putText(image_rgb, text, (box_left + 4, box_top - baseline - 2), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)  # Draw the label text on top of the colored background box.


def build_full_image_outputs(image_path, output_root, filtered_rows, draw_boxes, draw_labels):  # Define a helper that maps all tile detections and masks back onto the original image.
    original_bgr = cv2.imread(str(image_path))  # Read the original image from disk as a BGR image.
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)  # Convert the original BGR image into RGB for display and blending.
    overlay_rgb = original_rgb.copy()  # Copy the original RGB image so the overlay stays separate from the clean input.
    combined_mask = np.zeros((original_rgb.shape[0], original_rgb.shape[1]), dtype=np.uint8)  # Create one empty full-image binary mask that will accumulate every kept detection mask.
    for index, row in enumerate(filtered_rows):  # Loop over every globally kept detection row.
        color = get_annotation_color(index)  # Pick one stable display color for the current detection.
        mask_path = build_mask_path(output_root, row)  # Rebuild the binary mask path for the current detection.
        if mask_path.exists():  # Blend the saved SAM mask only when the expected mask file exists.
            tile_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)  # Read the current binary mask from disk as a grayscale image.
            if tile_mask is not None:  # Continue only when OpenCV successfully reads the current binary mask.
                tile_mask_bool = tile_mask > 0  # Convert the tile-local binary mask into one boolean mask.
                global_mask = np.zeros((original_rgb.shape[0], original_rgb.shape[1]), dtype=bool)  # Create one empty full-image boolean mask for the current detection.
                tile_top = int(row["tile_top"])  # Read the tile top coordinate for the current detection.
                tile_left = int(row["tile_left"])  # Read the tile left coordinate for the current detection.
                tile_height = tile_mask_bool.shape[0]  # Read the current tile mask height.
                tile_width = tile_mask_bool.shape[1]  # Read the current tile mask width.
                global_mask[tile_top:tile_top + tile_height, tile_left:tile_left + tile_width] = tile_mask_bool  # Paste the tile-local mask back into the full-image mask canvas.
                overlay_rgb[global_mask] = (overlay_rgb[global_mask] * 0.45 + color * 0.55).astype(np.uint8)  # Blend the current full-image mask onto the overlay image.
                combined_mask[global_mask] = 255  # Mark the current full-image mask area inside the combined binary mask.
        if bool(draw_boxes):  # Draw the current global bounding box only when box drawing is enabled.
            left = int(round(row["global_x1"]))  # Read and round the current global left coordinate.
            top = int(round(row["global_y1"]))  # Read and round the current global top coordinate.
            right = int(round(row["global_x2"]))  # Read and round the current global right coordinate.
            bottom = int(round(row["global_y2"]))  # Read and round the current global bottom coordinate.
            cv2.rectangle(overlay_rgb, (left, top), (right, bottom), tuple(int(channel) for channel in color.tolist()), 2)  # Draw the current global bounding box on the full-image overlay.
        if bool(draw_labels):  # Draw the current label only when label drawing is enabled.
            label_text = f"{row['phrase']} {row['confidence']:.2f}"  # Build one compact label string for the current detection.
            draw_multilingual_label(overlay_rgb, label_text, int(round(row["global_x1"])), int(round(row["global_y1"])), color)  # Draw the current label on the full-image overlay.
    full_overlay_path = Path(str(output_root)) / "full_image_overlay.jpg"  # Build the output path for the full-image overlay image.
    full_mask_path = Path(str(output_root)) / "full_image_mask.png"  # Build the output path for the combined full-image mask image.
    Image.fromarray(overlay_rgb).save(full_overlay_path)  # Save the current full-image overlay image to disk.
    Image.fromarray(combined_mask).save(full_mask_path)  # Save the combined full-image mask image to disk.
    return Image.fromarray(overlay_rgb), Image.fromarray(combined_mask), full_overlay_path, full_mask_path  # Return the overlay image, the mask image, and both saved file paths.


def write_filtered_detections_csv(filtered_rows, output_root):  # Define a helper that writes the globally filtered detections into one CSV file.
    output_path = Path(str(output_root)) / "full_image_detections.csv"  # Build the filtered detections CSV path under the current run folder.
    field_names = ["tile_name", "row_index", "col_index", "tile_left", "tile_top", "tile_right", "tile_bottom", "detection_index", "phrase", "confidence", "tile_x1", "tile_y1", "tile_x2", "tile_y2", "global_x1", "global_y1", "global_x2", "global_y2"]  # Define the fixed CSV field order for filtered detections.
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:  # Open the filtered detections CSV file for writing.
        writer = csv.DictWriter(csv_file, fieldnames=field_names)  # Build the CSV writer for the filtered detections file.
        writer.writeheader()  # Write the filtered detections CSV header row.
        for row in filtered_rows:  # Loop over every globally filtered detection row.
            writer.writerow(row)  # Write the current filtered detection row into the CSV file.
    return output_path  # Return the filtered detections CSV path to the caller.


def write_yolo_files(filtered_rows, image_path, output_root, class_name):  # Define a helper that exports the current globally filtered detections in YOLO detection format.
    yolo_root = Path(str(output_root)) / "yolo"  # Build the YOLO export root under the current run folder.
    yolo_root.mkdir(parents=True, exist_ok=True)  # Create the YOLO export root folder when it does not already exist.
    label_path = yolo_root / f"{Path(str(image_path)).stem}.txt"  # Build the YOLO label path for the copied source image.
    classes_path = yolo_root / "classes.txt"  # Build the YOLO classes file path under the current run folder.
    with Image.open(image_path) as image:  # Open the copied source image so YOLO coordinates can be normalized from its dimensions.
        image_width, image_height = image.size  # Read the copied source image width and height once for normalization.
    normalized_class_name = parse_prompt_to_class_name(class_name, class_name)  # Normalize the current class name into one filesystem-safe token.
    classes_path.write_text(f"{normalized_class_name}\n", encoding="utf-8")  # Write the single current class name into the YOLO classes file.
    with label_path.open("w", encoding="utf-8") as label_file:  # Open the YOLO label file for writing.
        for row in filtered_rows:  # Loop over every globally filtered detection row.
            left = float(row["global_x1"])  # Read the current global left coordinate.
            top = float(row["global_y1"])  # Read the current global top coordinate.
            right = float(row["global_x2"])  # Read the current global right coordinate.
            bottom = float(row["global_y2"])  # Read the current global bottom coordinate.
            box_width = max(0.0, right - left)  # Compute the current box width in pixels.
            box_height = max(0.0, bottom - top)  # Compute the current box height in pixels.
            center_x = left + (box_width / 2.0)  # Compute the current box center x coordinate in pixels.
            center_y = top + (box_height / 2.0)  # Compute the current box center y coordinate in pixels.
            normalized_center_x = center_x / float(image_width) if image_width else 0.0  # Normalize the current center x coordinate into YOLO space.
            normalized_center_y = center_y / float(image_height) if image_height else 0.0  # Normalize the current center y coordinate into YOLO space.
            normalized_width = box_width / float(image_width) if image_width else 0.0  # Normalize the current box width into YOLO space.
            normalized_height = box_height / float(image_height) if image_height else 0.0  # Normalize the current box height into YOLO space.
            label_file.write(f"0 {normalized_center_x:.6f} {normalized_center_y:.6f} {normalized_width:.6f} {normalized_height:.6f}\n")  # Write one YOLO detection line for the current globally filtered box.
    return label_path, classes_path  # Return the YOLO label path and the YOLO classes path to the caller.


def summarize_grounded_results(raw_rows, filtered_rows, rows, cols, prompt, class_name):  # Define a helper that converts one Grounded-SAM run into one short readable summary.
    summary_lines = []  # Create a list that will collect the Grounded-SAM summary lines.
    summary_lines.append(f"Prompt: {prompt}")  # Append the current prompt line.
    summary_lines.append(f"Class: {class_name}")  # Append the current class name line.
    summary_lines.append(f"Tiles: {int(rows)} x {int(cols)}")  # Append the current tile grid size line.
    summary_lines.append(f"Raw detections: {len(raw_rows)}")  # Append the raw per-tile detection count line.
    summary_lines.append(f"Filtered detections: {len(filtered_rows)}")  # Append the globally filtered detection count line.
    for index, row in enumerate(filtered_rows[:20]):  # Loop over at most the first twenty filtered detections so the summary stays readable.
        summary_lines.append(f"#{index + 1} | {row['phrase']} | score={row['confidence']:.4f} | bbox=({int(round(row['global_x1']))}, {int(round(row['global_y1']))}, {int(round(row['global_x2']))}, {int(round(row['global_y2']))})")  # Append one compact line for the current filtered detection.
    return "\n".join(summary_lines)  # Join and return the Grounded-SAM summary text.


def build_manual_label_path(image_path, output_root):  # Define a helper that builds the legacy manual YOLO label path for one image.
    return normalize_output_root(output_root) / "manual_yolo" / f"{Path(str(image_path)).stem}.txt"  # Return the deterministic legacy manual YOLO label path.


def build_manual_classes_path(output_root):  # Define a helper that builds the legacy manual classes file path.
    return normalize_output_root(output_root) / "manual_yolo" / "classes.txt"  # Return the deterministic legacy manual classes file path.


def build_manual_bundle_path(image_path, output_root):  # Define a helper that builds the deterministic legacy manual-download zip path for one image.
    return normalize_output_root(output_root) / "manual_yolo" / "downloads" / f"{Path(str(image_path)).stem}.zip"  # Return the legacy manual-download zip path for the current image.


def build_run_bundle_path(run_folder):  # Define a helper that builds the deterministic current-run zip path for one semi-automatic run folder.
    return Path(str(run_folder)).parent / f"{Path(str(run_folder)).name}.zip"  # Return the sibling zip path that packages the current run folder.


def make_zip_from_folder(folder_path, archive_path):  # Define a helper that packages one folder into one zip file path.
    source_folder = Path(str(folder_path))  # Convert the source folder path into a Path object.
    target_archive = Path(str(archive_path))  # Convert the requested archive path into a Path object.
    target_archive.parent.mkdir(parents=True, exist_ok=True)  # Create the archive parent folder before writing the zip file.
    if target_archive.exists():  # Remove the older archive first so repeated exports replace the previous zip cleanly.
        target_archive.unlink()  # Delete the previous archive so the new zip can be rebuilt without conflicts.
    archive_base = target_archive.with_suffix("")  # Build the archive base path that shutil.make_archive expects without the .zip suffix.
    if Path(str(archive_base) + ".zip").exists():  # Remove the previous make_archive zip when it still exists at the same target location.
        Path(str(archive_base) + ".zip").unlink()  # Delete the previous auto-generated zip so the regenerated archive stays current.
    created_archive = shutil.make_archive(str(archive_base), "zip", root_dir=str(source_folder))  # Package the current source folder into a zip archive.
    return created_archive  # Return the created zip archive path to the caller.


def load_saved_manual_annotations(image_path, output_root):  # Define a helper that reloads saved legacy manual YOLO annotations for one image.
    if image_path is None or str(image_path).strip() == "":  # Return an empty list when there is no current image path.
        return []  # Exit early because there is no current image to load saved annotations for.
    label_path = build_manual_label_path(image_path, output_root)  # Build the current legacy manual YOLO label path.
    if not label_path.exists():  # Return an empty list when no saved legacy manual label file exists yet.
        return []  # Exit early because there are no legacy saved manual annotations to reload.
    with Image.open(image_path) as image:  # Open the current image so normalized YOLO values can be converted back into pixel coordinates.
        image_width, image_height = image.size  # Read the current image width and height once for denormalization.
    loaded_annotations = []  # Create a list that will collect the reloaded legacy manual annotations.
    for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():  # Loop over every YOLO annotation line in the saved legacy manual label file.
        parts = line.strip().split()  # Split the current YOLO annotation line into space-separated tokens.
        if len(parts) != 5:  # Skip malformed lines that do not match the expected one-class YOLO box format.
            continue  # Ignore the current malformed YOLO annotation line and continue with the next one.
        _, center_x, center_y, box_width, box_height = [float(value) for value in parts]  # Convert the current YOLO annotation line into normalized floating-point values.
        pixel_width = box_width * float(image_width)  # Convert the current normalized box width back into original-image pixels.
        pixel_height = box_height * float(image_height)  # Convert the current normalized box height back into original-image pixels.
        pixel_center_x = center_x * float(image_width)  # Convert the current normalized center x coordinate back into original-image pixels.
        pixel_center_y = center_y * float(image_height)  # Convert the current normalized center y coordinate back into original-image pixels.
        left = int(round(pixel_center_x - (pixel_width / 2.0)))  # Compute the current left box edge in original-image pixels.
        top = int(round(pixel_center_y - (pixel_height / 2.0)))  # Compute the current top box edge in original-image pixels.
        right = int(round(pixel_center_x + (pixel_width / 2.0)))  # Compute the current right box edge in original-image pixels.
        bottom = int(round(pixel_center_y + (pixel_height / 2.0)))  # Compute the current bottom box edge in original-image pixels.
        loaded_annotations.append({"bbox": [left, top, right, bottom], "score": 0.0, "point": None})  # Append one reconstructed legacy manual annotation dictionary into the output list.
    return loaded_annotations  # Return the reloaded legacy manual annotation list to the caller.


def save_manual_annotations(image_path, output_root, accepted_annotations, class_name):  # Define a helper that saves accepted manual annotations in the legacy YOLO format.
    if image_path is None or str(image_path).strip() == "":  # Fail early when there is no current image path.
        return "", "No active image is available for saving."  # Return one simple error message when there is no current image to save.
    label_path = build_manual_label_path(image_path, output_root)  # Build the legacy manual YOLO label path for the current image.
    label_path.parent.mkdir(parents=True, exist_ok=True)  # Create the legacy manual YOLO output folder if it does not already exist.
    classes_path = build_manual_classes_path(output_root)  # Build the shared legacy manual classes file path.
    with Image.open(image_path) as image:  # Open the current image so the accepted manual boxes can be normalized into YOLO coordinates.
        image_width, image_height = image.size  # Read the current image width and height once for YOLO normalization.
    with classes_path.open("w", encoding="utf-8") as classes_file:  # Open the legacy classes file for writing.
        classes_file.write(f"{class_name}\n")  # Write the current manual class name into the classes file.
    with label_path.open("w", encoding="utf-8") as label_file:  # Open the current legacy manual YOLO label file for writing.
        for annotation in accepted_annotations:  # Loop over every accepted manual annotation for the current image.
            left, top, right, bottom = annotation["bbox"]  # Read the current accepted manual bounding box.
            box_width = max(0.0, float(right - left))  # Compute the current manual box width in pixels.
            box_height = max(0.0, float(bottom - top))  # Compute the current manual box height in pixels.
            center_x = float(left) + (box_width / 2.0)  # Compute the current manual box center x coordinate in pixels.
            center_y = float(top) + (box_height / 2.0)  # Compute the current manual box center y coordinate in pixels.
            normalized_center_x = center_x / float(image_width) if image_width else 0.0  # Normalize the current center x coordinate into YOLO space.
            normalized_center_y = center_y / float(image_height) if image_height else 0.0  # Normalize the current center y coordinate into YOLO space.
            normalized_width = box_width / float(image_width) if image_width else 0.0  # Normalize the current box width into YOLO space.
            normalized_height = box_height / float(image_height) if image_height else 0.0  # Normalize the current box height into YOLO space.
            label_file.write(f"0 {normalized_center_x:.6f} {normalized_center_y:.6f} {normalized_width:.6f} {normalized_height:.6f}\n")  # Write one legacy YOLO box line for the current accepted manual annotation.
    return str(label_path), f"Saved manual annotations to {label_path}"  # Return the legacy label path plus one short success message.


def build_yolo_export_stem(image_path):  # Define a helper that builds the plain YOLO export stem from the original filename so later manual correction can overwrite the same image cleanly.
    source_path = Path(str(image_path)).resolve()  # Convert the incoming image path into one resolved absolute Path object before generating the export name.
    safe_stem = source_path.stem.replace(" ", "_")  # Replace spaces in the original image stem so the exported dataset filenames stay shell-friendly.
    return safe_stem  # Return the plain original image stem so later exports for the same image can overwrite earlier ones.


def export_yolo_only(image_path, run_folder, output_root):  # Define a helper that exports one processed image into standard YOLO-style images and labels folders.
    source_image_path = Path(image_path)  # Convert the incoming source image path into a Path object for safe path operations.
    source_run_folder = Path(run_folder)  # Convert the incoming run folder path into a Path object so generated labels can be located safely.
    target_root = Path(output_root)  # Convert the incoming export root into a Path object for folder creation and file writing.
    images_dir = target_root / "images"  # Build the standard YOLO images folder path under the requested export root.
    labels_dir = target_root / "labels"  # Build the standard YOLO labels folder path under the requested export root.
    images_dir.mkdir(parents=True, exist_ok=True)  # Create the YOLO images folder when it does not already exist.
    labels_dir.mkdir(parents=True, exist_ok=True)  # Create the YOLO labels folder when it does not already exist.
    export_stem = build_yolo_export_stem(source_image_path)  # Build the overwrite-friendly export stem from the original image filename.
    target_image_path = images_dir / f"{export_stem}{source_image_path.suffix.lower()}"  # Build the exported image path under the YOLO images folder.
    shutil.copy2(source_image_path, target_image_path)  # Copy the current source image into the YOLO images folder.
    label_name = source_image_path.stem + ".txt"  # Build the expected generated label filename for the current source image.
    candidate_labels = list(source_run_folder.rglob(label_name))  # Collect the generated YOLO label candidates under the current run folder.
    target_label_path = labels_dir / f"{export_stem}.txt"  # Build the exported label path under the YOLO labels folder.
    if not candidate_labels:  # Create one empty label file when the current source image produced no detections.
        target_label_path.write_text("", encoding="utf-8")  # Write one empty YOLO label file so images and labels still stay aligned one-to-one.
        return target_image_path, target_label_path  # Return the exported image path and the empty exported label path to the caller.
    source_label_path = candidate_labels[0]  # Pick the first generated YOLO label file found for the current image.
    shutil.copy2(source_label_path, target_label_path)  # Copy the generated YOLO label into the standard YOLO labels folder.
    return target_image_path, target_label_path  # Return the exported image path and label path to the caller.


def normalize_class_name(class_name):  # Define a helper that turns the current class name into one safe reusable dataset token.
    normalized_name = str(class_name).strip().replace(" ", "_").replace("/", "_").replace("\\", "_")  # Normalize the current class name so dataset contents stay shell-friendly.
    return normalized_name or "target_object"  # Return the normalized class name and fall back to one default token when the current name is empty.


def get_export_format_folder_name(export_format):  # Define a helper that maps one export-format label onto one stable dataset folder name.
    folder_map = {EXPORT_FORMAT_YOLO_BOX: "yolo_detection_dataset", EXPORT_FORMAT_YOLO_SEG: "yolo_segmentation_dataset", EXPORT_FORMAT_COCO_BOX: "coco_detection_dataset", EXPORT_FORMAT_COCO_SEG: "coco_instance_segmentation_dataset"}  # Build the mapping from user-facing export-format labels to filesystem-safe dataset folder names.
    return folder_map.get(export_format, "yolo_detection_dataset")  # Return the mapped dataset folder name and default back to YOLO detection when the current label is unknown.


def build_export_dataset_root(output_root, export_format):  # Define a helper that builds the dataset root folder for the currently selected export format.
    normalized_root = normalize_output_root(str(output_root))  # Normalize the current output-root text into one stable absolute export root path.
    return normalized_root / get_export_format_folder_name(export_format)  # Return the format-specific dataset root path under the normalized output root.


def build_manual_state_path(image_path, output_root):  # Define a helper that builds the reusable internal manual-state path for the current image.
    export_stem = build_yolo_export_stem(image_path)  # Build the current overwrite-friendly export stem from the original image filename.
    return normalize_output_root(str(output_root)) / "manual_state" / f"{export_stem}.json"  # Return the deterministic internal manual-state JSON path for the current image.


def bbox_to_xywh(bbox):  # Define a helper that converts one xyxy bounding box into one COCO-style xywh box.
    left, top, right, bottom = bbox  # Read the current xyxy bounding box coordinates into local variables.
    return [float(left), float(top), float(max(0, right - left)), float(max(0, bottom - top))]  # Return one COCO-style xywh box derived from the current xyxy bounding box.


def bbox_to_polygon(bbox):  # Define a helper that converts one bounding box into one rectangular polygon fallback.
    left, top, right, bottom = bbox  # Read the current bounding box coordinates into local variables.
    return [float(left), float(top), float(right), float(top), float(right), float(bottom), float(left), float(bottom)]  # Return one clockwise rectangular polygon matching the current bounding box.


def compute_polygon_area(flat_polygon):  # Define a helper that computes one polygon area from one flat x1 y1 x2 y2 list.
    if flat_polygon is None or len(flat_polygon) < 6:  # Fall back to zero area when the current polygon has too few coordinates to form one valid region.
        return 0.0  # Return zero because the current polygon cannot form one valid closed shape.
    point_array = np.asarray(flat_polygon, dtype=np.float32).reshape(-1, 2)  # Convert the current flat polygon list into one Nx2 point array for area computation.
    return float(abs(cv2.contourArea(point_array)))  # Return the absolute contour area of the current polygon point array.


def mask_to_polygons(mask):  # Define a helper that converts one boolean or binary mask into one or more flat segmentation polygons.
    if mask is None:  # Return an empty polygon list when the current mask does not exist.
        return []  # Exit early because there is no mask to convert into polygons.
    mask_array = np.asarray(mask)  # Convert the incoming mask into one NumPy array for consistent downstream handling.
    mask_uint8 = (mask_array.astype(np.uint8) * 255) if mask_array.dtype != np.uint8 else mask_array.copy()  # Convert the current mask into one uint8 binary image suitable for contour extraction.
    contour_result = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Extract the external contours from the current binary mask.
    contours = contour_result[0] if len(contour_result) == 2 else contour_result[1]  # Read the contour list from the OpenCV return value across OpenCV versions.
    polygon_list = []  # Create a list that will collect every valid segmentation polygon extracted from the current mask.
    for contour in contours:  # Loop over every extracted contour from the current binary mask.
        if contour is None or len(contour) < 3:  # Skip the current contour when it does not contain enough points to form one polygon.
            continue  # Ignore the current invalid contour and continue with the next one.
        flat_polygon = contour.reshape(-1, 2).astype(np.float32).flatten().tolist()  # Flatten the current contour into one COCO- and YOLO-friendly x1 y1 x2 y2 polygon list.
        if len(flat_polygon) < 6:  # Skip the current flattened contour when it still cannot form one valid polygon.
            continue  # Ignore the current too-short polygon and continue with the next one.
        polygon_list.append([float(value) for value in flat_polygon])  # Append the current valid polygon into the final polygon list.
    return polygon_list  # Return the full polygon list extracted from the current mask.


def normalize_annotation_polygons(annotation):  # Define a helper that always returns at least one polygon for the current annotation.
    polygon_list = annotation.get("polygons") or []  # Read the current annotation polygon list and fall back to one empty list when none is present.
    valid_polygons = [polygon for polygon in polygon_list if polygon is not None and len(polygon) >= 6]  # Keep only polygons that contain enough coordinates to form one valid region.
    if valid_polygons:  # Return the current saved polygons directly when the annotation already carries valid polygons.
        return valid_polygons  # Reuse the valid polygons already stored on the current annotation.
    return [bbox_to_polygon(annotation["bbox"])]  # Fall back to one rectangular polygon built from the current annotation bbox when no saved polygon exists.


def build_manual_state_payload(class_name, accepted_annotations):  # Define a helper that converts the current in-memory manual annotations into one JSON-safe state payload.
    serialized_annotations = []  # Create a list that will collect every serialized annotation entry for the internal manual-state file.
    for annotation in accepted_annotations or []:  # Loop over every currently accepted manual annotation.
        serialized_annotations.append({"bbox": [int(value) for value in annotation["bbox"]], "score": float(annotation.get("score", 0.0)), "point": [int(value) for value in annotation["point"]] if annotation.get("point") is not None else None, "polygons": [[float(value) for value in polygon] for polygon in normalize_annotation_polygons(annotation)]})  # Append one JSON-safe annotation entry built from the current accepted annotation.
    return {"class_name": normalize_class_name(class_name), "annotations": serialized_annotations}  # Return the final manual-state payload containing the class name plus all serialized annotations.


def save_manual_state(image_path, output_root, class_name, accepted_annotations):  # Define a helper that writes one reusable internal manual-state JSON file for the current image.
    state_path = build_manual_state_path(image_path, output_root)  # Build the current manual-state JSON path for the current image.
    state_path.parent.mkdir(parents=True, exist_ok=True)  # Create the manual-state parent folder when it does not already exist.
    payload = build_manual_state_payload(class_name, accepted_annotations)  # Build the JSON-safe manual-state payload from the current accepted annotations.
    state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")  # Write the current manual-state JSON payload to disk in UTF-8.
    return state_path  # Return the saved manual-state JSON path to the caller.


def load_saved_manual_state(image_path, output_root):  # Define a helper that reloads the internal manual-state JSON for the current image when it exists.
    state_path = build_manual_state_path(image_path, output_root)  # Build the current manual-state JSON path for the current image.
    if not state_path.exists():  # Return nothing when the current manual-state JSON file does not exist yet.
        return None  # Exit early because there is no internal manual-state payload to reload.
    payload = json.loads(state_path.read_text(encoding="utf-8"))  # Load the current manual-state JSON payload from disk.
    payload["annotations"] = payload.get("annotations", [])  # Normalize the current annotations field so later callers can rely on it always existing.
    return payload  # Return the normalized internal manual-state payload to the caller.


def write_classes_file(dataset_root, class_name):  # Define a helper that writes the current single-class classes.txt file for YOLO-style exports.
    dataset_root.mkdir(parents=True, exist_ok=True)  # Create the current dataset root folder when it does not already exist.
    classes_path = dataset_root / "classes.txt"  # Build the classes.txt path under the current dataset root.
    classes_path.write_text(f"{normalize_class_name(class_name)}\n", encoding="utf-8")  # Write the normalized single class name into the classes.txt file.
    return classes_path  # Return the classes.txt path to the caller.


def write_yolo_detection_labels(label_path, image_width, image_height, annotations):  # Define a helper that writes one YOLO detection label file from the current annotation list.
    label_path.parent.mkdir(parents=True, exist_ok=True)  # Create the YOLO label parent folder before writing the current label file.
    with label_path.open("w", encoding="utf-8") as label_file:  # Open the current YOLO detection label file for writing.
        for annotation in annotations:  # Loop over every current annotation that should be exported in YOLO detection format.
            left, top, right, bottom = annotation["bbox"]  # Read the current annotation bounding box coordinates.
            box_width = max(0.0, float(right - left))  # Compute the current bounding-box width in pixels.
            box_height = max(0.0, float(bottom - top))  # Compute the current bounding-box height in pixels.
            center_x = float(left) + (box_width / 2.0)  # Compute the current bounding-box center x coordinate in pixels.
            center_y = float(top) + (box_height / 2.0)  # Compute the current bounding-box center y coordinate in pixels.
            normalized_center_x = center_x / float(image_width) if image_width else 0.0  # Normalize the current center x coordinate into YOLO space.
            normalized_center_y = center_y / float(image_height) if image_height else 0.0  # Normalize the current center y coordinate into YOLO space.
            normalized_width = box_width / float(image_width) if image_width else 0.0  # Normalize the current bounding-box width into YOLO space.
            normalized_height = box_height / float(image_height) if image_height else 0.0  # Normalize the current bounding-box height into YOLO space.
            label_file.write(f"0 {normalized_center_x:.6f} {normalized_center_y:.6f} {normalized_width:.6f} {normalized_height:.6f}\n")  # Write one YOLO detection label line for the current annotation.
    return label_path  # Return the current YOLO detection label path to the caller.


def write_yolo_segmentation_labels(label_path, image_width, image_height, annotations):  # Define a helper that writes one YOLO segmentation label file from the current annotation list.
    label_path.parent.mkdir(parents=True, exist_ok=True)  # Create the YOLO segmentation label parent folder before writing the current label file.
    with label_path.open("w", encoding="utf-8") as label_file:  # Open the current YOLO segmentation label file for writing.
        for annotation in annotations:  # Loop over every current annotation that should be exported in YOLO segmentation format.
            normalized_polygon_values = []  # Create a list that will collect every normalized polygon value for the current annotation.
            for polygon in normalize_annotation_polygons(annotation):  # Loop over every valid polygon for the current annotation.
                point_array = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)  # Convert the current flat polygon list into one Nx2 point array.
                for point_x, point_y in point_array:  # Loop over every point in the current polygon.
                    normalized_polygon_values.append(f"{float(point_x) / float(image_width):.6f}" if image_width else "0.000000")  # Append the normalized x coordinate string for the current polygon point.
                    normalized_polygon_values.append(f"{float(point_y) / float(image_height):.6f}" if image_height else "0.000000")  # Append the normalized y coordinate string for the current polygon point.
            if normalized_polygon_values:  # Write one YOLO segmentation line only when at least one valid polygon point exists.
                label_file.write("0 " + " ".join(normalized_polygon_values) + "\n")  # Write one YOLO segmentation line for the current annotation.
    return label_path  # Return the current YOLO segmentation label path to the caller.


def load_or_initialize_coco_dataset(annotation_path):  # Define a helper that loads one COCO JSON dataset or creates one clean empty structure.
    if annotation_path.exists():  # Load the current COCO JSON dataset from disk when it already exists.
        coco_data = json.loads(annotation_path.read_text(encoding="utf-8"))  # Read and parse the existing COCO JSON dataset file.
    else:  # Build one empty COCO structure when the current annotation file does not exist yet.
        coco_data = {"images": [], "annotations": [], "categories": []}  # Create one minimal empty COCO dataset structure.
    coco_data.setdefault("images", [])  # Ensure the current COCO dataset contains one images list.
    coco_data.setdefault("annotations", [])  # Ensure the current COCO dataset contains one annotations list.
    coco_data.setdefault("categories", [])  # Ensure the current COCO dataset contains one categories list.
    return coco_data  # Return the normalized COCO dataset structure to the caller.


def upsert_coco_records(image_path, dataset_root, class_name, annotations, include_segmentation):  # Define a helper that overwrites one image entry inside one COCO dataset with the current annotations.
    images_dir = dataset_root / "images"  # Build the COCO images folder path under the current dataset root.
    images_dir.mkdir(parents=True, exist_ok=True)  # Create the COCO images folder when it does not already exist.
    annotation_path = dataset_root / "annotations.json"  # Build the COCO annotations JSON path under the current dataset root.
    source_image_path = Path(image_path)  # Convert the incoming image path into a Path object for safe file operations.
    target_image_path = images_dir / f"{build_yolo_export_stem(source_image_path)}{source_image_path.suffix.lower()}"  # Build the copied image path under the COCO images folder.
    shutil.copy2(source_image_path, target_image_path)  # Copy the current source image into the COCO images folder.
    with Image.open(source_image_path) as image:  # Open the current source image so the exported image width and height can be recorded in the COCO dataset.
        image_width, image_height = image.size  # Read the current source image width and height.
    coco_data = load_or_initialize_coco_dataset(annotation_path)  # Load the current COCO dataset or create one empty structure when it does not exist yet.
    normalized_class_name = normalize_class_name(class_name)  # Normalize the current class name before writing it into the COCO categories list.
    if not coco_data["categories"]:  # Initialize the categories list only when the current COCO dataset has no categories yet.
        coco_data["categories"] = [{"id": 1, "name": normalized_class_name, "supercategory": normalized_class_name}]  # Create one single-class COCO categories entry for the current dataset.
    else:  # Keep the category id stable but refresh the category naming when the dataset already exists.
        coco_data["categories"][0]["name"] = normalized_class_name  # Update the first category name to match the current normalized class name.
        coco_data["categories"][0]["supercategory"] = normalized_class_name  # Update the first category supercategory to match the current normalized class name.
    image_file_name = target_image_path.name  # Read the current exported COCO image filename once for image-record matching.
    coco_data["images"] = [image_record for image_record in coco_data["images"] if image_record.get("file_name") != image_file_name]  # Remove any older image record for the current exported image so the new record can overwrite it cleanly.
    existing_image_ids = [int(image_record.get("id", 0)) for image_record in coco_data["images"]]  # Collect the ids of the remaining COCO image records.
    image_id = (max(existing_image_ids) + 1) if existing_image_ids else 1  # Assign the next available COCO image id to the current exported image.
    coco_data["images"].append({"id": image_id, "file_name": image_file_name, "width": int(image_width), "height": int(image_height)})  # Append the new COCO image record for the current exported image.
    coco_data["annotations"] = [annotation_record for annotation_record in coco_data["annotations"] if int(annotation_record.get("image_id", -1)) != image_id]  # Remove any older annotation records linked to the current image id before rewriting them.
    next_annotation_id = max([int(annotation_record.get("id", 0)) for annotation_record in coco_data["annotations"]] + [0]) + 1  # Compute the next available COCO annotation id after the remaining annotations.
    for annotation in annotations:  # Loop over every current annotation that should be exported into COCO format.
        bbox_xywh = bbox_to_xywh(annotation["bbox"])  # Convert the current annotation bbox into one COCO-style xywh box.
        segmentation = [polygon for polygon in normalize_annotation_polygons(annotation)] if include_segmentation else []  # Build the current segmentation polygon list only when segmentation export is enabled.
        area_value = 0.0  # Initialize the current annotation area accumulator.
        if include_segmentation and segmentation:  # Compute polygon-based area when segmentation export is enabled and valid polygons exist.
            area_value = float(sum(compute_polygon_area(polygon) for polygon in segmentation))  # Sum the polygon areas for the current annotation.
        else:  # Fall back to bbox area when segmentation export is disabled or no valid polygons exist.
            area_value = float(bbox_xywh[2] * bbox_xywh[3])  # Compute the current annotation area from the bbox width and height.
        coco_data["annotations"].append({"id": int(next_annotation_id), "image_id": int(image_id), "category_id": 1, "bbox": bbox_xywh, "area": area_value, "iscrowd": 0, "segmentation": segmentation if include_segmentation else []})  # Append the current COCO annotation record into the dataset.
        next_annotation_id += 1  # Advance the COCO annotation id counter after writing the current annotation.
    annotation_path.write_text(json.dumps(coco_data, ensure_ascii=False, indent=2), encoding="utf-8")  # Save the updated COCO dataset back to disk as UTF-8 JSON.
    return {"image_path": target_image_path, "label_path": annotation_path, "classes_path": annotation_path}  # Return the exported image path and the shared COCO JSON path to the caller.


def write_yolo_records(image_path, dataset_root, class_name, annotations, include_segmentation):  # Define a helper that exports one image plus its annotations in standard YOLO folder layout.
    images_dir = dataset_root / "images"  # Build the YOLO images folder under the current dataset root.
    labels_dir = dataset_root / "labels"  # Build the YOLO labels folder under the current dataset root.
    images_dir.mkdir(parents=True, exist_ok=True)  # Create the YOLO images folder when it does not already exist.
    labels_dir.mkdir(parents=True, exist_ok=True)  # Create the YOLO labels folder when it does not already exist.
    source_image_path = Path(image_path)  # Convert the incoming image path into a Path object for safe file operations.
    export_stem = build_yolo_export_stem(source_image_path)  # Build the overwrite-friendly export stem from the original image filename.
    target_image_path = images_dir / f"{export_stem}{source_image_path.suffix.lower()}"  # Build the current exported image path under the YOLO images folder.
    label_path = labels_dir / f"{export_stem}.txt"  # Build the current exported label path under the YOLO labels folder.
    shutil.copy2(source_image_path, target_image_path)  # Copy the current source image into the YOLO images folder.
    with Image.open(source_image_path) as image:  # Open the current source image so annotation coordinates can be normalized using its real width and height.
        image_width, image_height = image.size  # Read the current source image width and height once for normalization.
    classes_path = write_classes_file(dataset_root, class_name)  # Write the current classes.txt file for the selected class name and keep the returned path.
    if include_segmentation:  # Write segmentation labels when the current export format is YOLO segmentation.
        write_yolo_segmentation_labels(label_path, image_width, image_height, annotations)  # Write one YOLO segmentation label file for the current image.
    else:  # Write detection labels when the current export format is YOLO detection.
        write_yolo_detection_labels(label_path, image_width, image_height, annotations)  # Write one YOLO detection label file for the current image.
    return {"image_path": target_image_path, "label_path": label_path, "classes_path": classes_path}  # Return the exported image, label, and classes paths to the caller.


def export_annotations_for_image(image_path, annotations, output_root, class_name, export_format):  # Define a helper that exports one normalized annotation list into the currently selected dataset format.
    dataset_root = build_export_dataset_root(output_root, export_format)  # Build the format-specific dataset root under the selected output root.
    if export_format == EXPORT_FORMAT_YOLO_BOX:  # Export the current annotations in YOLO detection format when the selector requests it.
        export_result = write_yolo_records(image_path, dataset_root, class_name, annotations, include_segmentation=False)  # Write the current image and labels into the YOLO detection folder layout.
    elif export_format == EXPORT_FORMAT_YOLO_SEG:  # Export the current annotations in YOLO segmentation format when the selector requests it.
        export_result = write_yolo_records(image_path, dataset_root, class_name, annotations, include_segmentation=True)  # Write the current image and labels into the YOLO segmentation folder layout.
    elif export_format == EXPORT_FORMAT_COCO_BOX:  # Export the current annotations in COCO detection format when the selector requests it.
        export_result = upsert_coco_records(image_path, dataset_root, class_name, annotations, include_segmentation=False)  # Upsert the current image and bbox annotations into the COCO detection JSON dataset.
    else:  # Export the current annotations in COCO instance-segmentation format for every other supported COCO segmentation request.
        export_result = upsert_coco_records(image_path, dataset_root, class_name, annotations, include_segmentation=True)  # Upsert the current image and polygon annotations into the COCO instance-segmentation JSON dataset.
    export_result["dataset_root"] = dataset_root  # Attach the current dataset root path to the export result payload for later display and packaging.
    export_result["format_name"] = export_format  # Attach the current format label to the export result payload for later display and packaging.
    return export_result  # Return the completed export result payload to the caller.


def build_auto_annotations_from_run(run_folder):  # Define a helper that converts one semi-automatic run folder into one normalized annotation list.
    run_path = Path(run_folder)  # Convert the incoming run folder path into a Path object for safe file operations.
    filtered_csv_path = run_path / "full_image_detections.csv"  # Build the filtered detections CSV path under the current run folder.
    filtered_rows = load_detection_rows(filtered_csv_path) if filtered_csv_path.exists() else []  # Reload the final globally filtered detection rows from the current run folder when they exist.
    normalized_annotations = []  # Create a list that will collect the normalized annotations reconstructed from the current run folder.
    for row in filtered_rows:  # Loop over every final globally filtered detection row from the current run folder.
        bbox = [int(round(row["global_x1"])), int(round(row["global_y1"])), int(round(row["global_x2"])), int(round(row["global_y2"]))]  # Convert the current global floating-point detection box into one integer xyxy bbox.
        mask_path = build_mask_path(run_path, row)  # Rebuild the binary mask path for the current filtered detection under the current run folder.
        polygons = []  # Create a list that will collect the current detection segmentation polygons.
        if mask_path.exists():  # Convert the saved binary mask into polygons only when the current run folder actually contains the expected mask file.
            mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)  # Read the current binary mask from disk as one grayscale image.
            if mask_image is not None:  # Continue only when OpenCV successfully reads the current binary mask file.
                polygons = mask_to_polygons(mask_image > 0)  # Convert the current binary mask into one or more segmentation polygons.
        normalized_annotations.append({"bbox": bbox, "score": float(row["confidence"]), "point": None, "polygons": polygons})  # Append one normalized annotation reconstructed from the current semi-automatic detection row.
    return normalized_annotations  # Return the full normalized annotation list reconstructed from the current run folder.


def mask_to_bbox(mask):  # Define a helper that computes one tight bounding box from one boolean or binary mask.
    mask_array = np.asarray(mask)  # Convert the incoming mask into one NumPy array for coordinate extraction.
    ys, xs = np.where(mask_array > 0)  # Find every positive-mask pixel coordinate in the current mask.
    if len(xs) == 0 or len(ys) == 0:  # Return nothing when the current mask has no positive pixels.
        return None  # Exit early because the current mask cannot produce one valid bounding box.
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]  # Return the tight xyxy bounding box covering every positive mask pixel.


def copy_source_image_to_run(source_image_path, output_root):  # Define a helper that copies one source image into one run folder while preserving its filename.
    source_path = Path(str(source_image_path))  # Convert the current source-image path into a Path object.
    output_root = Path(str(output_root))  # Convert the current run output-root path into a Path object.
    output_root.mkdir(parents=True, exist_ok=True)  # Create the current run output folder if it does not already exist.
    target_path = output_root / source_path.name  # Build the copied-image path inside the current run folder using the original source filename.
    shutil.copy2(source_path, target_path)  # Copy the current source image into the current run folder while preserving its original filename.
    return target_path  # Return the copied-image path to the caller.


def run_grounded_pipeline_for_path(source_image_path, prompt, class_name, output_root_text, device_mode, rows, cols, box_threshold, text_threshold, tile_nms_threshold, global_nms_threshold, max_detections, min_box_area, draw_boxes, draw_labels):  # Define a helper that runs the semi-automatic Grounded-SAM workflow for one source-image path.
    source_path = Path(str(source_image_path))  # Convert the current source-image path into a Path object.
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Build one timestamp string so every semi-automatic run gets its own output folder.
    base_output_root = normalize_output_root(output_root_text)  # Resolve the requested output-root path into one safe absolute folder.
    run_folder = base_output_root / f"{source_path.stem}_{run_stamp}"  # Build the current run folder underneath the requested output-root folder.
    copied_image_path = copy_source_image_to_run(source_path, run_folder)  # Copy the current source image into the current run folder before processing.
    args = build_args(image_path=copied_image_path, prompt=prompt, rows=rows, cols=cols, box_threshold=box_threshold, text_threshold=text_threshold, nms_threshold=tile_nms_threshold, max_detections=max_detections, min_box_area=min_box_area, output_root=run_folder, device_mode=device_mode)  # Build the shared runner namespace from the current settings.
    tile_rows, _ = crop_tiles(image_path=copied_image_path, rows=int(rows), cols=int(cols), output_root=run_folder)  # Generate the requested tile grid for the current source image.
    grounding_model, sam_predictor, device = get_cached_grounded_models(args)  # Reuse or build the heavy Grounded-SAM models for the current run.
    detections_csv_path = run_grounded_sam_on_tiles(tile_rows=tile_rows, args=args, output_root=run_folder, grounding_model=grounding_model, sam_predictor=sam_predictor)  # Run the shared tile Grounded-SAM pipeline over every generated tile.
    raw_rows = load_detection_rows(detections_csv_path)  # Read the raw per-tile detections from the generated detections CSV file.
    filtered_rows = apply_global_nms(raw_rows, global_nms_threshold)  # Apply a second global NMS pass to reduce duplicate boxes across neighboring tiles.
    filtered_csv_path = write_filtered_detections_csv(filtered_rows, run_folder)  # Save the globally filtered detections into a second CSV file.
    yolo_class_name = parse_prompt_to_class_name(prompt, class_name)  # Resolve one YOLO-safe class name from the current prompt and class controls.
    yolo_label_path, yolo_classes_path = write_yolo_files(filtered_rows, copied_image_path, run_folder, yolo_class_name)  # Export the globally filtered detections into YOLO text format inside the current run folder.
    full_overlay_image, full_mask_image, _, _ = build_full_image_outputs(copied_image_path, run_folder, filtered_rows, draw_boxes, draw_labels)  # Build the current full-image overlay and combined mask from the globally filtered detections.
    summary_text = summarize_grounded_results(raw_rows, filtered_rows, rows, cols, prompt, yolo_class_name)  # Build one readable summary from the current Grounded-SAM run.
    status_text = f"Device: {device} | Tiles: {len(tile_rows)} | Raw detections: {len(raw_rows)} | Filtered detections: {len(filtered_rows)} | Output: {run_folder}"  # Build one compact status line for the current Grounded-SAM run.
    run_zip_path = make_zip_from_folder(run_folder, build_run_bundle_path(run_folder))  # Package the full current-image run folder into one zip archive.
    return full_overlay_image, full_mask_image.convert("RGB"), [], [], summary_text, status_text, str(detections_csv_path), str(filtered_csv_path), str(yolo_label_path), str(yolo_classes_path), str(run_zip_path), str(copied_image_path), str(run_folder), None, None, "", "", "", "", ""  # Return the compact result tuple in the positions the desktop UI already expects.
