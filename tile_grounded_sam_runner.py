import argparse  # Import argparse so the script can read command-line arguments.
import csv  # Import csv so the script can save detection and tile metadata tables.
import sys  # Import sys so the script can adjust import paths before loading local packages.
from pathlib import Path  # Import Path so file system paths stay readable and safe.

import cv2  # Import OpenCV so the script can read and write images.
import numpy as np  # Import NumPy so the script can manipulate mask and box arrays.
import supervision as sv  # Import supervision so the script can annotate masks and boxes.
import torch  # Import torch so the script can run NMS and move SAM onto GPU.
import torchvision  # Import torchvision so the script can run box NMS.
from PIL import Image  # Import PIL Image so the script can crop the source image into tiles.

REPO_ROOT = Path(__file__).resolve().parent  # Resolve the repository root from the current script location.
SEGMENT_ANYTHING_REPO = REPO_ROOT / "segment_anything"  # Build the outer segment_anything repository path.
if str(SEGMENT_ANYTHING_REPO) not in sys.path:  # Add the outer SAM repository path to sys.path when it is not already present.
    sys.path.insert(0, str(SEGMENT_ANYTHING_REPO))  # Prepend the SAM repository path so Python finds the real inner package first.

from segment_anything import SamPredictor, sam_model_registry  # Import SAM model helpers for mask prediction after fixing the local package path.

from groundingdino.util.inference import Model  # Import the GroundingDINO inference wrapper used for text-prompt detection.


def resolve_device(device_mode):  # Define a helper that maps a requested device mode into a real runtime device string.
    requested_mode = str(device_mode).strip().lower() if device_mode is not None else "auto"  # Normalize the incoming device mode string and fall back to auto when it is missing.
    if requested_mode == "cpu":  # Force CPU execution when the user explicitly requests CPU mode.
        return "cpu"  # Return the CPU device string immediately.
    if requested_mode == "cuda":  # Try to force CUDA execution when the user explicitly requests GPU mode.
        return "cuda" if torch.cuda.is_available() else "cpu"  # Use CUDA only when it is available and otherwise fall back to CPU safely.
    return "cuda" if torch.cuda.is_available() else "cpu"  # Use auto mode by preferring CUDA and otherwise falling back to CPU.


def build_parser():  # Define a parser builder so runtime options stay organized.
    parser = argparse.ArgumentParser(description="Split an image into tiles and run Grounded-SAM on each tile.")  # Create the top-level parser with a short description.
    parser.add_argument("--image", default="inputs/test.png", help="Source image path.")  # Add the source image path argument and default it to the current test image.
    parser.add_argument("--rows", type=int, default=3, help="Number of tile rows.")  # Add the tile row count argument and default it to 3.
    parser.add_argument("--cols", type=int, default=3, help="Number of tile columns.")  # Add the tile column count argument and default it to 3.
    parser.add_argument("--output-root", default="outputs/test_tiles_grounded_sam", help="Root folder for tile images and outputs.")  # Add the output root argument so all generated files stay grouped together.
    parser.add_argument("--config", default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", help="GroundingDINO config path.")  # Add the GroundingDINO config path argument with the current working default.
    parser.add_argument("--grounding-checkpoint", default="weights/groundingdino_swint_ogc.pth", help="GroundingDINO checkpoint path.")  # Add the GroundingDINO checkpoint path argument with the current working default.
    parser.add_argument("--sam-checkpoint", default="weights/sam_vit_b_01ec64.pth", help="SAM checkpoint path.")  # Add the SAM checkpoint path argument with the current working default.
    parser.add_argument("--sam-encoder-version", default="vit_b", help="SAM encoder version.")  # Add the SAM encoder version argument and default it to vit_b.
    parser.add_argument("--prompt", default="transparent plastic bottle . crushed plastic bottle . pet bottle .", help="Text prompt for GroundingDINO.")  # Add the text prompt argument with a bottle-focused default.
    parser.add_argument("--box-threshold", type=float, default=0.18, help="GroundingDINO box threshold.")  # Add the box threshold argument with the current recommended default.
    parser.add_argument("--text-threshold", type=float, default=0.12, help="GroundingDINO text threshold.")  # Add the text threshold argument with the current recommended default.
    parser.add_argument("--nms-threshold", type=float, default=0.60, help="NMS threshold applied to tile detections.")  # Add the NMS threshold argument so duplicate boxes can be reduced per tile.
    parser.add_argument("--max-detections", type=int, default=10, help="Maximum detections to keep per tile after NMS.")  # Add the per-tile detection cap so SAM does not process too many boxes.
    parser.add_argument("--min-box-area", type=float, default=400.0, help="Minimum box area to keep per tile.")  # Add a minimum box area filter so tiny noisy boxes can be dropped.
    parser.add_argument("--device-mode", default="auto", choices=["auto", "cpu", "cuda"], help="Runtime device mode.")  # Add the runtime device mode argument so callers can force CPU or GPU when needed.
    parser.add_argument("--dry-run", action="store_true", help="Only create tiles and metadata without running DINO or SAM.")  # Add a dry-run switch so the user can inspect tile generation without inference.
    return parser  # Return the configured parser to the caller.


def compute_ranges(length, parts):  # Define a helper that splits one dimension into nearly equal ranges.
    base = length // parts  # Compute the base tile size for this dimension.
    remainder = length % parts  # Compute how many early tiles need one extra pixel.
    ranges = []  # Create a list that will collect the dimension ranges.
    start = 0  # Initialize the running start coordinate.
    for index in range(parts):  # Loop once for each requested tile part.
        extra = 1 if index < remainder else 0  # Distribute any leftover pixels across the first few tiles.
        end = start + base + extra  # Compute the end coordinate for the current part.
        ranges.append((start, end))  # Save the current range tuple.
        start = end  # Advance the running start coordinate to the next part.
    return ranges  # Return the full set of ranges to the caller.


def ensure_paths_exist(args):  # Define a helper that validates the key input paths before inference starts.
    image_path = Path(args.image)  # Convert the source image path into a Path object.
    config_path = Path(args.config)  # Convert the config path into a Path object.
    grounding_checkpoint_path = Path(args.grounding_checkpoint)  # Convert the GroundingDINO checkpoint path into a Path object.
    sam_checkpoint_path = Path(args.sam_checkpoint)  # Convert the SAM checkpoint path into a Path object.
    if not image_path.exists():  # Stop early when the source image cannot be found.
        raise FileNotFoundError(f"Source image not found: {image_path}")  # Raise a clear error for a missing source image.
    if not config_path.exists():  # Stop early when the config file cannot be found.
        raise FileNotFoundError(f"Config file not found: {config_path}")  # Raise a clear error for a missing config file.
    if not grounding_checkpoint_path.exists():  # Stop early when the GroundingDINO checkpoint cannot be found.
        raise FileNotFoundError(f"GroundingDINO checkpoint not found: {grounding_checkpoint_path}")  # Raise a clear error for a missing GroundingDINO checkpoint.
    if not sam_checkpoint_path.exists():  # Stop early when the SAM checkpoint cannot be found.
        raise FileNotFoundError(f"SAM checkpoint not found: {sam_checkpoint_path}")  # Raise a clear error for a missing SAM checkpoint.
    return image_path, config_path, grounding_checkpoint_path, sam_checkpoint_path  # Return the validated paths to the caller.


def crop_tiles(image_path, rows, cols, output_root):  # Define a helper that crops the source image into a grid of tile images.
    tile_input_dir = output_root / "tiles"  # Set the folder that will hold the generated tile images.
    tile_input_dir.mkdir(parents=True, exist_ok=True)  # Create the tile image folder if it does not already exist.
    metadata_path = output_root / "tile_metadata.csv"  # Set the path for the tile metadata CSV file.
    with Image.open(image_path) as image:  # Open the source image inside a context manager.
        width, height = image.size  # Read the source image width and height once for tiling.
        x_ranges = compute_ranges(width, cols)  # Split the width into tile column ranges.
        y_ranges = compute_ranges(height, rows)  # Split the height into tile row ranges.
        tile_rows = []  # Create a list that will hold one metadata dictionary per tile.
        for row_index, (top, bottom) in enumerate(y_ranges):  # Loop over every tile row range.
            for col_index, (left, right) in enumerate(x_ranges):  # Loop over every tile column range.
                tile_name = f"tile_r{row_index}_c{col_index}.png"  # Build a stable file name for the current tile image.
                tile_path = tile_input_dir / tile_name  # Build the full output path for the current tile image.
                tile_image = image.crop((left, top, right, bottom))  # Crop the current tile region from the source image.
                tile_image.save(tile_path)  # Save the current tile image to disk.
                tile_rows.append({  # Append the current tile metadata row into the in-memory list.
                    "tile_name": tile_name,  # Save the tile image file name.
                    "row_index": row_index,  # Save the tile row index.
                    "col_index": col_index,  # Save the tile column index.
                    "left": left,  # Save the left coordinate in the original image.
                    "top": top,  # Save the top coordinate in the original image.
                    "right": right,  # Save the right coordinate in the original image.
                    "bottom": bottom,  # Save the bottom coordinate in the original image.
                    "width": right - left,  # Save the tile width in pixels.
                    "height": bottom - top,  # Save the tile height in pixels.
                    "tile_path": str(tile_path),  # Save the tile image path as text for downstream use.
                })  # Finish appending the current metadata row.
    with metadata_path.open("w", newline="", encoding="utf-8") as csv_file:  # Open the tile metadata CSV file for writing.
        writer = csv.DictWriter(csv_file, fieldnames=list(tile_rows[0].keys()))  # Create a CSV writer using the first tile row as the schema.
        writer.writeheader()  # Write the metadata CSV header row.
        writer.writerows(tile_rows)  # Write all tile metadata rows into the CSV file.
    return tile_rows, metadata_path  # Return the tile metadata list and CSV path to the caller.


def build_models(args, config_path, grounding_checkpoint_path, sam_checkpoint_path):  # Define a helper that builds the GroundingDINO and SAM models once.
    device = resolve_device(getattr(args, "device_mode", "auto"))  # Pick the runtime device by respecting the requested device mode and available hardware.
    grounding_model = Model(  # Build the GroundingDINO wrapper model with the requested config and checkpoint.
        model_config_path=str(config_path),  # Pass the GroundingDINO config path into the wrapper.
        model_checkpoint_path=str(grounding_checkpoint_path),  # Pass the GroundingDINO checkpoint path into the wrapper.
        device=device,  # Ask the wrapper to place the model on the selected device.
    )  # Finish the GroundingDINO wrapper model construction.
    sam = sam_model_registry[args.sam_encoder_version](checkpoint=str(sam_checkpoint_path))  # Build the requested SAM encoder from the selected checkpoint.
    sam.to(device=device)  # Move the SAM model onto the selected device.
    sam_predictor = SamPredictor(sam)  # Build the SAM predictor used for box-prompted mask extraction.
    return grounding_model, sam_predictor, device  # Return both models and the selected device to the caller.


def filter_detections(detections, phrases, nms_threshold, max_detections, min_box_area):  # Define a helper that filters raw DINO detections before SAM sees them.
    if detections.xyxy is None or len(detections.xyxy) == 0:  # Return early when DINO produced no boxes at all.
        return detections, phrases  # Return the original empty detections and phrases unchanged.
    xyxy = detections.xyxy  # Read the raw xyxy boxes from the detections object.
    confidence = detections.confidence  # Read the raw confidence array from the detections object.
    widths = np.maximum(0.0, xyxy[:, 2] - xyxy[:, 0])  # Compute the width of every raw detection box.
    heights = np.maximum(0.0, xyxy[:, 3] - xyxy[:, 1])  # Compute the height of every raw detection box.
    areas = widths * heights  # Compute the area of every raw detection box.
    keep_by_area = areas >= float(min_box_area)  # Keep only the boxes whose area passes the configured threshold.
    xyxy = xyxy[keep_by_area]  # Apply the area filter to the box array.
    confidence = confidence[keep_by_area]  # Apply the area filter to the confidence array.
    filtered_phrases = [phrase for phrase, keep in zip(phrases, keep_by_area) if keep]  # Apply the area filter to the phrase list.
    if len(xyxy) == 0:  # Return early when area filtering removes every box.
        detections.xyxy = xyxy  # Store the empty filtered box array back into the detections object.
        detections.confidence = confidence  # Store the empty filtered confidence array back into the detections object.
        return detections, filtered_phrases  # Return the now-empty detections object and filtered phrase list.
    nms_indices = torchvision.ops.nms(  # Run NMS on the filtered detections to remove duplicate overlapping boxes.
        torch.from_numpy(xyxy),  # Convert the filtered box array into a torch tensor for NMS.
        torch.from_numpy(confidence),  # Convert the filtered confidence array into a torch tensor for NMS.
        float(nms_threshold),  # Pass the configured NMS threshold into torchvision.
    ).cpu().numpy().tolist()  # Convert the kept NMS indices back into a Python list.
    nms_indices = nms_indices[: int(max_detections)]  # Keep only the configured maximum number of detections after NMS.
    detections.xyxy = xyxy[nms_indices]  # Store the final kept box array back into the detections object.
    detections.confidence = confidence[nms_indices]  # Store the final kept confidence array back into the detections object.
    filtered_phrases = [filtered_phrases[index] for index in nms_indices]  # Keep only the phrases associated with the final kept detections.
    return detections, filtered_phrases  # Return the filtered detections object and phrase list to the caller.


def segment_boxes(sam_predictor, image_bgr, xyxy):  # Define a helper that converts DINO boxes into SAM masks.
    if len(xyxy) == 0:  # Return an empty mask list when there are no boxes to segment.
        return []  # Return an empty list because SAM has no work to do.
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # Convert the tile image from BGR into RGB for SAM.
    sam_predictor.set_image(image_rgb)  # Cache the current tile image inside the SAM predictor.
    result_masks = []  # Create a list that will collect the best SAM mask for each box.
    for box in xyxy:  # Loop over every kept detection box.
        masks, scores, _ = sam_predictor.predict(box=box, multimask_output=True)  # Ask SAM for multiple mask candidates for the current box prompt.
        best_index = int(np.argmax(scores))  # Pick the highest-scoring SAM mask candidate for the current box.
        result_masks.append(masks[best_index])  # Append the selected boolean mask into the result list.
    return result_masks  # Return the full list of selected boolean masks to the caller.


def build_labels(phrases, confidence):  # Define a helper that builds readable labels for visual annotation.
    labels = []  # Create a list that will collect one label string per detection.
    for phrase, score in zip(phrases, confidence):  # Loop over every phrase and confidence pair.
        labels.append(f"{phrase} {float(score):0.2f}")  # Format the current phrase and score into a compact label string.
    return labels  # Return the full label list to the caller.


def save_detection_rows(csv_writer, tile_row, detections, phrases):  # Define a helper that writes final detection rows into the CSV output.
    if len(detections.xyxy) == 0:  # Skip CSV writing when the current tile has no detections.
        return  # Exit early because there is nothing to save for this tile.
    for det_index, (xyxy, confidence, phrase) in enumerate(zip(detections.xyxy, detections.confidence, phrases)):  # Loop over every final detection and its phrase.
        csv_writer.writerow({  # Write one detection row into the detections CSV file.
            "tile_name": tile_row["tile_name"],  # Save the tile image file name.
            "row_index": tile_row["row_index"],  # Save the tile row index.
            "col_index": tile_row["col_index"],  # Save the tile column index.
            "tile_left": tile_row["left"],  # Save the tile left coordinate in the original image.
            "tile_top": tile_row["top"],  # Save the tile top coordinate in the original image.
            "tile_right": tile_row["right"],  # Save the tile right coordinate in the original image.
            "tile_bottom": tile_row["bottom"],  # Save the tile bottom coordinate in the original image.
            "detection_index": det_index,  # Save the zero-based detection index inside the tile.
            "phrase": phrase,  # Save the final kept phrase text.
            "confidence": float(confidence),  # Save the final kept confidence score.
            "tile_x1": float(xyxy[0]),  # Save the box left coordinate inside the tile.
            "tile_y1": float(xyxy[1]),  # Save the box top coordinate inside the tile.
            "tile_x2": float(xyxy[2]),  # Save the box right coordinate inside the tile.
            "tile_y2": float(xyxy[3]),  # Save the box bottom coordinate inside the tile.
            "global_x1": float(tile_row["left"] + xyxy[0]),  # Save the box left coordinate mapped back into the original image.
            "global_y1": float(tile_row["top"] + xyxy[1]),  # Save the box top coordinate mapped back into the original image.
            "global_x2": float(tile_row["left"] + xyxy[2]),  # Save the box right coordinate mapped back into the original image.
            "global_y2": float(tile_row["top"] + xyxy[3]),  # Save the box bottom coordinate mapped back into the original image.
        })  # Finish writing the current detection row.


def run_grounded_sam_on_tiles(tile_rows, args, output_root, grounding_model, sam_predictor):  # Define a helper that runs DINO and SAM on every generated tile.
    tile_output_root = output_root / "tile_outputs"  # Set the folder that will hold one output subfolder per tile.
    tile_output_root.mkdir(parents=True, exist_ok=True)  # Create the tile output root folder if needed.
    summary_pred_dir = output_root / "predictions"  # Set the folder that will collect copied overlay images for fast review.
    summary_pred_dir.mkdir(parents=True, exist_ok=True)  # Create the summary prediction folder if needed.
    summary_mask_dir = output_root / "masks"  # Set the folder that will collect one subfolder per tile of binary masks.
    summary_mask_dir.mkdir(parents=True, exist_ok=True)  # Create the summary mask folder if needed.
    detections_csv_path = output_root / "detections.csv"  # Set the CSV path that will store all kept detections across tiles.
    with detections_csv_path.open("w", newline="", encoding="utf-8") as detections_csv_file:  # Open the detections CSV file for writing.
        fieldnames = [  # Define the schema for the detections CSV file.
            "tile_name",  # Save the tile image file name.
            "row_index",  # Save the tile row index.
            "col_index",  # Save the tile column index.
            "tile_left",  # Save the tile left coordinate in the original image.
            "tile_top",  # Save the tile top coordinate in the original image.
            "tile_right",  # Save the tile right coordinate in the original image.
            "tile_bottom",  # Save the tile bottom coordinate in the original image.
            "detection_index",  # Save the detection index within the tile.
            "phrase",  # Save the phrase text associated with the detection.
            "confidence",  # Save the detection confidence score.
            "tile_x1",  # Save the box left coordinate within the tile.
            "tile_y1",  # Save the box top coordinate within the tile.
            "tile_x2",  # Save the box right coordinate within the tile.
            "tile_y2",  # Save the box bottom coordinate within the tile.
            "global_x1",  # Save the box left coordinate within the original image.
            "global_y1",  # Save the box top coordinate within the original image.
            "global_x2",  # Save the box right coordinate within the original image.
            "global_y2",  # Save the box bottom coordinate within the original image.
        ]  # Finish defining the detections CSV field list.
        writer = csv.DictWriter(detections_csv_file, fieldnames=fieldnames)  # Create the detections CSV writer.
        writer.writeheader()  # Write the detections CSV header row.
        for tile_row in tile_rows:  # Loop over every generated tile metadata row.
            tile_path = Path(tile_row["tile_path"])  # Convert the current tile path back into a Path object.
            tile_stem = tile_path.stem  # Read the tile stem so output folders stay short and stable.
            tile_output_dir = tile_output_root / tile_stem  # Build the output folder for the current tile.
            tile_output_dir.mkdir(parents=True, exist_ok=True)  # Create the current tile output folder if needed.
            tile_mask_dir = summary_mask_dir / tile_stem  # Build the output folder that will hold the binary masks for the current tile.
            tile_mask_dir.mkdir(parents=True, exist_ok=True)  # Create the current tile mask folder if needed.
            print(f"[RUN] {tile_stem}")  # Print a short progress line for the current tile.
            if args.dry_run:  # Skip inference entirely when dry-run mode is enabled.
                continue  # Move on to the next tile without running DINO or SAM.
            image_bgr = cv2.imread(str(tile_path))  # Read the current tile image from disk as BGR.
            detections, phrases = grounding_model.predict_with_caption(  # Run GroundingDINO caption-based detection on the current tile.
                image=image_bgr,  # Pass the current tile image into GroundingDINO.
                caption=args.prompt,  # Pass the configured text prompt into GroundingDINO.
                box_threshold=float(args.box_threshold),  # Pass the configured box threshold into GroundingDINO.
                text_threshold=float(args.text_threshold),  # Pass the configured text threshold into GroundingDINO.
            )  # Finish the current GroundingDINO prediction call.
            detections, phrases = filter_detections(  # Filter the raw GroundingDINO detections before SAM sees them.
                detections=detections,  # Pass the raw detections object into the filter helper.
                phrases=phrases,  # Pass the raw phrase list into the filter helper.
                nms_threshold=float(args.nms_threshold),  # Pass the configured NMS threshold into the filter helper.
                max_detections=int(args.max_detections),  # Pass the configured detection cap into the filter helper.
                min_box_area=float(args.min_box_area),  # Pass the configured minimum box area into the filter helper.
            )  # Finish the current filtering step.
            if len(detections.xyxy) > 0:  # Build a dense boolean mask stack only when final detections exist.
                mask_list = segment_boxes(sam_predictor=sam_predictor, image_bgr=image_bgr, xyxy=detections.xyxy)  # Run SAM on the final kept boxes and collect one boolean mask per box.
                detections.mask = np.stack(mask_list).astype(bool)  # Convert the Python mask list into a dense boolean mask tensor that supervision can annotate.
            else:  # Store an empty boolean mask tensor when the current tile has no final detections.
                detections.mask = np.empty((0, image_bgr.shape[0], image_bgr.shape[1]), dtype=bool)  # Create an empty mask tensor with the current tile shape.
            box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)  # Create a box annotator that colors detections by index instead of class id.
            mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)  # Create a mask annotator that colors detections by index instead of class id.
            label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)  # Create a label annotator that colors text by detection index instead of class id.
            labels = build_labels(phrases=phrases, confidence=detections.confidence)  # Build readable annotation labels for the current tile.
            annotated_image = image_bgr.copy()  # Copy the raw tile image so annotation does not modify the source image.
            if len(detections.xyxy) > 0:  # Annotate only when at least one final detection survived filtering.
                annotated_image = mask_annotator.annotate(scene=annotated_image, detections=detections)  # Draw the SAM masks onto the tile image.
                annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)  # Draw the final DINO boxes onto the tile image.
                annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)  # Draw the final DINO labels onto the tile image.
            raw_image_path = tile_output_dir / "raw_image.jpg"  # Build the output path for the raw tile image.
            pred_image_path = tile_output_dir / "grounded_sam_pred.jpg"  # Build the output path for the final Grounded-SAM overlay image.
            cv2.imwrite(str(raw_image_path), image_bgr)  # Save the raw tile image into the current tile output folder.
            cv2.imwrite(str(pred_image_path), annotated_image)  # Save the final Grounded-SAM overlay image into the current tile output folder.
            summary_pred_path = summary_pred_dir / f"{tile_stem}_grounded_sam.jpg"  # Build the summary overlay path for the current tile.
            cv2.imwrite(str(summary_pred_path), annotated_image)  # Save a copied overlay image into the summary prediction folder.
            if len(detections.xyxy) > 0:  # Export one binary mask image per detection only when detections exist.
                for det_index, mask in enumerate(detections.mask):  # Loop over every SAM mask associated with the final kept detections.
                    mask_path = tile_mask_dir / f"{tile_stem}_mask_{det_index:02d}.png"  # Build the binary mask output path for the current detection.
                    mask_image = (np.asarray(mask).astype(np.uint8) * 255)  # Convert the boolean mask into an 8-bit binary image.
                    cv2.imwrite(str(mask_path), mask_image)  # Save the binary mask image into the current tile mask folder.
            save_detection_rows(csv_writer=writer, tile_row=tile_row, detections=detections, phrases=phrases)  # Save the final kept detections into the detections CSV file.
    return detections_csv_path  # Return the detections CSV path to the caller after all tiles have been processed.


def main():  # Define the script entry point so the end-to-end workflow is easy to follow.
    parser = build_parser()  # Build the command-line parser.
    args = parser.parse_args()  # Parse the command-line arguments into a namespace.
    image_path, config_path, grounding_checkpoint_path, sam_checkpoint_path = ensure_paths_exist(args)  # Validate the key input paths before doing any work.
    output_root = Path(args.output_root)  # Convert the output root argument into a Path object.
    output_root.mkdir(parents=True, exist_ok=True)  # Create the output root folder if it does not already exist.
    tile_rows, metadata_path = crop_tiles(image_path=image_path, rows=int(args.rows), cols=int(args.cols), output_root=output_root)  # Split the source image into the requested tile grid and save tile metadata.
    print(f"[INFO] Created {len(tile_rows)} tiles from {image_path}.")  # Print a short summary of the tile generation step.
    print(f"[INFO] Tile metadata saved to {metadata_path}.")  # Print the tile metadata CSV path for later review.
    if args.dry_run:  # Stop after tile generation when dry-run mode is enabled.
        print("[DONE] Dry run finished after tile generation.")  # Print a short dry-run completion message.
        return  # Exit early because inference was intentionally skipped.
    grounding_model, sam_predictor, device = build_models(  # Build the GroundingDINO and SAM models once for the full tile run.
        args=args,  # Pass the parsed arguments into the model builder helper.
        config_path=config_path,  # Pass the validated GroundingDINO config path into the model builder helper.
        grounding_checkpoint_path=grounding_checkpoint_path,  # Pass the validated GroundingDINO checkpoint path into the model builder helper.
        sam_checkpoint_path=sam_checkpoint_path,  # Pass the validated SAM checkpoint path into the model builder helper.
    )  # Finish the model construction step.
    print(f"[INFO] Running models on device: {device}.")  # Print the selected inference device for visibility.
    detections_csv_path = run_grounded_sam_on_tiles(  # Run Grounded-SAM on every generated tile and save outputs.
        tile_rows=tile_rows,  # Pass the generated tile metadata list into the tile runner helper.
        args=args,  # Pass the parsed arguments into the tile runner helper.
        output_root=output_root,  # Pass the output root path into the tile runner helper.
        grounding_model=grounding_model,  # Pass the built GroundingDINO model into the tile runner helper.
        sam_predictor=sam_predictor,  # Pass the built SAM predictor into the tile runner helper.
    )  # Finish the full tile inference step.
    print(f"[DONE] Grounded-SAM overlays saved under {output_root / 'predictions'}.")  # Print the summary overlay folder after the full run.
    print(f"[DONE] Binary masks saved under {output_root / 'masks'}.")  # Print the binary mask folder after the full run.
    print(f"[DONE] Detection table saved to {detections_csv_path}.")  # Print the detections CSV path after the full run.


if __name__ == "__main__":  # Run the main workflow only when this file is executed directly.
    main()  # Start the tile generation and Grounded-SAM inference workflow.
