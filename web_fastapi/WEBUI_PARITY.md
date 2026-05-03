# WebUI Parity Notes

This file tracks the FastAPI web version against the Qt desktop version.

## Implemented

- Dataset loading from a server-side folder path.
- Browser upload for multiple images or a folder.
- Separate browser inputs for selecting a few images and selecting an entire folder.
- Image list and previous/next navigation.
- Remove-current-image action from the active browser dataset.
- Canvas image preview with mouse-wheel zoom.
- Canvas bbox editing with empty-area drag-to-create, selection, move, corner resize, edge resize, and numeric coordinate edits.
- Separate workflow modes for Manual SAM and Semi-Automatic Grounded-SAM.
- Manual SAM point prompt endpoint.
- Manual SAM candidate acceptance before export.
- Semi-automatic Grounded-SAM run for the current image.
- Batch semi-automatic run from the browser by processing images sequentially.
- Export format selection shared with the desktop backend.
- YOLO detection export.
- YOLO segmentation export.
- COCO detection JSON export.
- COCO instance segmentation JSON export.
- Dataset ZIP download through tokenized download links.
- Explicit download buttons for current-run ZIP and dataset ZIP.
- Docker entry through `Dockerfile.fastapi`.

## Desktop Features Matched

- Shared model backend through `annotation_backend.py`.
- Shared output folder behavior.
- Shared device mode options: `auto`, `cpu`, and `cuda`.
- Shared prompt, class name, tiling, threshold, NMS, max detections, and min-area settings.
- Shared overwrite-friendly dataset output filenames.
- Shared export folder layouts under `images/`, `labels/`, `classes.txt`, or COCO `annotations.json`.

## Known Gaps

- Manual SAM candidate currently displays the accepted bbox and not the full translucent mask overlay.
- Batch progress is browser-side sequential status, not a server-side job queue.
- Manual mode uses click-to-prompt; panning is currently more comfortable in semi-auto mode.
- There is no account system or multi-user persistent database.
- There is no polygon vertex editor yet; segmentation export uses SAM polygons or bbox fallback polygons.

## Smoke Tests Run

- `python -m py_compile web_fastapi/app.py annotation_backend.py`
- Imported `web_fastapi.app:app` successfully.
- Started `uvicorn` and called `/api/config`.
- Created a dummy image dataset with FastAPI `TestClient`.
- Loaded the dataset through `/api/dataset/from-folder`.
- Served the image through `/api/image/{session_id}/{image_id}`.
- Exported one edited bbox through `/api/export/image`.
- Downloaded the generated dataset ZIP through `/api/download/{token}`.

## External Tool Comparison Targets

- CVAT: browser annotation workflows, bbox/polygon editing, and YOLO/COCO style export.
- Label Studio: image annotation, ML-assisted workflows, and export formats.
- Roboflow Annotate: browser annotation, dataset organization, and export-oriented workflow.

## Reference Pages Checked

- https://docs.cvat.ai/docs/dataset_management/formats/
- https://labelstud.io/guide/export.html
- https://docs.roboflow.com/datasets/dataset-versions/exporting-data
