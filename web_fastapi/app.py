from pathlib import Path  # Use Path for safe cross-platform filesystem handling.
import shutil  # Use shutil for copying uploaded files into isolated session folders.
import sys  # Use sys so this web package can import the repository-level backend module.
from uuid import uuid4  # Use uuid4 to create collision-resistant session identifiers.

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # Import FastAPI primitives for the web API.
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware so local frontend requests stay simple.
from fastapi.responses import FileResponse  # Import FileResponse so images and zip files can be downloaded directly.
from fastapi.responses import HTMLResponse  # Import HTMLResponse so the root page can be served with explicit no-cache headers.
from fastapi.responses import RedirectResponse  # Import RedirectResponse so root can force users onto the hard-reset route.
from pydantic import BaseModel, Field  # Import Pydantic models for typed JSON request bodies.
import cv2  # Import OpenCV so manual SAM endpoint can read images as RGB arrays.
import numpy as np  # Import NumPy so manual SAM endpoint can build point prompt arrays.

REPO_ROOT = Path(__file__).resolve().parents[1]  # Resolve the project root one level above this web_fastapi folder.
if str(REPO_ROOT) not in sys.path:  # Ensure repository modules are importable when uvicorn starts inside web_fastapi.
    sys.path.insert(0, str(REPO_ROOT))  # Prepend the repository root so annotation_backend resolves correctly.

import annotation_backend as backend  # Import the shared Grounded-SAM and export backend used by desktop tools.

APP_TITLE = "Semi-Automatic Annotation WebUI"  # Define the API title shown in generated docs.
APP_VERSION = "webui-20260503-14-clean-header"  # Define a visible frontend build version so stale browser pages are easy to identify.
WEB_ROOT = Path(__file__).resolve().parent  # Resolve the folder that contains this FastAPI app.
STATIC_ROOT = WEB_ROOT / "static"  # Resolve the static frontend folder served by FastAPI.
SESSION_ROOT = backend.DEFAULT_OUTPUT_ROOT / "fastapi_web_sessions"  # Store web sessions under the existing output root.
SUPPORTED_IMAGE_SUFFIXES = backend.SUPPORTED_IMAGE_SUFFIXES  # Reuse the backend-supported image suffix set.
SUPPORTED_VIDEO_SUFFIXES = {
    ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"
}
SESSIONS = {}  # Keep lightweight in-memory session metadata for this single-process local tool.
DOWNLOADS = {}  # Keep generated download paths behind opaque tokens instead of exposing arbitrary filesystem paths.

app = FastAPI(title=APP_TITLE)  # Create the FastAPI application instance.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])  # Allow local browser access without CORS friction.


@app.middleware("http")  # Register middleware that prevents stale browser cache during active UI development.
async def add_no_cache_headers(request, call_next):  # Define middleware that adds no-cache headers to every response.
    response = await call_next(request)  # Let the requested route produce its normal response first.
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"  # Tell browsers not to reuse old HTML or JavaScript.
    response.headers["Pragma"] = "no-cache"  # Add a legacy no-cache header for older clients.
    response.headers["Expires"] = "0"  # Add an immediate expiry header.
    return response  # Return the response with cache-prevention headers.


class FolderLoadRequest(BaseModel):  # Define the request shape for loading a server-side image folder.
    folder_path: str = Field(default="", description="Server-side folder path containing images.")  # Store the folder path typed by the user.


class AutoRunRequest(BaseModel):  # Define the request shape for one semi-automatic Grounded-SAM run.
    session_id: str  # Store the active browser session id.
    image_id: str  # Store the selected image id inside the session.
    prompt: str = backend.DEFAULT_PROMPT  # Store the GroundingDINO text prompt.
    class_name: str = backend.DEFAULT_CLASS_NAME  # Store the export class name.
    output_root: str = str(backend.DEFAULT_OUTPUT_ROOT / "fastapi_web_runs")  # Store the output root for run folders.
    export_format: str = backend.EXPORT_FORMAT_YOLO_BOX  # Store the selected export format.
    device_mode: str = "auto"  # Store the requested device mode.
    rows: int = 3  # Store the tile row count.
    cols: int = 3  # Store the tile column count.
    box_threshold: float = 0.18  # Store the GroundingDINO box threshold.
    text_threshold: float = 0.18  # Store the GroundingDINO text threshold.
    tile_nms_threshold: float = 0.6  # Store the tile-level NMS threshold.
    global_nms_threshold: float = 0.5  # Store the full-image NMS threshold.
    max_detections: int = 20  # Store the maximum detections per tile.
    min_box_area: float = 50.0  # Store the minimum accepted bbox area.
    draw_boxes: bool = True  # Store whether backend overlays should draw boxes.
    draw_labels: bool = True  # Store whether backend overlays should draw labels.


class ExportRequest(BaseModel):  # Define the request shape for saving edited annotations.
    session_id: str  # Store the active browser session id.
    image_id: str  # Store the selected image id inside the session.
    annotations: list[dict] = Field(default_factory=list)  # Store the edited annotation objects from the browser.
    class_name: str = backend.DEFAULT_CLASS_NAME  # Store the requested class name.
    output_root: str = str(backend.DEFAULT_OUTPUT_ROOT / "fastapi_web_runs")  # Store the output root for exported datasets.
    export_format: str = backend.EXPORT_FORMAT_YOLO_BOX  # Store the selected export format.


class ManualCandidateRequest(BaseModel):  # Define the request shape for one manual SAM point prompt.
    session_id: str | None = None
    image_id: str | None = None
    sessionId: str | None = None
    imageId: str | None = None
    x: float
    y: float
    device_mode: str = "auto"
    deviceMode: str | None = None


def ensure_session_root() -> None:  # Define a helper that creates the session root before session writes.
    SESSION_ROOT.mkdir(parents=True, exist_ok=True)  # Create the session root and parents when missing.


def is_supported_image_path(path: Path) -> bool:  # Define a helper that checks whether a path is an image supported by the backend.
    return path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES  # Return true only for files with supported suffixes.


def normalize_annotation(annotation: dict) -> dict:  # Define a helper that normalizes browser annotation objects before export.
    bbox = [int(round(float(value))) for value in annotation.get("bbox", [0, 0, 0, 0])]  # Convert bbox values into integer xyxy coordinates.
    polygons = annotation.get("polygons") or [backend.bbox_to_polygon(bbox)]  # Keep existing polygons or fall back to a rectangular bbox polygon.
    return {"bbox": bbox, "score": float(annotation.get("score", 1.0)), "point": annotation.get("point"), "polygons": polygons}  # Return one backend-compatible annotation record.


def cv2_read_rgb(image_path: str) -> np.ndarray:  # Define a helper that loads one image path as an RGB NumPy array.
    image_bgr = cv2.imread(str(image_path))  # Read the image from disk using OpenCV's BGR convention.
    if image_bgr is None:  # Fail when OpenCV cannot read the requested image.
        raise HTTPException(status_code=422, detail="Image could not be read.")  # Return a clear API error for unreadable images.
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # Convert BGR into RGB for SAM.


def np_array_point(point_x: int, point_y: int) -> np.ndarray:  # Define a helper that builds a SAM point coordinate array.
    return np.array([[int(point_x), int(point_y)]], dtype=np.float32)  # Return one Nx2 foreground point coordinate array.


def np_array_label() -> np.ndarray:  # Define a helper that builds a SAM foreground label array.
    return np.array([1], dtype=np.int32)  # Return one SAM foreground point label.


def make_public_image_record(session_id: str, image_path: Path) -> dict:  # Define a helper that converts an image path into a browser-facing metadata record.
    image_id = image_path.stem + "_" + uuid4().hex[:8]  # Build a short id that avoids collisions for duplicate filenames.
    return {"id": image_id, "name": image_path.name, "path": str(image_path), "url": f"/api/image/{session_id}/{image_id}"}  # Return the browser-safe image record.


def extract_video_frames_to_images(
    video_path: Path,
    output_folder: Path,
    frame_stride: int,
    max_frames: int = 0,
) -> list[Path]:
    """
    從影片中每 frame_stride 幀擷取一張圖片，存成 jpg 後回傳圖片路徑清單。

    例如：
    frame_stride = 30
    代表第 0, 30, 60, 90... 幀會被擷取出來。
    """

    frame_stride = max(1, int(frame_stride))
    max_frames = max(0, int(max_frames))

    output_folder.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Video could not be opened.")

    saved_paths = []
    frame_index = 0
    saved_index = 0

    video_stem = video_path.stem

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        if frame_index % frame_stride == 0:
            image_path = output_folder / f"{video_stem}_frame_{frame_index:06d}.jpg"

            success = cv2.imwrite(str(image_path), frame)

            if success:
                saved_paths.append(image_path)
                saved_index += 1

            if max_frames > 0 and saved_index >= max_frames:
                break

        frame_index += 1

    cap.release()

    if not saved_paths:
        raise HTTPException(status_code=400, detail="No frames were extracted from the video.")

    return saved_paths


def create_session_from_paths(image_paths: list[Path]) -> dict:  # Define a helper that creates one isolated web session from existing image paths.
    ensure_session_root()  # Ensure the session root exists before creating a new session folder.
    session_id = uuid4().hex  # Generate one unique session id for this browser dataset.
    session_folder = SESSION_ROOT / session_id  # Build the isolated session folder path.
    images_folder = session_folder / "images"  # Build the folder where source images are copied.
    images_folder.mkdir(parents=True, exist_ok=True)  # Create the copied-image folder.
    images = []  # Create the list that will store public image metadata.
    for index, source_path in enumerate(image_paths):  # Loop over every discovered or uploaded source image.
        target_path = images_folder / f"{index:05d}_{source_path.name}"  # Build a deterministic copied image filename.
        shutil.copy2(source_path, target_path)  # Copy the source image into the session folder.
        images.append(make_public_image_record(session_id, target_path))  # Append one public image metadata record.
    session_payload = {"id": session_id, "folder": str(session_folder), "images": images, "runs": {}}  # Build the in-memory session payload.
    SESSIONS[session_id] = session_payload  # Store the session payload in memory for later API calls.
    return session_payload  # Return the new session payload.


def get_session(session_id: str) -> dict:  # Define a helper that returns a session or fails with an API error.
    if session_id not in SESSIONS:  # Reject missing or expired session ids.
        raise HTTPException(status_code=404, detail="Session not found.")  # Return a 404 response for unknown sessions.
    return SESSIONS[session_id]  # Return the active session payload.


def get_image_record(session_id: str, image_id: str) -> dict:  # Define a helper that returns one image record or fails with an API error.
    session_payload = get_session(session_id)  # Load the active session payload.
    for image_record in session_payload["images"]:  # Loop through every image registered in this session.
        if image_record["id"] == image_id:  # Match the requested image id.
            return image_record  # Return the matching image record.
    raise HTTPException(status_code=404, detail="Image not found.")  # Return a 404 response when the image id is not registered.


def package_dataset(output_root: str, export_format: str) -> str:  # Define a helper that zips the current format-specific dataset folder.
    dataset_root = backend.build_export_dataset_root(output_root, export_format)  # Resolve the dataset root for the requested export format.
    archive_path = dataset_root.parent / f"{dataset_root.name}.zip"  # Build the zip path next to the dataset folder.
    return str(backend.make_zip_from_folder(dataset_root, archive_path)) if dataset_root.exists() else ""  # Return the new zip path only when the dataset folder exists.


def build_download_url(file_path: str) -> str:  # Define a helper that converts a filesystem path into one download endpoint URL.
    if not file_path:  # Return no URL when there is no generated file path.
        return ""  # Exit early with an empty download URL.
    file_path = Path(file_path).resolve()  # Resolve the generated file path before registering it.
    token = uuid4().hex  # Generate an opaque download token for the browser.
    DOWNLOADS[token] = str(file_path)  # Store the token-to-path mapping in memory.
    return f"/api/download/{token}"  # Return a tokenized download URL instead of a raw filesystem path.


@app.get("/api/config")  # Register the configuration endpoint used by the browser UI.
def get_config() -> dict:  # Define the configuration endpoint implementation.
    return {"version": APP_VERSION, "export_formats": backend.EXPORT_FORMAT_OPTIONS, "defaults": {"prompt": backend.DEFAULT_PROMPT, "class_name": backend.DEFAULT_CLASS_NAME, "output_root": str(backend.DEFAULT_OUTPUT_ROOT / "fastapi_web_runs")}}  # Return frontend defaults, version, and export formats.


@app.get("/", response_class=HTMLResponse)  # Register an explicit root route before the static fallback route.
def get_index() -> RedirectResponse:  # Define the root HTML response implementation.
    return RedirectResponse(url="/hard-reset-webui-20260503-14", status_code=307, headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0", "Pragma": "no-cache", "Expires": "0"})  # Force root users onto the hard-reset route.


@app.get("/webui", response_class=HTMLResponse)  # Register a fresh non-root WebUI route that avoids stale cached root HTML.
def get_webui() -> HTMLResponse:  # Define the fresh WebUI HTML response implementation.
    index_path = STATIC_ROOT / "index.html"  # Build the frontend index path.
    html_text = index_path.read_text(encoding="utf-8")  # Read the current frontend HTML from disk on every request.
    return HTMLResponse(content=html_text, headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0", "Pragma": "no-cache", "Expires": "0"})  # Return current HTML with no-cache headers.


@app.get("/webui/{build_id}", response_class=HTMLResponse)  # Register a versioned WebUI route for cache-busting by path.
def get_versioned_webui(build_id: str) -> HTMLResponse:  # Define the versioned WebUI HTML response implementation.
    index_path = STATIC_ROOT / "index.html"  # Build the frontend index path.
    html_text = index_path.read_text(encoding="utf-8")  # Read the current frontend HTML from disk on every request.
    return HTMLResponse(content=html_text, headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0", "Pragma": "no-cache", "Expires": "0"})  # Return current HTML with no-cache headers.


@app.get("/hard-reset-webui-20260503-08", response_class=HTMLResponse)  # Register a never-before-used hard-reset WebUI route.
def get_hard_reset_webui() -> HTMLResponse:  # Define the hard-reset WebUI HTML response implementation.
    index_path = STATIC_ROOT / "index.html"  # Build the frontend index path.
    html_text = index_path.read_text(encoding="utf-8")  # Read the current frontend HTML from disk on every request.
    return HTMLResponse(content=html_text, headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0", "Pragma": "no-cache", "Expires": "0"})  # Return current HTML with no-cache headers.


@app.get("/hard-reset-webui-20260503-09", response_class=HTMLResponse)  # Register the floating-toolbar WebUI route.
def get_floating_toolbar_webui() -> HTMLResponse:  # Define the floating-toolbar WebUI HTML response implementation.
    index_path = STATIC_ROOT / "index.html"  # Build the frontend index path.
    html_text = index_path.read_text(encoding="utf-8")  # Read the current frontend HTML from disk on every request.
    return HTMLResponse(content=html_text, headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0", "Pragma": "no-cache", "Expires": "0"})  # Return current HTML with no-cache headers.


@app.get("/hard-reset-webui-20260503-10", response_class=HTMLResponse)  # Register the semi-auto download WebUI route.
def get_auto_download_webui() -> HTMLResponse:  # Define the semi-auto download WebUI HTML response implementation.
    index_path = STATIC_ROOT / "index.html"  # Build the frontend index path.
    html_text = index_path.read_text(encoding="utf-8")  # Read the current frontend HTML from disk on every request.
    return HTMLResponse(content=html_text, headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0", "Pragma": "no-cache", "Expires": "0"})  # Return current HTML with no-cache headers.


@app.get("/hard-reset-webui-20260503-11", response_class=HTMLResponse)  # Register the accept-candidate toolbar WebUI route.
def get_accept_toolbar_webui() -> HTMLResponse:  # Define the accept-candidate toolbar WebUI HTML response implementation.
    index_path = STATIC_ROOT / "index.html"  # Build the frontend index path.
    html_text = index_path.read_text(encoding="utf-8")  # Read the current frontend HTML from disk on every request.
    return HTMLResponse(content=html_text, headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0", "Pragma": "no-cache", "Expires": "0"})  # Return current HTML with no-cache headers.


@app.get("/hard-reset-webui-20260503-12", response_class=HTMLResponse)  # Register the toolbar color WebUI route.
def get_toolbar_color_webui() -> HTMLResponse:  # Define the toolbar color WebUI HTML response implementation.
    index_path = STATIC_ROOT / "index.html"  # Build the frontend index path.
    html_text = index_path.read_text(encoding="utf-8")  # Read the current frontend HTML from disk on every request.
    return HTMLResponse(content=html_text, headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0", "Pragma": "no-cache", "Expires": "0"})  # Return current HTML with no-cache headers.


@app.get("/hard-reset-webui-20260503-13", response_class=HTMLResponse)  # Register the clean manual panel WebUI route.
def get_clean_manual_panel_webui() -> HTMLResponse:  # Define the clean manual panel WebUI HTML response implementation.
    index_path = STATIC_ROOT / "index.html"  # Build the frontend index path.
    html_text = index_path.read_text(encoding="utf-8")  # Read the current frontend HTML from disk on every request.
    return HTMLResponse(content=html_text, headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0", "Pragma": "no-cache", "Expires": "0"})  # Return current HTML with no-cache headers.


@app.get("/hard-reset-webui-20260503-14", response_class=HTMLResponse)  # Register the clean header WebUI route.
def get_clean_header_webui() -> HTMLResponse:  # Define the clean header WebUI HTML response implementation.
    index_path = STATIC_ROOT / "index.html"  # Build the frontend index path.
    html_text = index_path.read_text(encoding="utf-8")  # Read the current frontend HTML from disk on every request.
    return HTMLResponse(content=html_text, headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0", "Pragma": "no-cache", "Expires": "0"})  # Return current HTML with no-cache headers.


@app.post("/api/dataset/from-folder")  # Register the server-side folder loading endpoint.
def load_dataset_from_folder(request: FolderLoadRequest) -> dict:  # Define the folder loading endpoint implementation.
    folder_path = Path(request.folder_path).expanduser().resolve()  # Resolve the user-provided folder path on the server machine.
    if not folder_path.exists() or not folder_path.is_dir():  # Validate that the requested folder exists and is a directory.
        raise HTTPException(status_code=400, detail="Folder does not exist on the server.")  # Return a clear API error for invalid folders.
    image_paths = sorted(path for path in folder_path.rglob("*") if is_supported_image_path(path))  # Recursively collect supported image files.
    if not image_paths:  # Reject folders that contain no supported images.
        raise HTTPException(status_code=400, detail="No supported images found in the folder.")  # Return a clear API error for empty image folders.
    return create_session_from_paths(image_paths)  # Create and return a new dataset session.


@app.post("/api/dataset/upload")  # Register the browser upload endpoint.
async def upload_dataset(files: list[UploadFile] = File(...)) -> dict:  # Define the upload endpoint implementation.
    ensure_session_root()  # Ensure the root output folder exists before writing uploads.
    upload_session = SESSION_ROOT / ("upload_" + uuid4().hex)  # Build a temporary folder for raw uploaded files.
    upload_session.mkdir(parents=True, exist_ok=True)  # Create the temporary upload folder.
    copied_paths = []  # Create the list that will collect saved upload paths.
    for upload_file in files:  # Loop over every browser-uploaded file.
        original_name = Path(upload_file.filename or "upload.jpg").name  # Normalize the uploaded filename to avoid nested paths.
        if Path(original_name).suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:  # Skip files that are not supported images.
            continue  # Ignore the current non-image upload and continue.
        target_path = upload_session / f"{len(copied_paths):05d}_{original_name}"  # Build a deterministic upload save path.
        target_path.write_bytes(await upload_file.read())  # Save the uploaded file bytes to disk.
        copied_paths.append(target_path)  # Append the saved upload path for session creation.
    if not copied_paths:  # Reject uploads that contained no supported images.
        raise HTTPException(status_code=400, detail="No supported images were uploaded.")  # Return a clear API error for empty uploads.
    return create_session_from_paths(copied_paths)  # Create and return a dataset session from saved uploads.


@app.post("/api/dataset/video-frames")
async def upload_video_and_extract_frames(
    video: UploadFile = File(...),
    frame_stride: int = Form(30),
    max_frames: int = Form(0),
) -> dict:
    """
    上傳影片並依照 frame_stride 抽幀，抽出的圖片會直接建立成 WebUI session。
    """

    ensure_session_root()

    original_name = Path(video.filename or "uploaded_video.mp4").name
    suffix = Path(original_name).suffix.lower()

    if suffix not in SUPPORTED_VIDEO_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format: {suffix}",
        )

    if frame_stride < 1:
        raise HTTPException(
            status_code=400,
            detail="frame_stride must be >= 1.",
        )

    video_session = SESSION_ROOT / ("video_" + uuid4().hex)
    video_session.mkdir(parents=True, exist_ok=True)

    video_path = video_session / f"input_video{suffix}"
    video_path.write_bytes(await video.read())

    frames_folder = video_session / "extracted_frames"

    extracted_paths = extract_video_frames_to_images(
        video_path=video_path,
        output_folder=frames_folder,
        frame_stride=frame_stride,
        max_frames=max_frames,
    )

    return create_session_from_paths(extracted_paths)


@app.get("/api/image/{session_id}/{image_id}")  # Register the endpoint that serves dataset images to the canvas.
def get_image(session_id: str, image_id: str) -> FileResponse:  # Define the image serving endpoint implementation.
    image_record = get_image_record(session_id, image_id)  # Resolve the requested image record.
    return FileResponse(image_record["path"])  # Return the image file directly to the browser.


@app.post("/api/auto/run")  # Register the endpoint that runs semi-automatic annotation on one image.
def run_auto(request: AutoRunRequest) -> dict:  # Define the semi-automatic endpoint implementation.
    image_record = get_image_record(request.session_id, request.image_id)  # Resolve the selected image record.
    result_tuple = backend.run_grounded_pipeline_for_path(image_record["path"], request.prompt, request.class_name, request.output_root, request.device_mode, request.rows, request.cols, request.box_threshold, request.text_threshold, request.tile_nms_threshold, request.global_nms_threshold, request.max_detections, request.min_box_area, request.draw_boxes, request.draw_labels)  # Run the shared Grounded-SAM pipeline.
    run_folder = result_tuple[12]  # Read the run folder path from the shared backend result tuple.
    run_zip = result_tuple[10]  # Read the run zip path from the shared backend result tuple.
    annotations = [normalize_annotation(annotation) for annotation in backend.build_auto_annotations_from_run(run_folder)]  # Convert the run folder into editable browser annotations.
    backend.export_annotations_for_image(image_record["path"], annotations, request.output_root, request.class_name, request.export_format)  # Export the initial auto annotations into the selected dataset format.
    dataset_zip = package_dataset(request.output_root, request.export_format)  # Package the current export dataset into a downloadable zip.
    session_payload = get_session(request.session_id)  # Load the session payload so this result can be cached.
    session_payload["runs"][request.image_id] = {"annotations": annotations, "run_folder": run_folder, "run_zip": run_zip, "dataset_zip": dataset_zip}  # Cache the latest run metadata for this image.
    return {"annotations": annotations, "run_folder": run_folder, "run_zip_url": build_download_url(run_zip), "dataset_zip_url": build_download_url(dataset_zip), "summary": result_tuple[4], "status": result_tuple[5]}  # Return editable annotations and download links.


@app.post("/api/manual/candidate")  # Register the endpoint that runs classic SAM from one clicked point.
def run_manual_candidate(request: ManualCandidateRequest) -> dict:  # Define the manual SAM candidate endpoint implementation.
    session_id = request.session_id or request.sessionId
    image_id = request.image_id or request.imageId
    device_mode = request.device_mode or request.deviceMode or "auto"

    if not session_id or not image_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "Missing session_id/image_id. "
                f"request={request.model_dump()}"
            ),
        )

    click_x = int(round(float(request.x)))
    click_y = int(round(float(request.y)))

    image_record = get_image_record(session_id, image_id)
    image_rgb = cv2_read_rgb(image_record["path"])
    image_height, image_width = image_rgb.shape[:2]

    print("=" * 80)
    print("Manual SAM request")
    print("image path:", image_record["path"])
    print("image size:", image_width, image_height)
    print("clicked x, y:", click_x, click_y)
    print("device:", device_mode)
    print("=" * 80)

    if click_x < 0 or click_y < 0 or click_x >= image_width or click_y >= image_height:
        raise HTTPException(
            status_code=400,
            detail=f"Clicked point is outside the image. click=({click_x}, {click_y}), image=({image_width}, {image_height})",
        )

    _, predictor = backend.get_manual_sam_bundle(
        backend.DEFAULT_SAM_CHECKPOINT,
        backend.DEFAULT_SAM_ENCODER,
        device_mode,
    )

    predictor.set_image(image_rgb)

    input_point = np_array_point(click_x, click_y)
    input_label = np_array_label()

    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    print("SAM scores:", scores)

    best_index = int(scores.argmax())
    best_mask = masks[best_index]

    bbox = backend.mask_to_bbox(best_mask)

    print("best index:", best_index)
    print("bbox:", bbox)

    if bbox is None:
        raise HTTPException(
            status_code=422,
            detail="SAM did not produce a usable mask.",
        )

    annotation = {
        "bbox": [int(value) for value in bbox],
        "score": float(scores[best_index]),
        "point": [click_x, click_y],
        "polygons": backend.mask_to_polygons(best_mask),
    }

    return {
        "candidate": normalize_annotation(annotation),
        "status": f"SAM candidate score={float(scores[best_index]):.4f}, bbox={bbox}",
    }


@app.post("/api/export/image")  # Register the endpoint that saves edited annotations for one image.
def export_image(request: ExportRequest) -> dict:  # Define the edited export endpoint implementation.
    image_record = get_image_record(request.session_id, request.image_id)  # Resolve the selected image record.
    annotations = [normalize_annotation(annotation) for annotation in request.annotations]  # Normalize browser-edited annotations for backend export.
    export_result = backend.export_annotations_for_image(image_record["path"], annotations, request.output_root, request.class_name, request.export_format)  # Export the edited annotations into the selected dataset format.
    dataset_zip = package_dataset(request.output_root, request.export_format)  # Package the updated dataset folder into a zip archive.
    session_payload = get_session(request.session_id)  # Load the active session payload.
    session_payload["runs"].setdefault(request.image_id, {})["annotations"] = annotations  # Cache the edited annotations in memory.
    session_payload["runs"].setdefault(request.image_id, {})["dataset_zip"] = dataset_zip  # Cache the latest dataset zip path in memory.
    return {"annotations": annotations, "export_result": {key: str(value) for key, value in export_result.items()}, "dataset_zip_url": build_download_url(dataset_zip), "status": f"Saved {len(annotations)} annotations."}  # Return save status and download link.


@app.get("/api/download/{token}")  # Register a tokenized download endpoint for zip files and generated artifacts.
def download_file(token: str) -> FileResponse:  # Define the download endpoint implementation.
    if token not in DOWNLOADS:  # Reject unknown or expired download tokens.
        raise HTTPException(status_code=404, detail="Download token not found.")  # Return a clear API error for invalid download tokens.
    file_path = Path(DOWNLOADS[token]).resolve()  # Resolve the file path registered for this token.
    if not file_path.exists() or not file_path.is_file():  # Validate that the requested download target exists.
        raise HTTPException(status_code=404, detail="File not found.")  # Return a clear API error for missing files.
    return FileResponse(str(file_path), filename=file_path.name)  # Return the requested file as a browser download.
