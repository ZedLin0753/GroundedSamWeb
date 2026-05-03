# Semi-Automatic Annotation Tool WebUI

FastAPI + Canvas WebUI for semi-automatic annotation with `GroundingDINO + SAM`.

This folder is the Web edition only. The Qt desktop edition is kept separately in `D:\work\GroundedSamDesktop`.

## Features

- Upload selected images or an image folder.
- Run prompt-based semi-automatic annotation with GroundingDINO + SAM.
- Use CPU or NVIDIA GPU depending on deployment.
- Edit bounding boxes directly on the browser canvas.
- Use manual SAM point prompts when semi-automatic detection is not good enough.
- Export detection labels in YOLO / COCO-oriented formats.
- Download current-image results or the whole dataset result.

## Required Files

```text
GroundedSamWeb/
|-- web_fastapi/
|-- annotation_backend.py
|-- tile_grounded_sam_runner.py
|-- GroundingDINO/
|-- segment_anything/
|-- weights/
|-- Dockerfile
|-- Dockerfile.cpu
|-- Dockerfile.gpu
|-- docker-compose.yml
|-- requirements.txt
|-- start_webui.py
|-- LICENSE
`-- THIRD_PARTY_NOTICES.md
```

`weights/` must contain:

```text
weights/groundingdino_swint_ogc.pth
weights/sam_vit_b_01ec64.pth
```

Download links:

- `groundingdino_swint_ogc.pth`: [GroundingDINO GitHub release](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth)
- `sam_vit_b_01ec64.pth`: [Segment Anything checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

The checkpoint files are intentionally not committed to GitHub because they are large model binaries. Put them in `weights/` before running inference.

## Run Without Docker

```powershell
conda activate ground_sam
cd D:\work\GroundedSamWeb
python .\start_webui.py
```

Open:

```text
http://127.0.0.1:8011/hard-reset-webui-20260503-14
```

## Docker CPU

Use this when the machine has no NVIDIA GPU or Docker GPU support.

```powershell
cd D:\work\GroundedSamWeb
docker compose build webui-cpu
docker compose up webui-cpu
```

Open:

```text
http://127.0.0.1:8011
```

Equivalent direct Docker commands:

```powershell
docker build -f Dockerfile.cpu -t grounded-sam-web:cpu .
docker run --rm -p 8011:8000 -v ${PWD}\weights:/app/weights -v ${PWD}\outputs:/app/outputs grounded-sam-web:cpu
```

## Docker GPU

Use this when the machine has an NVIDIA GPU, NVIDIA driver, Docker Desktop, and NVIDIA Container Toolkit.

```powershell
cd D:\work\GroundedSamWeb
docker compose build webui-gpu
docker compose up webui-gpu
```

Open:

```text
http://127.0.0.1:8011
```

Equivalent direct Docker commands:

```powershell
docker build -f Dockerfile.gpu -t grounded-sam-web:gpu .
docker run --rm --gpus all -p 8011:8000 -v ${PWD}\weights:/app/weights -v ${PWD}\outputs:/app/outputs grounded-sam-web:gpu
```

For GTX 1650 Ti Laptop, the default GPU build already includes compute capability `7.5`.

If you want to build only for GTX 1650 Ti and reduce build time:

```powershell
docker build -f Dockerfile.gpu --build-arg TORCH_CUDA_ARCH_LIST=7.5 -t grounded-sam-web:gpu .
```

## Runtime Device Selection

Inside the WebUI, use the `Device` option:

- `auto`: use GPU when available, otherwise CPU.
- `cuda`: try GPU, then safely fall back to CPU if CUDA is unavailable.
- `cpu`: force CPU.

## Open Source Notices

This tool depends on:

- Grounded-Segment-Anything
- GroundingDINO
- Segment Anything

Keep `LICENSE` and `THIRD_PARTY_NOTICES.md` when publishing to GitHub.
