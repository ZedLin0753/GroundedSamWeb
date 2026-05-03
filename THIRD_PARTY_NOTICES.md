# Third-Party Notices

This project includes application code written for this WebUI and selected third-party open-source components required for the GroundingDINO + SAM annotation workflow.

This notice is intended for attribution and portfolio transparency. It is not legal advice. Review the upstream licenses before commercial distribution.

## Project Scope

The original application work in this repository includes the FastAPI WebUI, canvas annotation workflow, Docker packaging, export utilities, and integration glue around the model pipeline.

The core model implementations are third-party projects listed below.

## Included Source Code

### GroundingDINO

- Upstream repository: <https://github.com/IDEA-Research/GroundingDINO>
- Local path: `GroundingDINO/`
- License: Apache License 2.0
- Local license file: `GroundingDINO/LICENSE`
- Role in this project: open-vocabulary text-prompt object proposal generation.

### Segment Anything

- Upstream repository: <https://github.com/facebookresearch/segment-anything>
- Local path: `segment_anything/`
- License: Apache License 2.0
- Local license file: `segment_anything/LICENSE`
- Role in this project: SAM model loading, point-prompt mask prediction, and mask utility code.

## Referenced Workflow

### Grounded-Segment-Anything

- Upstream repository: <https://github.com/IDEA-Research/Grounded-Segment-Anything>
- License: Apache License 2.0
- Role in this project: reference workflow for combining GroundingDINO-style prompt detection with SAM-style segmentation.
- Note: this repository does not claim ownership of the Grounded-Segment-Anything project or its upstream model components.

## Model Checkpoints

The model checkpoint files are intentionally not committed to this repository.

Required checkpoint names:

```text
weights/groundingdino_swint_ogc.pth
weights/sam_vit_b_01ec64.pth
```

Download sources:

- GroundingDINO Swin-T OGC checkpoint: <https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth>
- SAM ViT-B checkpoint: <https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth>

Checkpoint usage may be subject to upstream model terms, dataset terms, or research-use expectations. Review the upstream project documentation before redistribution or commercial use.

## Python And Runtime Dependencies

This project also depends on common Python packages and runtime components, including but not limited to:

- FastAPI
- Uvicorn
- PyTorch
- torchvision
- Transformers
- timm
- NumPy
- OpenCV
- Pillow
- pycocotools
- supervision
- Hugging Face Hub

Package versions are listed in `requirements.txt`. Their licenses are governed by their own upstream projects.

## Non-Affiliation

This project is an independent integration and annotation tool.

It is not affiliated with, endorsed by, or officially maintained by:

- Meta AI
- Facebook Research
- IDEA Research
- the GroundingDINO maintainers
- the Segment Anything maintainers
- the Grounded-Segment-Anything maintainers

## AI-Assisted Development Note

This repository was developed with AI-assisted programming support as part of the engineering workflow.

That statement is included for authorship transparency. It does not replace attribution, license compliance, or model checkpoint usage obligations.
