# THIRD_PARTY_NOTICES

This project includes or depends on third-party open-source software.

The list below is a practical notice summary for portfolio and repository use.

It is not a substitute for reviewing the original upstream licenses before commercial distribution.

## Core Components

### 1. Grounded-Segment-Anything

- Project: `Grounded-Segment-Anything`
- Role in this project: prompt-based candidate generation and segmentation workflow reference
- Upstream repository path in this workspace: [D:\work\GroundedSam](D:/work/GroundedSam)
- Observed license in local repository: `Apache License 2.0`

### 2. GroundingDINO

- Project: `GroundingDINO`
- Role in this project: open-vocabulary text-prompt object proposal generation
- Upstream repository path in this workspace: [D:\work\GroundedSam\GroundingDINO](D:/work/GroundedSam/GroundingDINO)
- Observed license in local repository: `Apache License 2.0`

### 3. Segment Anything

- Project: `Segment Anything`
- Role in this project: mask refinement, manual point-based segmentation, and automatic mask generation
- Upstream repository path in this workspace: [D:\work\GroundedSam\segment_anything](D:/work/GroundedSam/segment_anything)
- Observed license in local repository: `Apache License 2.0`

## Runtime / Library Dependencies

This project also uses common Python libraries and frameworks such as:

- `PyTorch`
- `torchvision`
- `Gradio`
- `NumPy`
- `OpenCV`
- `Pillow`
- `supervision`

Their licenses should also be reviewed if this tool is redistributed publicly, packaged into Docker images, or delivered to third parties.

## Recommendation Before Public Release

Before publishing this project publicly or distributing a packaged version, it is a good idea to:

1. keep the original upstream license files where required
2. review the licenses of all Python dependencies in the final environment
3. document model checkpoints and their upstream source
4. include attribution and notice files in the repository or container image

## AI-Assisted Development Note

This repository also documents AI-assisted development as part of the engineering workflow.

That statement is about authorship transparency and does not replace any upstream license obligations.
