<h1 align="center"><a href="https://johnnyboy1013.github.io/articutable-project/"><img src="https://johnnyboy1013.github.io/articutable-project/assets/favicon-180.png" width="32" height="32" align="middle" alt="ArticuTable project icon"></a> ArticuTable</h1>

<p align="center">Generating Instance-Level Interactive Rigid-Articulated 3D Tabletop Scenes from a Single Image</p>

<p align="center">
  <a href="https://johnnyboy1013.github.io/">Kai Lv</a> ·
  <a href="https://bryceyin13.github.io/">Yibo Yin</a> ·
  <a href="https://shawnricardo.github.io/">Lijun Guo</a> ·
  <a href="https://hengfan2010.github.io/">Heng Fan</a> ·
  <a href="https://zhangkaihao.github.io/">Kaihao Zhang</a> ·
  <a href="https://xingpingdong.github.io/">Xingping Dong</a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2610.05249"><img src="https://johnnyboy1013.github.io/assets/arxiv-icon.svg" width="18" height="18" align="middle" alt="">&nbsp;Paper</a>&emsp;
  <a href="https://johnnyboy1013.github.io/articutable-project/"><img src="https://johnnyboy1013.github.io/assets/articutable-icon.webp?v=2" width="18" height="18" align="middle" alt="">&nbsp;Project Page</a>&emsp;
  <a href="https://johnnyboy1013.github.io/articutable-project/#demo"><img src="assets/demo-icon.svg" width="18" height="18" align="middle" alt="">&nbsp;3D Demo</a>&emsp;
  <a href="https://johnnyboy1013.github.io/articutable-project/#videos"><img src="assets/video-icon.svg" width="18" height="18" align="middle" alt="">&nbsp;Videos</a>&emsp;
  <a href="https://drive.google.com/file/d/1udTcI9ESuk-TbKOYjIQ4ti8t4CMoGW4b/view?usp=sharing"><img src="https://johnnyboy1013.github.io/assets/google-drive-icon.png" width="18" height="18" align="middle" alt="">&nbsp;Dataset</a>
</p>

[![ArticuTable reconstructs interactive articulated tabletop scenes from a single image](https://johnnyboy1013.github.io/articutable-project/assets/teaser.png?v=20261008-v2)](https://johnnyboy1013.github.io/articutable-project/)

## Overview

ArticuTable turns a single generated or real tabletop image into an input-view-consistent, simulation-ready 3D scene with executable part-level motion. **GRAM** recovers articulated object structure and motion ranges; **PSGSR** registers the reconstructed objects into a coherent scene. This repository contains the core pipeline and both components.

## Dataset

[Download ArticuTable-100](https://drive.google.com/file/d/1udTcI9ESuk-TbKOYjIQ4ti8t4CMoGW4b/view?usp=sharing), a collection of 100 simulation-ready tabletop scenes (~20 GB).

## Repository structure

```text
pipeline/   image preprocessing, image-to-3D, segmentation, and orchestration
gram/       primitive segmentation, kinematic inference, and URDF export
psgsr/      pose, scale, and geometry-aware scene registration
prompts/    prompts used by the pipeline and GRAM
```

## Setup

Python 3.10 or newer is required.

```bash
conda create -n "YOUR_ARTICUTABLE_CONDA_ENV" python=3.10
conda activate "YOUR_ARTICUTABLE_CONDA_ENV"
python -m pip install -r requirements.txt
```

Install Blender, FFmpeg, TRELLIS.2, SAM3, P3-SAM, VGGT,
DINO, and their model weights separately. Point GRAM to P3-SAM with:

```bash
export GRAM_P3SAM_CHECKOUT="/path/to/Hunyuan3D-Part/P3-SAM"
export GRAM_P3SAM_CHECKPOINT="/path/to/p3sam.safetensors"
export GRAM_BLENDER="/path/to/blender"
```

## OpenAI API

Multimodal reasoning and image generation use the OpenAI Responses API.

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
export OPENAI_API_BASE="YOUR_API_BASE"

export ARTICUTABLE_MLLM_MODEL="YOUR_ARTICUTABLE_MLLM_MODEL"
export ARTICUTABLE_MLLM_REASONING_EFFORT="YOUR_REASONING_EFFORT"

export GRAM_MLLM_MODEL="YOUR_GRAM_MLLM_MODEL"
export GRAM_MLLM_REASONING_EFFORT="YOUR_REASONING_EFFORT"

export IMAGE_ORCHESTRATOR_MODEL="YOUR_IMAGE_ORCHESTRATOR_MODEL"
export IMAGE_ORCHESTRATOR_REASONING_EFFORT="YOUR_REASONING_EFFORT"
export IMAGE_GENERATION_MODEL="YOUR_IMAGE_GENERATION_MODEL"
```

## Conda environments

Create separate Conda environments for the core pipeline, SAM 3, TRELLIS.2,
GRAM/P3-SAM, and VGGT/PSGSR. Configure their names or Python executables before
running the pipeline:

```bash
conda create -n "YOUR_SAM3_CONDA_ENV" python=3.10
conda create -n "YOUR_TRELLIS2_CONDA_ENV" python=3.10
conda create -n "YOUR_GRAM_CONDA_ENV" python=3.10
conda create -n "YOUR_VGGT_CONDA_ENV" python=3.10

export SAM3_CONDA_ENV="YOUR_SAM3_CONDA_ENV"
export TRELLIS2_PYTHON="/path/to/YOUR_TRELLIS2_CONDA_ENV/bin/python"
export GRAM_CONDA_ENV="YOUR_GRAM_CONDA_ENV"
export VGGT_CONDA_ENV="YOUR_VGGT_CONDA_ENV"
```

Select runtime devices explicitly:

```bash
export SAM3_CUDA_VISIBLE_DEVICES="YOUR_GPU_ID"
export TRELLIS2_GPUS="YOUR_GPU_ID_LIST"
export VGGT_DEVICE="cuda:YOUR_GPU_ID"
export MINIMA_DEVICE="cuda:YOUR_GPU_ID"
export ISAAC_GPU="YOUR_GPU_ID"
export ISAAC_SIM_ROOT="/path/to/isaac-sim"
```

## Run the ArticuTable pipeline

List the available stages:

```bash
python -m pipeline.cli --scene example --list-stages
```

Run the pipeline on a generated or real input image with TRELLIS.2 and GRAM:

```bash
python -m pipeline.cli \
  --scene example \
  --input-image /path/to/image.png \
  --articulation-method gram
```

## GRAM

GRAM performs primitive segmentation, kinematic structure inference,
MLLM-guided joint fitting, semantic state reasoning, physics-based motion-range
validation, and URDF export.

```bash
python -m gram.cli all \
  --input /path/to/object.glb \
  --run-dir /path/to/gram-output \
  --p3sam-python /path/to/p3sam/python \
  --p3sam-checkout /path/to/Hunyuan3D-Part/P3-SAM \
  --p3sam-checkpoint /path/to/p3sam.safetensors \
  --blender /path/to/blender
```

## PSGSR

PSGSR accepts front and top observations, their segmentation results, the scene
blueprint, and the image-to-3D manifest:

```bash
python -m psgsr.registration \
  --front-image /path/to/front.png \
  --top-image /path/to/top.png \
  --front-segmentation /path/to/front_segmentation.json \
  --top-segmentation /path/to/top_segmentation.json \
  --blueprint /path/to/blueprint.json \
  --image-to-3d-manifest /path/to/image_to_3d_manifest.json \
  --output-dir /path/to/output
```

## Citation

If you use ArticuTable in your research, please cite:

```bibtex
@misc{lv2026articutablegeneratinginstancelevelinteractive,
  title={ArticuTable: Generating Instance-Level Interactive Rigid-Articulated 3D Tabletop Scenes from a Single Image},
  author={Kai Lv and Yibo Yin and Lijun Guo and Heng Fan and Kaihao Zhang and Xingping Dong},
  year={2026},
  eprint={2610.05249},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2610.05249},
}
```
