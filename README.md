# NVIDIA Nemotron Parse Setup Guide

This guide provides step-by-step instructions for setting up and running NVIDIA Nemotron Parse v1.1 with vLLM.

## Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Driver Version >= 12.8
- Docker (for container-based setup) or a virtual machine image

## 1. Environment Setup

### Option 1: Using a Base Container Image

Start with the NVIDIA CUDA development container:

```bash
docker run -it --rm --gpus all nvcr.io/nvidia/cuda-dl-base:25.05-cuda12.9-devel-ubuntu24.04 /bin/bash
```

Verify CUDA toolkit installation:

```bash
nvcc --version
```

Expected output:

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Wed_Jan_15_19:20:09_PST_2025
Cuda compilation tools, release 12.8, V12.8.61
Build cuda_12.8.r12.8/compiler.35404655_0
```

> **Note:** The host CUDA version (from `nvidia-smi`) must be >= 12.8

Install Python development packages:

```bash
apt-get update && apt-get install -y python3.12-dev
```

### Option 2: Directly Using a Virtual Machine Image

If running directly on a VMI without a container:

```bash
nvidia-smi
```

Ensure the output shows `CUDA Version: 12.8` or `12.9`

## 2. Verify CUDA_HOME (Optional but Recommended)

> **Warning:** Without properly setting `CUDA_HOME`, vLLM wheel compilation might fail

Check if `CUDA_HOME` is set:

```bash
echo "CUDA_HOME: $CUDA_HOME" && echo "PATH: $PATH" | grep cuda
```

The output should show file paths pointing to `/usr/local/cuda/bin` or similar. Base container images typically have this pre-configured.

## 3. Python Environment Setup

### Install UV Package Manager

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
```

Re-open your terminal or source the environment, then create a Python virtual environment:

```bash
# Create UV environment with Python 3.12
uv venv --python 3.12 --seed
source .venv/bin/activate
```

## 4. Install vLLM

### Option 1: Use Prebuilt Wheel (Recommended)

> **Warning:** You must specify a specific prebuilt wheel URL. This installation takes a couple of minutes.

```bash
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_LOCATION=https://github.com/vllm-project/vllm/releases/download/v0.11.0/vllm-0.11.0-cp38-abi3-manylinux1_x86_64.whl \
uv pip install "git+https://github.com/amalad/vllm.git@nemotron_parse"
```

### Option 2: Build from Source

> **Note:** This method works when installing vLLM directly on NVIDIA GPU Optimized AMI but may not work with the `nvcr.io/nvidia/cuda-dl-base:25.05-cuda12.9-devel-ubuntu24.04` container.

This process takes approximately 40 minutes to 2 hours:

```bash
uv pip install "git+https://github.com/amalad/vllm.git@nemotron_parse" 2>&1 | tee output.txt
```

## 5. Install Additional Dependencies

```bash
uv pip install timm albumentations open-clip-torch
```

## 6. Launch vLLM Server

> **Tip:** The official repository uses the vLLM Python SDK. This setup includes a `template.jinja` file to enable vLLM server usage.

### Create Chat Template

Create a `template.jinja` file in your working directory:

```jinja
{% for message in messages %}{{ message['content'] }}{% endfor %}
```

### Start the Server

```bash
vllm serve nvidia/NVIDIA-Nemotron-Parse-v1.1 \
    --dtype bfloat16 \
    --max-num-seqs 4 \
    --limit-mm-per-prompt '{"image": 1}' \
    --trust-remote-code \
    --port 8000 \
    --chat-template template.jinja
```

**Configuration Notes:**
- Adjust `--max-num-seqs` based on your GPU capacity (4 is suitable for an L4 GPU)
- The server will be available at `http://localhost:8000`

## 7. Next Steps

Follow the provided notebooks to interact with the Nemotron Parse model.

## Troubleshooting

- **CUDA_HOME not set:** Ensure your CUDA toolkit is properly installed and environment variables are configured
- **Engine core failed:** Verify GPU compatibility and CUDA driver version
- **Wheel compilation errors:** Try using the prebuilt wheel option instead of building from source

## License

Please refer to the original NVIDIA Nemotron Parse repository for licensing information.
