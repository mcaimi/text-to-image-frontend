# Text To Image Demo Frontend

This is the frontend application for the [Text To Image Stable Diffusion Demo](https://github.com/mcaimi/text-to-image-demo)
It exposes a way to perform inference with a Stable Diffusion compatible model in the backend.

It is built with FastAPI and Gradio and works both locally (in a virtual environment for example) or inside Openshift.
It has however a dependency on the KServe model server engine that runs in OCP AI. Look [here](https://github.com/mcaimi/kserve-diffusers-demo) for the backend code.

![gradio_app](assets/gradio_app.png)

## Run Locally

Normally the software will run in "remote" mode, meaning that it will use the "INFER_URL" environment variable to locate the API endpoint to use to perform inference.
Usually that endpoint is hosted remotely on Openshift AI.

If the "RUN_LOCALLY" env variable is set to True, then inference is done locally by loading the model specified in the "MODEL_NAME" env variable. It expects to find this file under a specific path:

```bash
$ export RUN_LOCALLY="True"
$ export MODEL_NAME="DreamShaper_8_pruned.safetensors" # only supports single safetensors models

$ fastapi dev frontend.py
```

It tries to use any discovered GPU that is supported by pytorch, but it can run on CPU (*very* slowly)

Tested:
- Apple Metal Performance Shaders
- NVIDIA CUDA
- AMD CPU

