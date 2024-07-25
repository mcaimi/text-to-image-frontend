# Text To Image Demo Frontend

This is the frontend application for the [Text To Image Stable Diffusion Demo](https://github.com/mcaimi/text-to-image-demo)
It exposes a way to perform inference with a Stable Diffusion compatible model in the backend.

It is built with FastAPI and Gradio and works both locally (in a virtual environment for example) or inside Openshift.
It has however a dependency on the KServe model server engine that runs in OCP AI. Look [here](https://github.com/mcaimi/kserve-diffusers-demo) for the backend code.

![gradio_app](assets/gradio_app.png)

## Run Locally

Normally the software will run in "remote" mode, meaning that it will use the "INFER_URL" environment variable to locate the API endpoint to use to perform inference.
Usually that endpoint is hosted remotely on Openshift AI.

If the "RUN_LOCALLY" env variable is set to "yes", then inference is run locally using an accelerator if available and supported by pytorch.
The application expects to find model checkpoint files in the `models/stable-diffusion` folder.

```bash
$ export RUN_LOCALLY="yes"

# development mode
$ fastapi dev

# production mode
$ fastapi run
```

Run on Openshift (assuming namespace stable-diffusion and model server instance stable-diffusion):

```bash

$ oc new-app --name stable-diffusion-frontend -e INFER_URL=$(oc get route -n istio-system stable-diffusion-stable-diffusion -o jsonpath='{.spec.host}') quay.io/marcocaimi/stable-diffusion-frontend:latest

```

It tries to use any discovered GPU that is supported by pytorch, but it can run on CPU (*very* slowly)

Tested:
- Apple Metal Performance Shaders
- NVIDIA CUDA
- AMD CPU

