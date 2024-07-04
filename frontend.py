#!/usr/bin/env python

# load libraries
import os,sys
try:
    from fastapi import FastAPI
    from fastapi.responses import RedirectResponse
    import starlette.status as status
except Exception as e:
    print(f"Caught exception: {e}")
    sys.exit(-1)

# ENVIRONMENT VARIABLES
# the inference endpoint exposed by the model server
# this actually is the entry point that openshift exposes to hide the containerized workload
# KServe is serverless actually
INFER_URL = os.environ.get("INFER_URL", "localhost")
MODEL_NAME = os.environ.get("MODEL_NAME", "DreamShaper_8_pruned.safetensors")
RUN_LOCALLY = os.environ.get("RUN_LOCALLY", "no")

# this is the inference method exposed by the KServe Model Server
infer_endpoint = f"{INFER_URL}/v1/models/model:predict"

# run locally?
if RUN_LOCALLY == "yes":
    from libs.sd_ui_local import StableDiffusionUI, GRADIO_CUSTOM_PATH
    # build gradio ui object
    sd_ui = StableDiffusionUI()
    sd_ui.buildUi()
else:
    from libs.sd_ui_remote import StableDiffusionUI, GRADIO_CUSTOM_PATH
    from libs.callbacks import generate_image
    # build gradio ui object
    sd_ui = StableDiffusionUI(infer_endpoint)
    sd_ui.buildUi(func_callback=generate_image)

# build the application object
sd_app = FastAPI()

# add a root path
@sd_app.get("/")
async def get_root():
    # Redirect to the main Gradio App
    return RedirectResponse(url=GRADIO_CUSTOM_PATH, status_code=status.HTTP_302_FOUND)

# attach gradio app
sd_ui.registerFastApiEndpoint(sd_app)

