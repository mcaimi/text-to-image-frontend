#!/usr/bin/env python

# load libraries
import os,sys
import base64
try:
    from fastapi import FastAPI
    import requests
    from PIL import Image
    import gradio as gr
    import io
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
GRADIO_CUSTOM_PATH = "/gradio"

# this is the inference method exposed by the KServe Model Server
infer_endpoint = f"{INFER_URL}/v1/models/model:predict"

# An example JSON payload:
#
# // example payload:
#  {
#    "instances": [
#      {
#        "prompt": "photo of the beach",
#        "negative_prompt": "ugly, deformed, bad anatomy",
#        "num_inference_steps": 20,
#        "width": 512,
#        "height": 512,
#        "guidance_scale": 7,
#        // "seed": 772847624537827,
#      }
#    ]
#  }
# define the call function
def rest_request(url, prompt,
                 negative_prompt="",
                 steps=10,
                 width=512, height=512,
                 cfg=7,
                 timeout=600,
                 tls_verify=False):
    # prepare payload
    json_data = {
        "instances": [
            {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": steps,
                "width": width,
                "height": height,
                "guidance_scale": cfg,
            }
        ]
    }

    # call the inference service
    response = requests.post(url, json=json_data, verify=tls_verify, timeout=timeout)

    # extract the resoponse payload
    response_dict = response.json()
    return response_dict


# build the callback function
def generate_image(url, prompt,
                 negative_prompt="",
                 steps=10,
                 width=512, height=512,
                 cfg=7,
                 timeout=600,
                 tls_verify=False):
    # call the generation function
    kserve_response = rest_request(url, prompt, negative_prompt=negative_prompt,
                                   steps=steps, width=width, height=height,
                                   cfg=cfg, timeout=timeout, tls_verify=tls_verify)

    # extract the payload
    image_payload = kserve_response.get("predictions")[0].get("image").get("b64")
    # decode from base64
    img_data = base64.b64decode(image_payload)

    # return image bytes
    return Image.open(io.BytesIO(img_data))


# build gradio application
sd_ui = gr.Interface(fn=generate_image,
                     inputs=[gr.Textbox(value=infer_endpoint, label="Inference URL"),
                            gr.Textbox(label="Prompt"),
                            gr.Textbox(label="Negative Prompt"),
                            gr.Slider(label="Denoising Steps", value=5, minimum=1, maximum=100, step=1),
                            gr.Number(label="Width", value=512), gr.Number(label="Height", value=512),
                            gr.Slider(label="Guidance Scale", value=7, minimum=1, maximum=100, step=0.5)],
                     outputs=gr.Image())


# build the application object
sd_app = FastAPI()

# add a root path
@sd_app.get("/")
async def get_root():
    # Redirect to the main Gradio App
    return RedirectResponse(url=GRADIO_CUSTOM_PATH, status_code=status.HTTP_302_FOUND)

# attach gradio app
sd_app = gr.mount_gradio_app(sd_app, sd_ui, path=GRADIO_CUSTOM_PATH)

