#!/usr/bin/env python

# import libs
try:
    import gradio as gr
except Exception as e:
    print(f"Caught exception: {e}")
    raise e

# define globals
GRADIO_CUSTOM_PATH="/sdui"
GRADIO_MODELS_PATH="models/stable-diffusion"

class StableDiffusionUI(object):
    def __init__(self, inference_endpoint):
        self.sd_ui = None
        self.sd_pipeline = None
        self.inference_endpoint = inference_endpoint

    # build the user interface object
    def buildUi(self, func_callback):
        # build gradio application
        self.sd_ui = gr.Interface(title="Stable Diffusion Txt2Img Demo", fn=func_callback,
                             inputs=[gr.Textbox(value=self.inference_endpoint, label="Inference URL"),
                                    gr.Textbox(label="Prompt"),
                                    gr.Textbox(label="Negative Prompt"),
                                    gr.Slider(label="Denoising Steps", value=5, minimum=1, maximum=100, step=1),
                                    gr.Number(label="Width", value=512), gr.Number(label="Height", value=512),
                                    gr.Slider(label="Guidance Scale", value=7, minimum=1, maximum=100, step=0.5),
                                    gr.Number(label="Seed", value=-1)],
                             outputs=[gr.Image(label="Generated Image", format="png", show_download_button=True), gr.JSON(label="Generation Parameters")])

    # build interface for a locally hosted model
    def localUi(self, model):
        try:
            import torch
            from diffusers import StableDiffusionPipeline
        except Exception as e:
            print(f"Cannot import transformers: {e}")
            raise e

        # check for GPU
        accelerator = "cpu"
        if torch.cuda.is_available():
            print(f"Running on GPU!")
            accelerator = "cuda"
            dtype = torch.float16
            self.sd_pipeline = StableDiffusionPipeline.from_single_file(model, torch_dtype=dtype, use_safetensors=True)
        else:
            print(f"NO GPU FOUND.")
            self.sd_pipeline = StableDiffusionPipeline.from_single_file(model, use_safetensors=True)

        self.sd_pipeline.to(accelerator)
        self.sd_ui = gr.Interface.from_pipeline(self.sd_pipeline)

    # register application in FastAPI
    def registerFastApiEndpoint(self, fastApiApp, path=GRADIO_CUSTOM_PATH):
        fastApiApp = gr.mount_gradio_app(fastApiApp, self.sd_ui, path=path)
