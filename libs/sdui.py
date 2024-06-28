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
            import torch.cuda as tc
            import torch.backends.mps as tmps
            from diffusers import StableDiffusionPipeline
        except Exception as e:
            print(f"Cannot import transformers: {e}")
            raise e

        # check for GPU
        accelerator = "cpu"
        if tmps.is_available():
            print(f"Apple Metal Shaders Available!")
            accelerator = "mps"
            dtype = torch.float16
            self.sd_pipeline = StableDiffusionPipeline.from_single_file(model, torch_dtype=dtype, use_safetensors=True)
        elif tc.is_available():
            device_name = tc.get_device_name()
            device_capabilities = tc.get_device_capability()
            device_available_mem, device_total_mem = [x / 1024**3 for x in tc.mem_get_info()]
            print(f"A GPU is available! [{device_name} - {device_capabilities} - {device_available_mem}/{device_total_mem} GB VRAM]")
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
