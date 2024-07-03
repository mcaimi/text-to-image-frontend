#!/usr/bin/env python

# import libs
try:
    import gradio as gr
    import random
    from libs.callbacks import local_prediction, RANDOM_BIT_LENGTH
except Exception as e:
    print(f"Caught exception: {e}")
    raise e

# define globals
GRADIO_CUSTOM_PATH="/sdui"
GRADIO_MODELS_PATH="models/stable-diffusion"

class StableDiffusionUI(object):
    def __init__(self):
        self.sd_ui = None
        self.sd_pipeline = None
        self.accelerator = None

    # generation callback
    def gen_callback(self, prompt, negative_prompt, steps, width, height, cfg, seed):
        # call local callback function
        return local_prediction(self.sd_pipeline,
                                prompt=prompt,
                                negative_prompt=negative_prompt,
                                steps=steps,
                                width=width, height=height,
                                seed=seed,
                                guidance_scale=cfg,
                                accelerator=self.accelerator)

    # read html components
    def html_component(self, path):
        try:
            with open(path) as x:
                return "".join([ i.strip() for i in x.readlines()])
        except:
            return f"<h1> {path} not found </h1>"

    # build interface for a locally hosted model
    def buildUi(self, model):
        try:
            import torch
            import torch.cuda as tc
            import torch.backends.mps as tmps
            from diffusers import StableDiffusionPipeline
        except Exception as e:
            print(f"Cannot import transformers: {e}")
            raise e

        # check for GPU
        self.accelerator = "cpu"
        if tmps.is_available():
            print(f"Apple Metal Shaders Available!")
            self.accelerator = "mps"
            dtype = torch.float16
            self.sd_pipeline = StableDiffusionPipeline.from_single_file(model, torch_dtype=dtype, use_safetensors=True)
        elif tc.is_available():
            device_name = tc.get_device_name()
            device_capabilities = tc.get_device_capability()
            device_available_mem, device_total_mem = [x / 1024**3 for x in tc.mem_get_info()]
            print(f"A GPU is available! [{device_name} - {device_capabilities} - {device_available_mem}/{device_total_mem} GB VRAM]")
            self.accelerator = "cuda"
            dtype = torch.float16
            self.sd_pipeline = StableDiffusionPipeline.from_single_file(model, torch_dtype=dtype, use_safetensors=True)
        else:
            print(f"NO GPU FOUND.")
            self.sd_pipeline = StableDiffusionPipeline.from_single_file(model, use_safetensors=True)

        self.sd_pipeline.to(self.accelerator)
        with gr.Blocks() as sdInterface:
            gr.HTML(value=self.html_component("assets/header.html"))
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(label="Prompt")
                    negative_prompt = gr.Textbox(label="Negative Prompt")
                    with gr.Row():
                        submit_btn = gr.Button("Generate")
                    with gr.Accordion("Additional Parameters"):
                        steps = gr.Slider(label="Denoising Steps", value=5, minimum=1, maximum=100, step=1)
                        cfg = gr.Slider(label="Guidance Scale", value=7, minimum=1, maximum=100, step=0.5)
                        seed = gr.Number(label="Seed", value=-1)
                        with gr.Row():
                            width = gr.Number(label="Width", value=512)
                            height = gr.Number(label="Height", value=512)
                with gr.Column():
                    output_image = gr.Image(label="Generated Image", format="png", show_download_button=True)
                    json_out = gr.JSON(label="Generation Parameters")

            # attach function callbacks
            submit_btn.click(fn=self.gen_callback, inputs=[prompt, negative_prompt, steps, width, height, cfg, seed], outputs=[output_image, json_out], api_name=False)

        self.sd_ui = sdInterface
#        self.sd_ui = gr.Interface(title="Stable Diffusion Txt2Img Demo",
#                                  article="Visit us at <a href=https://redhat.com>RedHat</a>",
#                                  fn=self.gen_callback,
#                                  inputs=[gr.Textbox(label="Prompt"),
#                                    gr.Textbox(label="Negative Prompt"),
#                                    gr.Slider(label="Denoising Steps", value=5, minimum=1, maximum=100, step=1),
#                                    gr.Number(label="Width", value=512), gr.Number(label="Height", value=512),
#                                    gr.Slider(label="Guidance Scale", value=7, minimum=1, maximum=100, step=0.5),
#                                    gr.Number(label="Seed", value=-1)],
#                             outputs=[gr.Image(label="Generated Image", format="png", show_download_button=True), gr.JSON(label="Generation Parameters")])

    # register application in FastAPI
    def registerFastApiEndpoint(self, fastApiApp, path=GRADIO_CUSTOM_PATH):
        fastApiApp = gr.mount_gradio_app(fastApiApp, self.sd_ui, path=path)
