#!/usr/bin/env python

# import libs
try:
    import gradio as gr
except Exception as e:
    print(f"Caught exception: {e}")
    raise e

# define globals
GRADIO_CUSTOM_PATH = "/sdui"
schedulers = ["DPM++ 2M",
              "DPM++ SDE",
              "DPM2",
              "Euler a",
              "Euler",
              "Heun",
              "LMS"]

class StableDiffusionUI(object):
    def __init__(self, inference_endpoint):
        self.sd_ui = None
        self.inference_endpoint = inference_endpoint

    # read html components
    def html_component(self, path):
        try:
            with open(path) as x:
                return "".join([i.strip() for i in x.readlines()])
        except Exception:
            raise gr.Error(f"Html Component {path} not found", duration=5)

    # build the user interface object
    def buildUi(self, func_callback):
        # render interface
        with gr.Blocks(theme=gr.themes.Soft()) as sdInterface:
            gr.HTML(value=self.html_component("assets/header.html"))
            endpoint = gr.Textbox(value=self.inference_endpoint, label="Inference URL")
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(label="Prompt")
                    negative_prompt = gr.Textbox(label="Negative Prompt")
                    with gr.Row():
                        submit_btn = gr.Button(value="Generate", variant="primary")
                        clear_btn = gr.ClearButton(value="Clear")
                    with gr.Accordion("Additional Parameters", open=False):
                        steps = gr.Slider(label="Denoising Steps", value=5, minimum=1, maximum=100, step=1)
                        cfg = gr.Slider(label="Guidance Scale", value=7, minimum=1, maximum=100, step=0.5)
                        seed = gr.Number(label="Seed", value=-1)
                        scheduler_dropdown = gr.Dropdown(scale=2, min_width=300, multiselect=False, label="Scheduler", choices=schedulers, value="DPM++ 2M")
                        with gr.Row():
                            width = gr.Number(label="Width", value=512)
                            height = gr.Number(label="Height", value=512)
                with gr.Column():
                    output_image = gr.Image(label="Generated Image", format="png", show_download_button=True)
                    with gr.Accordion("Image Parameters", open=False):
                        json_out = gr.JSON(label="Generation Parameters")

            # attach function callbacks
            submit_btn.click(fn=func_callback, inputs=[endpoint, prompt, negative_prompt, steps, width, height, cfg, seed, scheduler_dropdown], outputs=[output_image, json_out], api_name=False)
            clear_btn.add(components=[prompt, negative_prompt, steps, width, height, cfg, seed])

        self.sd_ui = sdInterface

    # register application in FastAPI
    def registerFastApiEndpoint(self, fastApiApp, path=GRADIO_CUSTOM_PATH):
        fastApiApp = gr.mount_gradio_app(fastApiApp, self.sd_ui, path=path)
