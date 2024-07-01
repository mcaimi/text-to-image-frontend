#!/usr/bin/env python

# load libs
import base64
try:
    import requests
    from PIL import Image
    from torch import Generator
    import random
    import io
except Exception as e:
    print(f"Caught exception: {e}")
    raise e

RANDOM_BIT_LENGTH = 64

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
#        "seed": 772847624537827,
#      }
#    ]
#  }

# prepare the payload
def prepare_payload(prompt,
                 negative_prompt="",
                 steps=10,
                 width=512, height=512,
                 cfg=7,
                 seed=-1):
    # prepare seed
    if seed == -1:
        custom_seed = random.getrandbits(RANDOM_BIT_LENGTH)
        print(f"Generating with random seed: {custom_seed}")
    else:
        custom_seed = seed
        print(f"Generating with constant seed: {custom_seed}")

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
                "seed": custom_seed,
            }
        ]
    }

    # return built payload
    return json_data


# define the call function
def rest_request(url, json_data,
                 tls_verify=False, timeout=600):
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
                 seed=-1,
                 timeout=600,
                 tls_verify=False):
    # build request payload
    kserve_request = prepare_payload(prompt, negative_prompt=negative_prompt,
                                   steps=steps, width=width, height=height,
                                   cfg=cfg, seed=seed)

    # call the generation function
    kserve_response = rest_request(url, json_data=kserve_request, tls_verify=tls_verify, timeout=timeout)

    # extract the payload
    image_payload = kserve_response.get("predictions")[0].get("image").get("b64")
    # decode from base64
    img_data = base64.b64decode(image_payload)

    # return image bytes
    return Image.open(io.BytesIO(img_data)), kserve_request

# callback for sd generation on a local machine
def local_prediction(model_pipeline,
                 prompt,
                 negative_prompt="",
                 steps=10,
                 width=512, height=512,
                 guidance_scale=7,
                 seed=-1,
                 accelerator="cpu"):
    # prepare generator object
    if seed==-1:
        gen = Generator(accelerator).manual_seed(random.getrandbits(RANDOM_BIT_LENGTH))
    else:
        gen = Generator(accelerator).manual_seed(seed)


    # generate image from prompt
    prediction = model_pipeline(prompt=prompt,
                                negative_prompt=negative_prompt,
                                num_inference_steps=steps,
                                width=width,
                                height=height,
                                guidance_scale=guidance_scale,
                                generator=gen)

    # generation metagada payload
    metadata = prepare_payload(prompt=prompt,
                               negative_prompt=negative_prompt,
                               steps=steps, width=width, height=height,
                               cfg=guidance_scale, seed=seed)

    return prediction.images[0], metadata
