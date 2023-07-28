import logging
import os
from time import time
import torch
from diffusers import DiffusionPipeline
import diffusers
from djl_python.inputs import Input
from djl_python.outputs import Output
from typing import Optional
from io import BytesIO
from PIL import Image


def get_torch_dtype_from_str(dtype: str):
    if dtype == "fp16":
        return torch.float16
    raise ValueError(
        f"Invalid data type: {dtype}. DeepSpeed currently only supports fp16 for stable diffusion"
    )

class StableDiffusionService(object):

    def __init__(self):
        self.pipeline = None
        self.initialized = False
        self.logger = logging.getLogger()
        self.model_id_or_path = None
        self.data_type = None
        self.device = None
        self.world_size = None
        self.max_tokens = None
        self.save_image_dir = None

    def initialize(self, properties: dict):
        # model_id can point to huggingface model_id or local directory.
        # If option.model_id points to a s3 bucket, we download it and set model_id to the download directory.
        # Otherwise we assume model artifacts are in the model_dir
        self.model_id_or_path = properties.get("model_id") or properties.get(
            "model_dir")
        self.data_type = get_torch_dtype_from_str(properties.get("dtype"))
        self.max_tokens = int(properties.get("max_tokens", "1024"))
        self.device = int(os.getenv("LOCAL_RANK", "0"))

        if os.path.exists(self.model_id_or_path):
            config_file = os.path.join(self.model_id_or_path,
                                       "model_index.json")
            if not os.path.exists(config_file):
                raise ValueError(
                    f"{self.model_id_or_path} does not contain a model_index.json."
                    f"This is required for loading stable diffusion models from local storage"
                )

        kwargs = {"torch_dtype": torch.float16, "revision": "fp16"}
        
        start = time()
        pipeline = DiffusionPipeline.from_pretrained(self.model_id_or_path,
                                                     device_map='auto',
                                                     low_cpu_mem_usage=True,
                                                     **kwargs
                                                    )
    
        duration = time()-start 
        self.logger.info(f'Loaded model in {duration} seconds')

        self.pipeline = pipeline
        self.initialized = True


    def inference(self, inputs: Input):
        try:
            content_type = inputs.get_property("Content-Type")
            if content_type == "application/json":
                request = inputs.get_as_json()
                prompt = request.pop("prompt")
                params = request.pop("parameters", {})
                start = time()
                result = self.pipeline(prompt, **params)
                duration = time() - start
                self.logger.info(f'Inference took {duration} seconds')
            elif content_type and content_type.startswith("text/"):
                prompt = inputs.get_as_string()
                result = self.pipeline(prompt)
            else:
                # in case an image and a prompt is sent in the input
                init_image = Image.open(BytesIO(
                    inputs.get_as_bytes())).convert("RGB")
                request = inputs.get_as_json("json")
                prompt = request.pop("prompt")
                params = request.pop("parameters", {})
                result = self.pipeline(prompt, image=init_image, **params)

            img = result.images[0]
            buf = BytesIO()
            img.save(buf, format="PNG")
            byte_img = buf.getvalue()
            outputs = Output().add(byte_img).add_property(
                "content-type", "image/png")

        except Exception as e:
            logging.exception("Inference failed")
            outputs = Output().error(str(e))
        return outputs


_service = StableDiffusionService()


def handle(inputs: Input) -> Optional[Output]:
    if not _service.initialized:
        _service.initialize(inputs.get_properties())

    if inputs.is_empty():
        return None

    return _service.inference(inputs)