from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from PIL import Image
import requests
import torch


model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_id,
                                                               torch_dtype=torch.float16,
                                                               low_cpu_mem_usage=True,).to(0)

processor = AutoProcessor.from_pretrained(model_id)

conversation = [
    {

      "role": "user",
      "content": [
          {"type": "text", "text": "prompt"},
          {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

image_file = "URL"
raw_image = Image.open(requests.get(image_file, stream=True).raw)
inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)

output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))
