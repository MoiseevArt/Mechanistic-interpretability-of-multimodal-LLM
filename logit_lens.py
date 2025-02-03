from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
import requests
import torch


model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16).to(0)
processor = AutoProcessor.from_pretrained(model_id)

conversation = [
    {

      "role": "user",
      "content": [
          {"type": "text", "text": "what do you see? In one word "},
          {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

image_file = "http://farm7.staticflickr.com/6208/6082497230_dd2497547e_z.jpg"
raw_image = Image.open(requests.get(image_file, stream=True).raw)
inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)

output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))

# Dictionary for storing logits
logits_per_layer = {}


# Hook function for saving logits
def hook_fn(module, input, output, layer_idx):
    with torch.no_grad():
        logits_per_layer[layer_idx] = model.language_model.lm_head(output[0])


# Add hooks to all layers of Qwen2DecoderLayer
hooks = []
for layer_idx, layer in enumerate(model.language_model.model.layers):
    hook = layer.register_forward_hook(lambda module, input, output, idx=layer_idx: hook_fn(module, input, output, idx))
    hooks.append(hook)

# Run the text through the model
with torch.no_grad():
    outputs = model(**inputs)

# Removing hooks
for hook in hooks:
    hook.remove()


# Select layers for analysis
selected_layers = [0, 5, 10, 15, 20, 23]

# Building separate graphs for each layer
for layer_idx in selected_layers:
    plt.figure(figsize=(10, 4))

    # We obtain token probabilities via softmax
    probs = F.softmax(logits_per_layer[layer_idx][0, -1], dim=-1).cpu().numpy()

    plt.plot(probs)
    plt.xlabel("token ID")
    plt.ylabel("Probability")
    plt.title(f"Probability distribution of tokens on a layer {layer_idx}")

    plt.show()
