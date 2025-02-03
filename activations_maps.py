from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import requests
import torch


model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_id,
                                                               torch_dtype=torch.float16,
                                                               low_cpu_mem_usage=True).to("cuda")

processor = AutoProcessor.from_pretrained(model_id)

image_url = "URL"
raw_image = Image.open(requests.get(image_url, stream=True).raw)


transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
])

image_resized = transform(raw_image).unsqueeze(0).to(0, dtype=torch.float16)

activations = {}


def hook_fn(module, input, output):
    activations["vit_features"] = output


# We hang a hook on the desired layer ViT
vit_layer = model.vision_tower.vision_model.encoder.layers[20]  # index layer of ViT
hook = vit_layer.register_forward_hook(hook_fn)

# Run the image through the model
with torch.no_grad():
    _ = model.vision_tower(image_resized)

vit_activations = activations["vit_features"][0]

# Transform vit-activations into the form (C, H, W), where C is the number of channels
activations_maps = vit_activations.squeeze(0).cpu().numpy()

print(f"Shape of feature_maps: {activations_maps.shape}")


n_pixels = activations_maps.shape[0]
n_channels = activations_maps.shape[1]

print(f"Shape of feature_maps: {activations_maps.shape}")
print(f"Number of pixels: {n_pixels}")
print(f"Number of channels per pixel: {n_channels}")

first_10_maps = activations_maps[:, :10]
last_10_maps = activations_maps[:, -10:]


plt.figure(figsize=(10, 3))

# Displaying the first 10 activation cards
for i in range(10):
    feature_map_to_display = first_10_maps[:, i].reshape(int(n_pixels ** 0.5), int(n_pixels ** 0.5))
    plt.subplot(1, 10, i + 1)  # 1 строка, 10 столбцов
    plt.imshow(feature_map_to_display, cmap='viridis', interpolation='nearest')
    plt.axis('off')
plt.tight_layout()
plt.show()

# Create a second chart to display the last 10 activation maps
plt.figure(figsize=(10, 3))

# Displaying the last 10 activation cards
for i in range(10):
    feature_map_to_display = last_10_maps[:, i].reshape(int(n_pixels ** 0.5), int(n_pixels ** 0.5))
    plt.subplot(1, 10, i + 1)
    plt.imshow(feature_map_to_display, cmap='viridis', interpolation='nearest')
    plt.axis('off')

plt.tight_layout()
plt.show()

# delete hook
hook.remove()
