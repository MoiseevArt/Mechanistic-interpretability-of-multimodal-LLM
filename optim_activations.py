from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import requests
import torch
import torch
import torch.optim as optim
import kornia.augmentation as K
from kornia.filters import gaussian_blur2d

model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_id,
                                                               torch_dtype=torch.float16,
                                                               low_cpu_mem_usage=True).to("cuda")

processor = AutoProcessor.from_pretrained(model_id)

image_url = "https://farm9.staticflickr.com/8196/8125509122_e8348e8d89_z.jpg"
raw_image = Image.open(requests.get(image_url, stream=True).raw)


transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
])

image_resized = transform(raw_image).unsqueeze(0).to(0, dtype=torch.float16)

activations = {}

# Configuration
steps = 1000
lr = 0.05
image_size = 384
jitter_amount = 0.2  # Random Shift
blur_sigma = 0.5  # Blur power
regScale = 1e-5  # regularization

model.eval()
for param in model.parameters():
    param.requires_grad = False


target_layer = model.vision_tower.vision_model.encoder.layers[5]  # Selecting the target layer
target_neuron_idx = 1151  # Neuron index for visualization

activations = None


def hook_fn(module, inp, outp):
    global activations
    activations = outp


hook = target_layer.self_attn.register_forward_hook(hook_fn)

# initialization of optimized noisy image
base_image = torch.randn(1, 3, image_size, image_size, device="cuda") * 0.1
base_image.requires_grad_(True)

optimizer = optim.Adam([base_image], lr=lr)

# Augmentations (Kornia for differentiability)
jitter = K.RandomAffine(degrees=0, translate=(jitter_amount, jitter_amount), p=1.0)

# Normalization
mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)


for step in range(steps):
    optimizer.zero_grad()

    img = torch.sigmoid(base_image)  # Image generation via sigmoid

    # Augmentations & Normalization
    img = jitter(img)
    img = gaussian_blur2d(img, kernel_size=(3, 3), sigma=(blur_sigma, blur_sigma))
    img_normalized = (img - mean) / std

    # Forward pass
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        _ = model.vision_tower(img_normalized)

    # loss
    neuron_act = activations[0][:, :, target_neuron_idx].mean()

    loss = -neuron_act + regScale * torch.norm(base_image, p=2)
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")


hook.remove()


def denormalize(img):
    return img * std + mean


# Image transformation
final_img = denormalize(base_image).detach().cpu()
final_img = (final_img - final_img.min()) / (final_img.max() - final_img.min())  # Normalize to [0, 1]

# Visualization
plt.figure(figsize=(8, 8))
plt.imshow(final_img.squeeze().permute(1, 2, 0))
plt.axis("off")
plt.show()
