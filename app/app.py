import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gradio as gr
import cv2
from PIL import Image
from torchvision import models, transforms

# ── Device ──────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on: {device}")

# ── Model Definition ────────────────────────────────────────
class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes=38):
        super().__init__()
        self.backbone = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1
        )
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in list(self.backbone.parameters())[-20:]:
            param.requires_grad = True

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


# ── Load Model ──────────────────────────────────────────────
print("Loading model...")
checkpoint = torch.load('best_model.pth', map_location=device)
CLASS_NAMES = checkpoint['class_names']

model = PlantDiseaseModel(num_classes=38)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()
print(f"Model loaded. Classes: {len(CLASS_NAMES)}")

# ── Preprocessing ───────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ── Grad-CAM ────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, class_idx


# Initialize Grad-CAM on last conv block
gradcam = GradCAM(model, model.backbone.features[-1])

# ── Helper: apply heatmap overlay ───────────────────────────
def apply_heatmap(original_np, cam, alpha=0.4):
    cam_resized = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam_resized),
        cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = alpha * heatmap + (1 - alpha) * original_np
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay

# ── Format class name for display ───────────────────────────
def format_class_name(name):
    parts = name.split('___')
    if len(parts) == 2:
        plant = parts[0].replace('_', ' ')
        disease = parts[1].replace('_', ' ')
        return f"{plant} — {disease}"
    return name.replace('_', ' ')

# ── Main prediction function ─────────────────────────────────
def predict(image):
    if image is None:
        return {}, None

    # Convert PIL to numpy for display
    img_pil = Image.fromarray(image).convert('RGB')
    img_resized = img_pil.resize((224, 224))
    original_np = np.array(img_resized)

    # Preprocess for model
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    # Generate Grad-CAM (requires grad)
    img_tensor.requires_grad_(False)
    cam, pred_idx = gradcam.generate(img_tensor)

    # Get top 5 predictions
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)[0]

    top5_probs, top5_indices = probs.topk(5)

    # Format predictions for gr.Label
    predictions = {
        format_class_name(CLASS_NAMES[idx.item()]): round(prob.item(), 4)
        for prob, idx in zip(top5_probs, top5_indices)
    }

    # Generate overlay image
    overlay = apply_heatmap(original_np, cam)
    overlay_pil = Image.fromarray(overlay)

    return predictions, overlay_pil


# ── Gradio Interface ─────────────────────────────────────────
title = "🌿 Plant Disease Detection"

description = """
Upload a leaf image to detect plant diseases using EfficientNet-B3 trained on the PlantVillage dataset.

**Supports 14 crop species across 38 classes** including Apple, Corn, Grape, Potato, Tomato, and more.

The **Grad-CAM visualization** highlights which regions of the leaf influenced the prediction — red/orange = high importance, blue = low importance.

**Model Performance:** 98.97% test accuracy on PlantVillage dataset (5,431 unseen images)
"""

article = """
### How it works
1. **EfficientNet-B3** pretrained on ImageNet — fine-tuned on 54,305 plant leaf images
2. **Transfer Learning** — ImageNet features (edges, textures) transfer well to leaf disease patterns
3. **Grad-CAM** — visualizes which leaf regions triggered the prediction

### Supported Crops
Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato

### Known Limitations
- Trained on controlled studio images — real-world field photos may reduce accuracy
- Hardest classes: Tomato Early Blight (88.6%), Corn Cercospora (91.1%)
- Most common confusion: Corn Northern Leaf Blight ↔ Cercospora (visually similar lesions)
"""

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(
        type="numpy",
        label="Upload Leaf Image"
    ),
    outputs=[
        gr.Label(
            num_top_classes=5,
            label="Top 5 Predictions"
        ),
        gr.Image(
            type="pil",
            label="Grad-CAM Visualization"
        )
    ],
    title=title,
    description=description,
    article=article,
    theme=gr.themes.Soft(),
    flagging_mode="never"
)

if __name__ == "__main__":
    demo.launch()