import torch
import gradio as gr
from PIL import Image
from torchvision import transforms
from torch import nn

from Critic import Discriminator

# Loading in the model weights
critic = Discriminator(in_channels=3)
critic.load_state_dict(torch.load("critic.pth", map_location=torch.device('cpu')))
critic.eval()

# BCE = nn.BCEWithLogitsLoss()

# Loading in the same transform so that model performance remains the same
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def check_defective(image):
    # Convert PIL to tensor
    tensor_image = transform(image).unsqueeze(0) 
    
    # Model output
    with torch.no_grad():
        result = critic(tensor_image)
    
    # Min–max normalize
    mat_min = result.min()
    mat_max = result.max()
    mat_normalized = (result - mat_min) / (mat_max - mat_min)
    
    # Mean score
    mean_score = mat_normalized.view(tensor_image.shape[0], -1).mean(1).item()
    
    # Decision
    if mean_score > 0.5:
        return f"Not Defective ✅ (score: {mean_score:.2f})"
    else:
        return f"Defective ❌ (score: {1 - mean_score:.2f})"

# launching Gradio Interface
iface = gr.Interface(
    fn=check_defective,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(),
    title="Defect Detector",
    description="Upload an image to check if it's defective or not."
)

iface.launch(share=True)