import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights

# Set device to GPU if available, else fallback to CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load and define the ResNet18 model architecture
try:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # Load ResNet18 architecture
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(512, 512),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(512, 256),
        torch.nn.Linear(256, 4)  # 4 output classes (adjust if necessary)
    )
    model.load_state_dict(
        torch.load("models/chest-ctscan_model.pth", map_location=torch.device(device)))  # Load model weights
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()  # Stop the app if the model fails to load

# Class names for the CT scan labels
class_names = [
    'Adenocarcinoma Left lower lobe',
    'Large Cell Carcinoma Left Hilum',
    'Normal',
    'Squamous Cell Carcinoma Left Hilum'
]

# Define the data transformation pipeline
inference_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
])

# Create the Streamlit app
st.title("CT Scan Classification")
st.write("Upload a CT scan image for classification.")

# File uploader
uploaded_file = st.file_uploader("Choose a CT scan image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Open the image
        image = Image.open(uploaded_file)

        # Convert RGBA (4-channel) images to RGB (3-channel)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        elif image.mode != 'RGB':
            st.error("The image is not in a supported format (should be RGB or RGBA).")
            st.stop()

        # Preprocess the image (apply transformations and convert to tensor)
        img_tensor = inference_transform(image).unsqueeze(0).to(device)

        # Make predictions
        with torch.no_grad():
            outputs = model(img_tensor)
            predicted_class = torch.argmax(outputs, dim=1).item()

        # Display the image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Display the results below the image
        st.write(f"**Prediction:** {class_names[predicted_class]}")

    except Exception as e:
        st.error(f"Error processing the image: {e}")
