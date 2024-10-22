import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np



# Load the trained model
class BaselineCNN(torch.nn.Module):
    def __init__(self):
        super(BaselineCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        self.batchnorm1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = torch.nn.MaxPool2d(2, 2)
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(64 * 32 * 32, 128)
        self.out = torch.nn.Linear(128, 4)  # 4 output classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.batchnorm1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x

@st.cache_resource()
def load_model():
    model = BaselineCNN()
    model.load_state_dict(torch.load("../models/alzheimers_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Define disease categories
disease_label_from_category = {
    0: "Mild Demented",
    1: "Moderate Demented",
    2: "Non-Demented",
    3: "Very Mild Demented",
}

# Preprocess the uploaded image
def preprocess_image(image):
    img = Image.open(image).convert("L")  # Grayscale
    img = img.resize((128, 128))  # Resize
    img = np.array(img)
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return img

# Hero section
st.markdown("""
<div class="hero">
    <h1>Alzheimer's Disease MRI Classifier</h1>
    <p>Detect the stage of Alzheimer's from MRI scans .</p>
</div>
""", unsafe_allow_html=True)

# Upload and process MRI image
uploaded_file = st.file_uploader("Upload an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image with a specified width (e.g., 300 pixels)
    st.image(uploaded_file, caption='Uploaded MRI Image', use_column_width=False, width=300)

    img_tensor = preprocess_image(uploaded_file)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output.data, 1)

    predicted_label = disease_label_from_category[predicted.item()]
    st.markdown(f"<h2>Predicted Class: {predicted_label}</h2>", unsafe_allow_html=True)
