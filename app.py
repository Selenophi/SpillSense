import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load Model
def load_model(model_path):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    num_classes = 2  # Background + Oil Spill
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor.cls_score = torch.nn.Linear(in_features, num_classes)
    model.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(in_features, num_classes * 4)
    
    in_features_mask = model.roi_heads.mask_predictor.mask_fcn_logits.in_channels
    model.roi_heads.mask_predictor.mask_fcn_logits = torch.nn.Conv2d(in_features_mask, num_classes, 1)
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

# Preprocess Image
def preprocess_image(image):
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image, image_tensor

# Detect Oil Spill
def detect_oil_spill(model, image_tensor, image, confidence_threshold=0.7):
    with torch.no_grad():
        output = model(image_tensor)
    
    scores = output[0]["scores"].numpy()
    masks = output[0]["masks"].detach().numpy()
    
    oil_spill_detected = any(scores > confidence_threshold)
    
    plt.imshow(image)
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            mask = masks[i][0]
            mask = (mask > 0.5).astype(np.uint8)
            plt.imshow(mask, alpha=0.6, cmap="Reds")
    plt.axis("off")
    return oil_spill_detected, plt

# Streamlit UI
st.title("Oil Spill Detection using Mask R-CNN")
st.write("Upload an image to detect oil spills.")

# File Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Load model
    model_path = "C:/Users/HP/OneDrive/Documents/ByteBash/extracted_model/OilSpillDetection_SAR.pth"  # Update the path
    model = load_model(model_path)
    
    # Preprocess and detect
    image, image_tensor = preprocess_image(image)
    oil_spill_detected, plot = detect_oil_spill(model, image_tensor, image)
    
    st.pyplot(plot)
    if oil_spill_detected:
        st.error("Oil Spill Detected!")
    else:
        st.success("No Oil Spill Detected.")
