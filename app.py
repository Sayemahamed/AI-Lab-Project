import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="COVID-19 Detection from Chest X-rays",
    page_icon="🫁",
    layout="wide"
)

def load_model(model_path):
    """
    Load the trained PyTorch model.
    """
    try:
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image):
    """
    Preprocess the input image for model prediction.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image).unsqueeze(0)

def predict(model, image):
    """
    Make prediction on the input image.
    """
    try:
        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            return probabilities.numpy()[0]
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return None

def main():
    # Add custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stTitle {
            color: #2c3e50;
            font-size: 3rem !important;
            font-weight: 700 !important;
            margin-bottom: 2rem !important;
        }
        .upload-box {
            border: 2px dashed #4a90e2;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            margin: 20px 0;
        }
        .result-box {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title and description
    st.title("COVID-19 Detection from Chest X-rays")
    st.markdown("""
        This application uses deep learning to detect COVID-19 from chest X-ray images. 
        Upload a chest X-ray image to get started.
    """)

    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info("""
        This model was trained on the COVID-19 Radiography Dataset, which contains:
        - 3,616 COVID-19 positive chest X-rays
        - 10,192 Normal chest X-rays
        
        The model uses a deep learning architecture optimized for medical image classification.
    """)

    # Model loading
    model_path = "model.pth"  # Update this with your model path
    if not os.path.exists(model_path):
        st.error("Model file not found. Please ensure the model is properly trained and saved.")
        return

    model = load_model(model_path)
    if model is None:
        st.error("Error loading the model. Please check the logs for details.")
        return

    # File uploader
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image...", 
        type=["png", "jpg", "jpeg"]
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        try:
            # Display the uploaded image
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Uploaded X-ray")
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, use_column_width=True)

            # Preprocess and predict
            preprocessed_image = preprocess_image(image)
            predictions = predict(model, preprocessed_image)

            if predictions is not None:
                with col2:
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.subheader("Prediction Results")
                    
                    # Display prediction probabilities
                    covid_prob = predictions[1] * 100
                    normal_prob = predictions[0] * 100

                    # Create a progress bar for COVID-19 probability
                    st.write("COVID-19 Probability:")
                    st.progress(float(covid_prob) / 100)
                    st.write(f"{covid_prob:.2f}%")

                    # Create a progress bar for Normal probability
                    st.write("Normal Probability:")
                    st.progress(float(normal_prob) / 100)
                    st.write(f"{normal_prob:.2f}%")

                    # Display the final prediction
                    prediction = "COVID-19 Positive" if covid_prob > normal_prob else "Normal"
                    confidence = max(covid_prob, normal_prob)
                    
                    st.markdown(f"""
                        ### Final Prediction:
                        **{prediction}** (Confidence: {confidence:.2f}%)
                    """)
                    
                    # Add a warning message
                    st.warning("""
                        **Disclaimer:** This is a preliminary screening tool and should not be used as a definitive diagnosis. 
                        Please consult with healthcare professionals for proper medical diagnosis.
                    """)
                    st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            logger.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()
