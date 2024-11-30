import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

# Configure Streamlit page
st.set_page_config(
    page_title="COVID-19 X-Ray Analysis",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Material Design styling
st.markdown("""
<style>
    /* Material Design Colors */
    :root {
        --primary: #2196F3;
        --primary-light: #6EC6FF;
        --primary-dark: #0069C0;
        --secondary: #424242;
        --surface: #FFFFFF;
        --background: #FAFAFA;
        --error: #B00020;
        --success: #4CAF50;
    }

    /* Typography */
    h1, h2, h3 {
        color: var(--primary-dark);
        font-family: 'Roboto', sans-serif;
        font-weight: 500;
    }

    /* Cards */
    .md-card {
        background: var(--surface);
        border-radius: 12px;
        padding: 24px;
        margin: 16px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: box-shadow 0.3s ease;
    }
    .md-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }

    /* Results */
    .result-card {
        text-align: center;
        padding: 24px;
        border-radius: 12px;
        margin-top: 24px;
    }
    .result-card.covid {
        background-color: #ffebee;
        border: 1px solid #ef5350;
    }
    .result-card.normal {
        background-color: #e8f5e9;
        border: 1px solid #66bb6a;
    }
    .confidence {
        font-size: 1.2rem;
        margin: 12px 0;
        color: var(--secondary);
    }
    .prediction {
        font-size: 1.8rem;
        font-weight: 500;
        margin: 16px 0;
    }
    .covid-prediction {
        color: #c62828;
    }
    .normal-prediction {
        color: #2e7d32;
    }

    /* Medical Disclaimer */
    .disclaimer {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 16px;
        margin: 16px 0;
        font-size: 0.9rem;
        color: #424242;
    }

    /* Upload Area */
    .upload-area {
        border: 2px dashed #90caf9;
        border-radius: 12px;
        padding: 32px;
        text-align: center;
        background: #e3f2fd;
        transition: all 0.3s ease;
    }
    .upload-area:hover {
        border-color: var(--primary);
        background: #bbdefb;
    }

    /* Progress Bar */
    .stProgress > div > div > div {
        background-color: var(--primary);
    }
</style>
""", unsafe_allow_html=True)

class NEURAL_NETWORK(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.Dropout2d(0.2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            nn.BatchNorm2d(256),
            nn.Flatten(),
            nn.Linear(256 * 37 * 37, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.model(x)

# Load and prepare model
model = NEURAL_NETWORK()
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu'), weights_only=True))
model.eval()

# Image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def process_image(image):
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    
    # Convert grayscale to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    
    with torch.inference_mode():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence * 100

def main():
    # Sidebar
    with st.sidebar:
        st.image("assets/covid-19.jpg", use_column_width=True)
        st.markdown("## About")
        st.markdown("""
        This AI-powered tool analyzes chest X-ray images to assist in COVID-19 detection.
        
        ### Dataset
        - 3,616 COVID-19 X-rays
        - 10,192 Normal X-rays
        - 299x299 pixel resolution
        
        ### Model
        - Deep CNN architecture
        - Trained on COVID-19 Radiography Dataset
        - Regular updates and improvements
        """)
        
        st.markdown("### 📊 Performance Metrics")
        st.progress(95, "Accuracy: 95%")
        st.progress(93, "Sensitivity: 93%")
        st.progress(96, "Specificity: 96%")

    # Main content
    st.markdown("# 🫁 COVID-19 X-Ray Analysis")
    st.markdown("### AI-Powered COVID-19 Detection from Chest X-Rays")

    # Medical Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <strong>⚕️ Medical Disclaimer:</strong><br>
        This tool is for research and educational purposes only. It should not be used as a substitute for professional medical advice, 
        diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider.
    </div>
    """, unsafe_allow_html=True)

    # Upload Section
    st.markdown('<div class="md-card">', unsafe_allow_html=True)
    st.markdown("### 📤 Upload X-Ray Image")
    uploaded_file = st.file_uploader("Choose a chest X-ray image (PNG, JPG, JPEG)", 
                                   type=["png", "jpg", "jpeg"],
                                   help="Upload a clear, front-view chest X-ray image")

    if uploaded_file:
        try:
            # Display image
            image = Image.open(uploaded_file)
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption="Uploaded X-Ray", use_column_width=True)
            
            with col2:
                with st.spinner("Analyzing X-ray..."):
                    # Process image
                    predicted_class, confidence = process_image(uploaded_file)
                    
                    # Display results
                    result_class = "normal" if predicted_class == 1 else "covid"
                    st.markdown(f"""
                        <div class="result-card {result_class}">
                            <h2>Analysis Results</h2>
                            <p class="prediction {'normal-prediction' if predicted_class == 1 else 'covid-prediction'}">
                                {'Normal X-Ray' if predicted_class == 1 else 'COVID-19 Detected'}
                            </p>
                            <p class="confidence">Confidence: {confidence:.1f}%</p>
                            <p>{'✅ No COVID-19 indicators detected' if predicted_class == 1 else '⚠️ Seek immediate medical attention'}</p>
                        </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional Information
    st.markdown("""
    <div class="md-card">
        <h3>💡 Tips for Best Results</h3>
        <ul>
            <li>Use clear, high-resolution X-ray images</li>
            <li>Ensure proper front-view chest X-ray positioning</li>
            <li>Avoid blurry or low-quality images</li>
            <li>Use PNG or JPEG format</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
