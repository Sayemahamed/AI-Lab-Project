import streamlit as st
from PIL import Image

def landing_page():
    st.markdown(
        """
        <style>
        .centered {
            text-align: center;
            padding: 0 2rem;
        }
        .main-title {
            color: #1f77b4;
            font-size: 3rem;
            margin-bottom: 2rem;
        }
        .description {
            font-size: 1.2rem;
            line-height: 1.7;
            margin: 2rem 0;
        }
        .content-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 1rem;
        }
        div.stButton > button {
            display: block;
            margin: 2rem auto;
            padding: 0.8rem 1.5rem;
            font-size: 1.2rem;
            background-color: #1f77b4;
            color: white;
            border-radius: 10px;
            transition: all 0.3s ease;
        }
        div.stButton > button:hover {
            background-color: #135c8d;
        }

        @media (max-width: 768px) {
            .main-title {
                font-size: 2rem;
            }
            .description {
                font-size: 1rem;
                padding: 0 1rem;
            }
            div.stButton > button {
                padding: 0.6rem 1.2rem;
                font-size: 1rem;
            }
            .centered {
                padding: 0 1rem;
            }
        }

        @media (max-width: 480px) {
            .main-title {
                font-size: 1.5rem;
            }
            .description {
                font-size: 0.9rem;
                padding: 0 0.5rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="main-title centered">COVID-19 Prediction Agent</h1>', unsafe_allow_html=True)
    
    # Responsive image layout
    if st.session_state.get('screen_width', 0) < 768:
        st.image("assets/covid-19.jpg", use_container_width=True)
        st.markdown(
            """
            <div class="centered description">
            Welcome to the COVID-19 Prediction Agent.<br><br>
            This advanced AI-powered tool helps medical professionals and researchers analyze chest X-ray images 
            to assist in COVID-19 detection.
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.image("assets/covid-19.jpg", width=500)
        with col2:
            st.markdown(
                """
                <div class="centered description">
                Welcome to the COVID-19 Prediction Agent.<br><br>
                This advanced AI-powered tool helps medical professionals and researchers analyze chest X-ray images 
                to assist in COVID-19 detection.
                </div>
                """,
                unsafe_allow_html=True
            )

    if st.button("Get Started"):
        st.session_state.page = "main"
    
    st.markdown('</div>', unsafe_allow_html=True)

def main_interface():
    st.markdown(
        """
        <style>
        /* Base styles */
        .centered-title {
            text-align: center;
            color: #1f77b4;
            font-size: 2.5rem;
            margin-bottom: 1rem;
        }
        .centered-subtitle {
            text-align: center;
            color: #2c3e50;
            font-size: 1.5rem;
            margin-bottom: 2rem;
        }
        .upload-section {
            text-align: center;
            font-size: 1.2rem;
            color: #1f77b4;
            margin-bottom: 1rem;
            font-weight: bold;
        }
        .stCamera {
            margin: 0 auto;
            max-width: 300px !important;
        }
        .stCamera > label {
            font-size: 0.9rem !important;
            margin-bottom: 0.5rem !important;
        }
        .stCamera > button {
            width: 100% !important;
            padding: 0.5rem !important;
            border-radius: 5px !important;
            background-color: #f0f2f6 !important;
            border: 1px solid #e0e3e9 !important;
        }
        .stFileUploader {
            padding: 1rem !important;
        }
        .result-container {
            text-align: center;
            margin-top: 1rem;
            padding: 1rem;
            background-color: #e3f2fd;
            border-radius: 5px;
        }

        /* Responsive styles */
        @media (max-width: 768px) {
            .centered-title {
                font-size: 2rem;
            }
            .centered-subtitle {
                font-size: 1.2rem;
            }
            .upload-section {
                font-size: 1rem;
            }
            [data-testid="stHorizontalBlock"] {
                flex-direction: column;
            }
            [data-testid="stHorizontalBlock"] > div {
                width: 100% !important;
                margin-bottom: 1rem;
            }
            .stCamera {
                max-width: 100% !important;
            }
        }

        @media (max-width: 480px) {
            .centered-title {
                font-size: 1.5rem;
            }
            .centered-subtitle {
                font-size: 1rem;
            }
            .upload-section {
                font-size: 0.9rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown('<h1 class="centered-title">Covid-19 Prediction Agent</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="centered-subtitle">Upload your Chest X-ray Image</h3>', unsafe_allow_html=True)
    
    # Responsive layout for upload options
    if st.session_state.get('screen_width', 0) < 768:
        st.markdown('<div class="upload-section">📁 Upload Image</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        st.markdown('<div class="upload-section">📸 Take Photo</div>', unsafe_allow_html=True)
        camera_file = st.camera_input("Take a picture", key="camera")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="upload-section">📁 Upload Image</div>', unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        with col2:
            st.markdown('<div class="upload-section">📸 Take Photo</div>', unsafe_allow_html=True)
            camera_file = st.camera_input("Take a picture", key="camera")
    
    # Handle image processing
    input_image = None
    if uploaded_file is not None:
        input_image = uploaded_file
    elif camera_file is not None:
        input_image = camera_file

    if input_image is not None:
        try:
            image = Image.open(input_image)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            st.markdown('<div class="centered">🤖 Classifying...</div>', unsafe_allow_html=True)

            # todo: send the image to the model
            result = 1
            st.markdown(
                f"""
                <div class="result-container">
                    <h3>Analysis Results</h3>
                    <p>Prediction: {result}</p>
                </div>
                """, 
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"Error processing image: {e}")

# Main application logic
def main():
    st.set_page_config(page_title="COVID-19 Prediction", layout="wide")

    # Screen width detection
    st.markdown(
        """
        <script>
            var width = window.innerWidth;
            window.parent.postMessage({
                type: 'streamlit:setScreenWidth',
                width: width
            }, '*');
        </script>
        """,
        unsafe_allow_html=True
    )

    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = "landing"

    # Page navigation
    if st.session_state.page == "landing":
        landing_page()
    elif st.session_state.page == "main":
        main_interface()

if __name__ == "__main__":
    main()
