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
        div.stButton > button {
            display: block;
            margin: 2rem auto;
            padding: 0.8rem 1.5rem;
            font-size: 1.2rem;
            background-color: #1f77b4;
            color: white;
            border-radius: 10px;
        }
        div.stButton > button:hover {
            background-color: #ffffff;
            color: #000000;
            transition: all 0.3s ease;
        }
        </style>
        """,
        unsafe_allow_html=True
    )  
    
    st.markdown('<h1 class="main-title centered">COVID-19 Chest X-Ray Prediction</h1>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.image("assets/covid-19.jpg", width=500)
    with col2:
        st.markdown(
        """
        <div class="centered description">
        Welcome to the COVID-19 Chest X-Ray Prediction application.<br><br>
        This advanced AI-powered tool helps medical professionals and researchers analyze chest X-ray images 
        to assist in COVID-19 detection. Upload your chest X-ray image to receive instant predictions using 
        our state-of-the-art deep learning model.
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("Get Started"):
        st.session_state.page = "main"

def main_interface():
    st.markdown(
        """
        <style>
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
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown('<h1 class="centered-title">Covid-19 Prediction Agent</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="centered-subtitle">Upload your Chest X-ray Image</h3>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.markdown('<div class="centered">🤖Classifying...</div>', unsafe_allow_html=True)

            # todo: send the image to the model
            result = 1
            st.header("Result")
            st.subheader(f"Prediction: {result}")

        except Exception as e:
            st.error(f"Error processing image: {e}")

# Main application logic
def main():
    st.set_page_config(page_title="COVID-19 Prediction", layout="wide")

    # Initialize session state for page navigation
    if 'page' not in st.session_state:
        st.session_state.page = "landing"

    # Page navigation
    if st.session_state.page == "landing":
        landing_page()
    elif st.session_state.page == "main":
        main_interface()

if __name__ == "__main__":
    main()
