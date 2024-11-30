# COVID-19 Detection from Chest X-rays

This project implements a deep learning model to detect COVID-19 from chest X-ray images using the COVID-19 Radiography Dataset. The project includes both a PyTorch-based deep learning model and a user-friendly Streamlit web interface for making predictions.

## Dataset

The COVID-19 Radiography Dataset is a comprehensive collection of chest X-ray images created by researchers from Qatar University, University of Dhaka, and their collaborators. The dataset contains:

### COVID-19 Images (3,616 total)
- 2,473 images from padchest dataset
- 183 images from German medical school
- 559 images from SIRM, Github, Kaggle & Twitter
- 400 images from additional Github sources

### Normal Images (10,192 total)
- 8,851 images from RSNA
- 1,341 images from Kaggle

### Additional Classes (not used in this project)
- 6,012 Lung Opacity images (Non-COVID lung infection) from RSNA
- 1,345 Viral Pneumonia images from various sources

All images are in PNG format with 299x299 pixel resolution. The dataset is regularly updated with new X-ray images as they become available.

## Features

- Deep learning model for COVID-19 detection from X-rays
- Interactive web interface using Streamlit
- Real-time predictions with confidence scores
- Data augmentation and preprocessing pipeline
- Comprehensive error handling and logging
- Medical disclaimer for responsible use

## Project Structure

```
.
├── app.py                          # Streamlit web application for model inference
├── data_preparing.ipynb           # Jupyter notebook for data preprocessing
├── trainer.ipynb                  # Jupyter notebook for model training
├── model.pth                      # Trained PyTorch model weights
├── requirements.txt               # Python package dependencies
├── LICENSE                        # MIT License file
├── README.md                      # Project documentation
├── .gitignore                     # Git ignore rules
├── assets/                        # Static assets and resources
├── prepared_data/                 # Directory for preprocessed dataset
├── prepared_data.zip             # Compressed preprocessed dataset
└── .venv/                         # Python virtual environment
```

### Key Components

#### Application Files
- `app.py`: Main Streamlit web application that provides the user interface for COVID-19 detection
- `data_preparing.ipynb`: Notebook containing data preprocessing pipeline, augmentation, and dataset preparation
- `trainer.ipynb`: Notebook with model architecture, training loop, and evaluation metrics

#### Model and Data
- `model.pth`: Pre-trained deep learning model weights (181MB)
  - Alternative download link: [Download model weights](https://drive.google.com/file/d/16yHvqQNumci1lMAuauXAVfAfhR3qdZAj/view?usp=sharing)
- `prepared_data/`: Directory containing the preprocessed and augmented dataset
- `prepared_data.zip`: Compressed version of the preprocessed dataset (513MB)

#### Configuration and Documentation
- `requirements.txt`: List of Python package dependencies
- `README.md`: Comprehensive project documentation
- `LICENSE`: MIT License file
- `.gitignore`: Specifies which files Git should ignore

#### Resources
- `assets/`: Directory containing static resources like images and styles
- `.venv/`: Python virtual environment directory containing project dependencies

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Sayemahamed/AI-Lab-Project.git
cd AI-Lab-Project
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

1. Prepare your dataset in the correct structure:
```python
from dataset import get_data_loaders

# Get train and validation data loaders
train_loader, val_loader = get_data_loaders(
    data_dir='COVID-19_Radiography_Dataset',
    batch_size=32,
    train_split=0.8
)
```

2. Train your model and save it as `model.pth`

### Using the Web Interface

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided URL (typically http://localhost:8501)

3. Upload a chest X-ray image and get real-time predictions

## Features

### Data Processing (`data_preparing.ipynb`)
- Custom PyTorch Dataset implementation
- Advanced data augmentation techniques
- Stratified train/validation split
- Class weight calculation for imbalanced data
- Comprehensive error handling and logging

### Web Interface (`app.py`)
- Modern, responsive UI
- Real-time predictions
- Probability visualization
- Medical disclaimer
- Error handling and logging
- Support for various image formats

## Dependencies

See `requirements.txt` for a complete list of dependencies. Key packages include:
- PyTorch and torchvision for deep learning
- Streamlit for web interface
- Pillow for image processing
- NumPy for numerical computations
- scikit-learn for data splitting

## Disclaimer

This tool is intended for research and educational purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult with healthcare professionals for medical advice and diagnosis.

## Citations

### Dataset Citations

#### COVID-19 Radiography Database
M.E.H. Chowdhury, T. Rahman, A. Khandakar, R. Mazhar, M.A. Kadir, Z.B. Mahbub, K.R. Islam, M.S. Khan, A. Iqbal, N. Al-Emadi, M.B.I. Reaz, M. T. Islam, "Can AI help in screening viral and COVID-19 pneumonia?" *IEEE Access*, vol. 8, pp. 132665-132676, 2020.  
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FACCESS.2020.3010287-blue)](https://doi.org/10.1109/ACCESS.2020.3010287)

#### Image Enhancement Study
T. Rahman, A. Khandakar, Y. Qiblawey, A. Tahir, S. Kiranyaz, S.B.A. Kashem, M.T. Islam, S.A. Maadeed, S.M. Zughaier, M.S. Khan, M.E. Chowdhury, "Exploring the Effect of Image Enhancement Techniques on COVID-19 Detection using Chest X-ray Images." *Computers in Biology and Medicine*, vol. 132, pp. 104319, 2021.  
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.compbiomed.2021.104319-blue)](https://doi.org/10.1016/j.compbiomed.2021.104319)

### Dataset Components

The dataset includes images from multiple sources that should be acknowledged:

1. **RSNA Pneumonia Detection Challenge Dataset**  
   Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases. *IEEE CVPR* 2017.  
   [![DOI](https://img.shields.io/badge/DOI-10.1109%2FCVPR.2017.369-blue)](https://doi.org/10.1109/CVPR.2017.369)

2. **PadChest Dataset**  
   Bustos A, Pertusa A, Salinas JM, de la Iglesia-Vayá M. PadChest: A large chest x-ray image dataset with multi-label annotated reports. *Medical Image Analysis*, 2020.  
   [![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.media.2020.101797-blue)](https://doi.org/10.1016/j.media.2020.101797)

### Access & Resources

- [COVID-19 Radiography Database on Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
- [IEEE Paper](https://doi.org/10.1109/ACCESS.2020.3010287)
- [Computers in Biology and Medicine Paper](https://doi.org/10.1016/j.compbiomed.2021.104319)

### How to Cite This Project

If you use this project in your research, please cite it as:

```bibtex
@software{COVID19_XRay_Detection,
  author = {Sayem Ahamed},
  title = {COVID-19 Detection from Chest X-rays},
  year = {2023},
  publisher = {GitHub},
  url = {https://github.com/Sayemahamed/AI-Lab-Project}
}
```

## Repository Information
- GitHub Repository: https://github.com/Sayemahamed/AI-Lab-Project.git
- Contact: sayemahamed183@gmail.com

## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
