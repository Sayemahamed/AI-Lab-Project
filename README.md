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
├── COVID-19_Radiography_Dataset/    # Dataset directory
│   ├── COVID/                       # COVID-19 positive X-rays
│   └── Normal/                      # Normal chest X-rays
├── dataset.py                       # Dataset loading and preprocessing
├── app.py                          # Streamlit web interface
├── model.pth                       # Trained model weights (after training)
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd [repository-name]
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

### Data Processing (`dataset.py`)
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

## Citation

If you use this project or dataset, please cite the following works:

```bibtex
@article{chowdhury2020can,
  title={Can AI help in screening viral and COVID-19 pneumonia?},
  author={Chowdhury, M.E.H. and Rahman, T. and Khandakar, A. and Mazhar, R. and Kadir, M.A. and Mahbub, Z.B. and Islam, K.R. and Khan, M.S. and Iqbal, A. and Al-Emadi, N. and others},
  journal={IEEE Access},
  volume={8},
  pages={132665--132676},
  year={2020},
  publisher={IEEE}
}

@article{rahman2020exploring,
  title={Exploring the Effect of Image Enhancement Techniques on COVID-19 Detection using Chest X-ray Images},
  author={Rahman, T. and Khandakar, A. and Qiblawey, Y. and Tahir, A. and Kiranyaz, S. and Kashem, S.B.A. and Islam, M.T. and Maadeed, S.A. and Zughaier, S.M. and Khan, M.S. and Chowdhury, M.E.},
  journal={Computers in Biology and Medicine},
  year={2021}
}
```

## License

[Your chosen license]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
