
# Sushrut Samitah: Phonocardiogram (PCG) Analysis and Diagnosis

**Sushrut Samitah** is a web application designed to analyze Phonocardiogram (PCG) signals using a Convolutional Neural Network (CNN) model. The application allows users to upload a PCG signal in `.wav` format, predict the heart condition based on the uploaded signal, and provide detailed information on various heart-related conditions.

## Installation

Ensure you have the following dependencies installed:

- Python 3.x
- NumPy
- SciPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib

You can install the required packages using pip:

```bash
pip install numpy scipy scikit-learn tensorflow matplotlib
```
## Dataset

The dataset consists of .wav files representing different classes of PCG signals. The dataset is structured into folders where each folder corresponds to one class.

dataset/
│
├── normal/
│   ├── file1.wav
│   ├── file2.wav
│   └── ...
├── aortic_stenosis/
│   ├── file1.wav
│   ├── file2.wav
│   └── ...
├── mitral_stenosis/
│   ├── file1.wav
│   ├── file2.wav
│   └── ...
├── mitral_valve_prolapse/
│   ├── file1.wav
│   ├── file2.wav
│   └── ...
└── pericardial_murmurs/
    ├── file1.wav
    ├── file2.wav
    └── ...
