# Sushrut Samitah: Phonocardiogram (PCG) Analysis and Diagnosis

**Sushrut Samitah** is a web application designed to analyze Phonocardiogram (PCG) signals using a Convolutional Neural Network (CNN) model. The application allows users to upload a PCG signal in `.wav` format, predict the heart condition based on the uploaded signal, and provide detailed information on various heart-related conditions.

## Table of Contents

- [Project Description](#project-description)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Description](#model-description)
- [Callbacks](#callbacks)
- [Contributing](#contributing)
- [License](#license)

## Project Description

Sushrut Samitah leverages deep learning to assist in diagnosing various heart conditions from PCG signals. It uses a pre-trained CNN model to classify signals into one of the following categories:

- Normal
- Aortic Stenosis
- Mitral Stenosis
- Mitral Valve Prolapse
- Pericardial Murmurs

The application provides a user-friendly interface to upload PCG files, visualize the heart sound signal, and predict the associated heart condition. For certain diagnoses, users can access detailed information about the condition and potential treatments.

## Features

- **File Upload**: Upload `.wav` files containing PCG signals.
- **Visualization**: View the uploaded heart sound signal as a graph.
- **Prediction**: Predict the heart condition using a CNN model.
- **Detailed Information**: Access in-depth details about specific heart conditions.
- **Doctor Connection**: (Feature under development) Option to connect with a healthcare professional.

## Installation

To run the application locally, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/sushrut_samitah.git
    cd sushrut_samitah
    ```

2. **Install the required packages**:
    Ensure you have the following dependencies installed:
    - Python 3.x
    - Dash
    - Keras / TensorFlow
    - SciPy
    - Scikit-learn
    - Plotly

    Install the required packages using pip:
    ```bash
    pip install dash keras scipy scikit-learn plotly
    ```

3. **Place your trained model**:
    Ensure that your trained CNN model (`pcg_model.h5`) is in the project directory.

4. **Run the application**:
    ```bash
    python app.py
    ```

5. **Access the application**:
    Open your web browser and go to `http://127.0.0.1:8050/` to view the application.

## Usage

### Upload a PCG Signal

- Click the 'Upload WAV File' button to upload a PCG `.wav` file.
- Once uploaded, the signal will be visualized as a line graph.

### Predict Heart Condition

- After uploading, click the 'Predict Heart Condition' button.
- The application will display the predicted condition with an option to learn more about the diagnosis.

### Navigate Information Pages

- Explore detailed pages for specific heart conditions like Aortic Stenosis, Mitral Stenosis, Mitral Valve Prolapse, and Pericardial Murmurs.

## Model Description

The CNN model used in this application is trained to classify PCG signals into five categories. It processes the input signal through several convolutional layers with ReLU activation, followed by max-pooling, and finally, a fully connected layer with softmax activation for classification.

## Callbacks

- **File Upload Callback**: Handles file parsing, preprocessing, visualization, and prepares the signal for prediction.
- **Prediction Callback**: Takes the preprocessed signal and predicts the heart condition using the CNN model.
- **Page Navigation Callback**: Manages the dynamic content displayed on different pages based on the user's navigation.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for discussion.

## License

This project is licensed under the MIT License.
