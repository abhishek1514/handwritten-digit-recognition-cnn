# Handwritten Digit Recognition using CNN

A deep learning project that recognizes handwritten digits (0–9) using a Convolutional Neural Network (CNN) built with PyTorch and the MNIST dataset.

The project includes an interactive Tkinter desktop application where users can draw handwritten digits and get predictions with confidence scores.

## Features

- MNIST handwritten digit dataset (60,000 training and 10,000 testing images).
- Image preprocessing and normalization using PyTorch.
- CNN-based feature extraction and classification.
- Model training using Adam optimizer and CrossEntropyLoss.
- Model evaluation on the MNIST test dataset.
- Saved model weights for reuse without retraining.
- Interactive Tkinter drawing canvas.
- Automatic digit cropping, resizing and centering.
- Top 3 prediction results with confidence scores.
- Clear canvas functionality.

## Technologies Used

- Python
- PyTorch
- Torchvision
- Convolutional Neural Network (CNN)
- MNIST Dataset
- Tkinter
- Pillow (PIL)
- Matplotlib

## Project Structure

```text
Handwritten-Digit-Recognition-CNN/
│
├── data/                  # Downloaded MNIST dataset
│
├── model.py               # Shared CNN architecture
│
├── train_model.py         # Dataset preprocessing and CNN training
│
├── app.py                 # Tkinter GUI and digit prediction
│
├── digit_cnn.pth          # Trained model weights
│
├── requirements.txt       # Project dependencies
│
├── README.md              # Project documentation
│
└── .gitignore             # Git ignored files
```

## CNN Architecture

The CNN consists of two convolutional blocks followed by fully connected classification layers.

| Layer | Configuration |
|---|---|
| Input | 1 × 28 × 28 |
| Conv2D | 1 → 16 channels, 3 × 3 |
| ReLU | Activation |
| MaxPool2D | 2 × 2 |
| Conv2D | 16 → 32 channels, 3 × 3 |
| ReLU | Activation |
| MaxPool2D | 2 × 2 |
| Flatten | 32 × 7 × 7 = 1568 |
| Linear | 1568 → 128 |
| ReLU | Activation |
| Dropout | 0.2 |
| Output | 10 classes (0–9) |

The CNN architecture is defined once in `model.py` and imported by both `train_model.py` and `app.py`.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/abhishek1514/handwritten-digit-recognition-cnn.git
```

### 2. Navigate to the project directory

```bash
cd handwritten-digit-recognition-cnn
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment (Windows PowerShell)

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

## Run the Application

The repository includes `digit_cnn.pth`, so you can launch the application without training the model again.

```bash
python app.py
```

Draw a digit between 0 and 9 and click **Predict** to see the result.

Click **Clear** to erase the drawing and start again.

## Train the Model

To train the CNN from scratch:

```bash
python train_model.py
```

The training script automatically:

1. Downloads the MNIST dataset.
2. Preprocesses and normalizes the images.
3. Visualizes sample handwritten digits.
4. Initializes the CNN architecture from `model.py`.
5. Trains the network for 5 epochs.
6. Evaluates the model on the test dataset.
7. Saves the trained weights to `digit_cnn.pth`.

The training configuration uses:
- Epochs: 5
- Batch size: 64
- Optimizer: Adam
- Learning rate: 0.001
- Loss function: CrossEntropyLoss

## Model Evaluation

Test accuracy is calculated using the 10,000 MNIST test images after training.

The actual accuracy is displayed in the terminal when `train_model.py` finishes.

**Note:** Prediction confidence represents the model's estimated probability and does not guarantee that the prediction is correct.

## Application Workflow

1. Draw a handwritten digit using the mouse.
2. Crop and preprocess the drawing.
3. Resize and center it into a 28 × 28 image.
4. Normalize the image using MNIST statistics.
5. Pass the image to the trained CNN.
6. Display the predicted digit and top 3 predictions.

## Future Improvements

- Improve recognition of user-drawn digits.
- Add prediction history.
- Improve GUI design and usability.
- Experiment with CNN architectures and training parameters.

## Author

GitHub: [abhishek1514](https://github.com/abhishek1514)

## License

This project is intended for educational and learning purposes.
