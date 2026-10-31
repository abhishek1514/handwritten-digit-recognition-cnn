# Handwritten Digit Recognition using CNN

A deep learning project that recognizes handwritten digits (0–9) using a Convolutional Neural Network (CNN) built with PyTorch and the MNIST dataset.

The project includes a Tkinter desktop application where users can draw handwritten digits and get real-time predictions from the trained CNN model.

## Project Features

- MNIST handwritten digit dataset.
- Image preprocessing and normalization using PyTorch.
- CNN-based feature extraction and classification.
- Model training and evaluation.
- Trained model weights saved using PyTorch.
- Interactive Tkinter drawing canvas.
- Digit prediction with confidence scores.
- Clear canvas functionality.

## Technologies Used

- Python
- PyTorch
- Torchvision
- CNN (Convolutional Neural Network)
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
├── train_model.py         # Dataset preprocessing and CNN training
│
├── app.py                 # Tkinter GUI and digit prediction
│
├── digit_cnn.pth          # Trained CNN model weights
│
├── requirements.txt       # Python dependencies
│
├── README.md              # Project documentation
│
└── .gitignore             # Git ignored files
```

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
py -m venv .venv
```

### 4. Activate the virtual environment (Windows PowerShell)

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Run the Application

The repository includes the trained model weights (`digit_cnn.pth`), so you can launch the application without training again.

```bash
python app.py
```

Draw a digit from 0 to 9 on the canvas and click **Predict** to see the model's prediction.

Click **Clear** to erase the canvas and draw another digit.

## Train the Model

To train the CNN from scratch, run:

```bash
python train_model.py
```

The training script automatically:

1. Downloads the MNIST dataset.
2. Preprocesses and normalizes the images.
3. Builds the CNN architecture.
4. Trains the model using Adam optimizer and CrossEntropyLoss.
5. Evaluates performance on the MNIST test dataset.
6. Saves the trained weights as `digit_cnn.pth`.

The default training configuration uses 5 epochs and a batch size of 64.

## CNN Architecture

| Layer | Configuration |
|---|---|
| Input | 1 × 28 × 28 grayscale image |
| Conv2D | 1 → 16 channels, 3 × 3 |
| ReLU | Activation |
| MaxPool2D | 2 × 2 |
| Conv2D | 16 → 32 channels, 3 × 3 |
| ReLU | Activation |
| MaxPool2D | 2 × 2 |
| Flatten | 32 × 7 × 7 |
| Linear | 1568 → 128 |
| ReLU | Activation |
| Dropout | 0.2 |
| Output | 10 digit classes (0–9) |

## Model Evaluation

The model reports its test accuracy after training on the MNIST test dataset.

**Note:** Prediction confidence is the model's estimated probability, not a guarantee of correctness.

## Future Improvements

- Improve handwritten digit centering and preprocessing.
- Increase prediction accuracy on user-drawn digits.
- Add prediction history.
- Improve the desktop interface.

## Author

**GitHub:** [abhishek1514](https://github.com/abhishek1514)

## License

This project is available for educational and learning purposes.
