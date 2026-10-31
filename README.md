# Handwritten Digit Recognition CNN

A PyTorch convolutional neural network that recognizes handwritten digits from
0 to 9. The Tkinter app provides a canvas for drawing a digit and displaying
the model's prediction.

## Project structure

```text
Handwritten-Digit-Recognition-CNN/
├── data/                 # MNIST is downloaded here when training
├── train_model.py        # Downloads MNIST and trains the CNN
├── app.py                # Drawing app and digit prediction
├── digit_cnn.pth         # Trained model weights
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

From this directory, create and activate a virtual environment, then install
the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the application

The included `digit_cnn.pth` model can be used without training:

```powershell
python app.py
```

Draw a digit on the canvas and select **Predict**. Select **Clear** to start
over.

## Train the model

Training automatically downloads the MNIST dataset into `data/MNIST/`:

```powershell
python train_model.py
```

The default is five epochs. Training overwrites `digit_cnn.pth`. You can
customize training with `--epochs`, `--batch-size`, and `--learning-rate`, for
example:

```powershell
python train_model.py --epochs 8 --batch-size 64 --learning-rate 0.001
```
