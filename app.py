
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

import torch

from PIL import Image, ImageDraw
from torchvision import transforms

from model import DigitCNN


# ==========================================
# STEP 1: DEVICE AND MODEL PATH
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "digit_cnn.pth"


# ==========================================
# STEP 2: LOAD TRAINED MODEL
# ==========================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        "digit_cnn.pth not found. "
        "Run train_model.py first."
    )


model = DigitCNN().to(device)

state_dict = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(state_dict)

model.eval()

print("Model loaded successfully")
print("Using device:", device)


# ==========================================
# STEP 3: IMAGE TRANSFORMATION
# ==========================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


# ==========================================
# STEP 4: IMAGE PREPROCESSING
# ==========================================

def preprocess_digit(image):

    bbox = image.getbbox()

    if bbox is None:
        return None

    # Crop the digit
    cropped = image.crop(bbox)

    # Resize while maintaining aspect ratio
    cropped.thumbnail(
        (20, 20),
        Image.Resampling.LANCZOS
    )

    # Create 28x28 black image
    final_image = Image.new(
        "L",
        (28, 28),
        0
    )

    # Center the digit
    x = (28 - cropped.width) // 2
    y = (28 - cropped.height) // 2

    final_image.paste(
        cropped,
        (x, y)
    )

    # Apply MNIST normalization
    return transform(final_image)


# ==========================================
# STEP 5: CREATE TKINTER WINDOW
# ==========================================

root = tk.Tk()

root.title("Handwritten Digit Recognition")

root.geometry("430x610")

root.resizable(False, False)

root.configure(bg="#172033")


tk.Label(
    root,
    text="HANDWRITTEN DIGIT RECOGNITION",
    font=("Arial", 16, "bold"),
    bg="#172033",
    fg="white"
).pack(pady=(20, 8))


tk.Label(
    root,
    text="Draw a digit between 0 and 9",
    font=("Arial", 11),
    bg="#172033",
    fg="#B9C5D8"
).pack(pady=(0, 15))


# ==========================================
# STEP 6: DRAWING CANVAS
# ==========================================

CANVAS_SIZE = 280

BRUSH_WIDTH = 18

canvas = tk.Canvas(
    root,
    width=CANVAS_SIZE,
    height=CANVAS_SIZE,
    bg="black",
    cursor="cross",
    highlightthickness=2,
    highlightbackground="#52627A"
)

canvas.pack()


# PIL image buffer
image_buffer = Image.new(
    "L",
    (CANVAS_SIZE, CANVAS_SIZE),
    0
)

draw = ImageDraw.Draw(image_buffer)

last_x = None
last_y = None


# ==========================================
# STEP 7: DRAWING FUNCTIONS
# ==========================================

def start_draw(event):

    global last_x, last_y

    last_x = event.x
    last_y = event.y


def draw_digit(event):

    global last_x, last_y

    if last_x is not None and last_y is not None:

        # Draw on Tkinter canvas
        canvas.create_line(
            last_x,
            last_y,
            event.x,
            event.y,
            fill="white",
            width=BRUSH_WIDTH,
            capstyle=tk.ROUND,
            joinstyle=tk.ROUND,
            smooth=True
        )

        # Draw on PIL image buffer
        draw.line(
            (
                last_x,
                last_y,
                event.x,
                event.y
            ),
            fill=255,
            width=BRUSH_WIDTH
        )

        # Rounded brush ends
        radius = BRUSH_WIDTH // 2

        draw.ellipse(
            (
                event.x - radius,
                event.y - radius,
                event.x + radius,
                event.y + radius
            ),
            fill=255
        )

    last_x = event.x
    last_y = event.y


def stop_draw(event):

    global last_x, last_y

    last_x = None
    last_y = None


canvas.bind("<Button-1>", start_draw)

canvas.bind("<B1-Motion>", draw_digit)

canvas.bind("<ButtonRelease-1>", stop_draw)


# ==========================================
# STEP 8: PREDICT DIGIT
# ==========================================

def predict_digit():

    if image_buffer.getbbox() is None:

        messagebox.showwarning(
            "Empty Canvas",
            "Please draw a digit first."
        )

        return

    # Preprocess image
    input_tensor = preprocess_digit(image_buffer)

    if input_tensor is None:
        return

    # Add batch dimension
    input_tensor = input_tensor.unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():

        logits = model(input_tensor)

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        top_probabilities, top_digits = torch.topk(
            probabilities,
            k=3,
            dim=1
        )

    predicted_digit = top_digits[0, 0].item()

    confidence = (
        top_probabilities[0, 0].item() * 100
    )

    # Display main prediction
    result_label.config(
        text=f"Predicted Digit: {predicted_digit}",
        fg="#70F0A0"
    )

    # Display top three predictions
    top_results = []

    for digit, probability in zip(
        top_digits[0].tolist(),
        top_probabilities[0].tolist()
    ):

        top_results.append(
            f"{digit}: {probability * 100:.2f}%"
        )

    confidence_label.config(
        text="Top Predictions: " + " | ".join(top_results)
    )

    print(
        f"Prediction: {predicted_digit}, "
        f"Confidence: {confidence:.2f}%"
    )


# ==========================================
# STEP 9: CLEAR CANVAS
# ==========================================

def clear_canvas():

    global image_buffer, draw, last_x, last_y

    canvas.delete("all")

    image_buffer = Image.new(
        "L",
        (CANVAS_SIZE, CANVAS_SIZE),
        0
    )

    draw = ImageDraw.Draw(image_buffer)

    last_x = None
    last_y = None

    result_label.config(
        text="Predicted Digit: --",
        fg="white"
    )

    confidence_label.config(
        text="Draw a digit to begin"
    )


# ==========================================
# STEP 10: BUTTONS
# ==========================================

button_frame = tk.Frame(
    root,
    bg="#172033"
)

button_frame.pack(pady=18)


tk.Button(
    button_frame,
    text="Predict",
    command=predict_digit,
    font=("Arial", 11, "bold"),
    bg="#2878E0",
    fg="white",
    width=13,
    pady=8,
    cursor="hand2"
).pack(side=tk.LEFT, padx=8)


tk.Button(
    button_frame,
    text="Clear",
    command=clear_canvas,
    font=("Arial", 11, "bold"),
    bg="#D94B4B",
    fg="white",
    width=13,
    pady=8,
    cursor="hand2"
).pack(side=tk.LEFT, padx=8)


# ==========================================
# STEP 11: RESULT DISPLAY
# ==========================================

result_label = tk.Label(
    root,
    text="Predicted Digit: --",
    font=("Arial", 17, "bold"),
    bg="#172033",
    fg="white"
)

result_label.pack(pady=(5, 8))


confidence_label = tk.Label(
    root,
    text="Draw a digit to begin",
    font=("Arial", 9),
    bg="#172033",
    fg="#B9C5D8",
    wraplength=400
)

confidence_label.pack(padx=10)


# ==========================================
# STEP 12: START APPLICATION
# ==========================================

root.mainloop()
