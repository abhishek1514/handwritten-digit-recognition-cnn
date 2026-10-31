
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torchvision import transforms


# ==========================================
# 1. DEVICE AND MODEL PATH
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = Path(__file__).resolve().parent / "digit_cnn.pth"


# ==========================================
# 2. CNN ARCHITECTURE
# Must match train_model.py exactly
# ==========================================

class DigitCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# ==========================================
# 3. LOAD TRAINED MODEL
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
print("Device:", device)


# ==========================================
# 4. IMAGE TRANSFORMATION
# ==========================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


# ==========================================
# 5. IMAGE PREPROCESSING FUNCTION
# ==========================================

def preprocess_digit(image):

    # Find digit boundaries
    bbox = image.getbbox()

    if bbox is None:
        return None

    # Crop only the drawn digit
    cropped = image.crop(bbox)

    # Add padding around the digit
    width, height = cropped.size

    padding = 20

    padded = Image.new(
        "L",
        (width + padding * 2, height + padding * 2),
        0
    )

    padded.paste(cropped, (padding, padding))

    # Resize while maintaining aspect ratio
    padded.thumbnail(
        (20, 20),
        Image.Resampling.LANCZOS
    )

    # Create 28x28 black image
    final_image = Image.new("L", (28, 28), 0)

    # Center digit
    x = (28 - padded.width) // 2
    y = (28 - padded.height) // 2

    final_image.paste(padded, (x, y))

    return transform(final_image)


# ==========================================
# 6. TKINTER WINDOW
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
# 7. DRAWING CANVAS
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


# PIL image stores the actual drawing
image_buffer = Image.new(
    "L",
    (CANVAS_SIZE, CANVAS_SIZE),
    0
)

draw = ImageDraw.Draw(image_buffer)

last_x = None
last_y = None


# ==========================================
# 8. DRAWING FUNCTIONS
# ==========================================

def start_draw(event):

    global last_x, last_y

    last_x = event.x
    last_y = event.y


def draw_digit(event):

    global last_x, last_y

    if last_x is not None and last_y is not None:

        # Visible canvas drawing
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

        # PIL image drawing
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

        # Rounded brush ending
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
# 9. PREDICTION FUNCTION
# ==========================================

def predict_digit():

    if image_buffer.getbbox() is None:

        messagebox.showwarning(
            "Empty Canvas",
            "Please draw a digit first."
        )

        return

    # Preprocess handwritten image
    input_tensor = preprocess_digit(image_buffer)

    if input_tensor is None:
        return

    # Add batch dimension
    input_tensor = input_tensor.unsqueeze(0).to(device)

    # Model prediction
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

    confidence = top_probabilities[0, 0].item() * 100

    # Display main prediction
    result_label.config(
        text=f"Predicted Digit: {predicted_digit}",
        fg="#70F0A0"
    )

    # Display top 3 predictions
    top_results = []

    for digit, probability in zip(
        top_digits[0].tolist(),
        top_probabilities[0].tolist()
    ):

        top_results.append(
            f"{digit}: {probability * 100:.2f}%"
        )

    confidence_label.config(
        text=(
            f"Top Predictions: {' | '.join(top_results)}"
        )
    )

    print(
        f"Prediction: {predicted_digit} | "
        f"Confidence: {confidence:.2f}%"
    )


# ==========================================
# 10. CLEAR CANVAS
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
# 11. BUTTONS
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
# 12. RESULT DISPLAY
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
# 13. START APPLICATION
# ==========================================

root.mainloop()
