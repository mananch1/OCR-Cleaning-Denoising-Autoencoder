"""
Document Denoising Web Interface (Inference Demo)
Deep Learning Mini-Project
"""

from flask import Flask, render_template, request, jsonify
from model import DenoisingAutoEncoder
from PIL import Image
from torchvision import transforms
import torch
import torch.nn.functional as F
import numpy as np
import io
import base64
import time
import os

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load trained model (with auto-reload on file modification)
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pth")
model = DenoisingAutoEncoder()
last_loaded_mtime = 0

def ensure_model_loaded():
    global last_loaded_mtime
    if os.path.exists(MODEL_PATH):
        mtime = os.path.getmtime(MODEL_PATH)
        if mtime != last_loaded_mtime:
            try:
                model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
                model.eval()
                last_loaded_mtime = mtime
                print(f"[OK] Loaded trained weights from {MODEL_PATH}")
            except Exception as e:
                print(f"[WARNING] Could not load {MODEL_PATH} ({e}). Train new weights for current architecture.")
    else:
        if last_loaded_mtime == 0:
            print(f"[WARNING] model.pth not found at {MODEL_PATH}")
            print("  The model will produce random outputs until you add the trained weights.")

ensure_model_loaded()

# ---------------------------------------------------------------------------
# Image transforms (must match training)
# ---------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


def img_to_base64(img):
    """Convert a PIL Image to a base64-encoded PNG string."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/denoise", methods=["POST"])
def denoise():
    """Accept an uploaded image, run DAE inference, return original + denoised as base64."""
    ensure_model_loaded()
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    try:
        img = Image.open(file.stream).convert("L")
    except Exception as e:
        return jsonify({"error": f"Invalid image: {str(e)}"}), 400

    # Keep original for display
    orig_w, orig_h = img.size
    orig_b64 = img_to_base64(img)

    # Run inference at 100% native resolution (pad to nearest multiple of 4)
    start = time.time()
    tensor = transforms.ToTensor()(img).unsqueeze(0)  # [1, 1, H, W]
    _, _, h, w = tensor.shape

    pad_h = (4 - h % 4) % 4
    pad_w = (4 - w % 4) % 4
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if pad_h > 0 or pad_w > 0:
        padded = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")
    else:
        padded = tensor

    with torch.no_grad():
        out_padded = model(padded)

    if pad_h > 0 or pad_w > 0:
        output = out_padded[:, :, pad_top:pad_top+h, pad_left:pad_left+w]
    else:
        output = out_padded
    elapsed = round((time.time() - start) * 1000, 1)

    # Convert output tensor back to PIL image at native resolution
    out_img = transforms.ToPILImage()(output.squeeze(0))
    denoised_b64 = img_to_base64(out_img)

    # OCR Binarized mode (clean threshold for text recognition pipelines)
    arr = np.array(out_img)
    # Thresholding: pixels brighter than 200 become pure white (255), rest pure black (0)
    binarized = (arr > 200).astype(np.uint8) * 255
    ocr_img = Image.fromarray(binarized)
    ocr_b64 = img_to_base64(ocr_img)

    return jsonify({
        "original": orig_b64,
        "denoised": denoised_b64,
        "ocr_cleaned": ocr_b64,
        "time_ms": elapsed,
    })


# ---------------------------------------------------------------------------
# Sample images endpoint (serve test images for quick demo)
# ---------------------------------------------------------------------------
@app.route("/samples")
def get_samples():
    """Return a list of available test images for quick demo."""
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test")
    if os.path.exists(test_dir):
        files = sorted([f for f in os.listdir(test_dir) if f.endswith(".png")])[:12]
        return jsonify({"samples": files})
    return jsonify({"samples": []})


@app.route("/samples/<filename>")
def serve_sample(filename):
    """Serve a test image file."""
    from flask import send_from_directory
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test")
    return send_from_directory(test_dir, filename)


if __name__ == "__main__":
    print("\n>>> Denoising AutoEncoder Demo")
    print("    Open http://localhost:5000 in your browser\n")
    app.run(debug=True, port=5000)
