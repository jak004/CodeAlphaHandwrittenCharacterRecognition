"""
Interactive drawing demo for handwritten character recognition.

Usage:
    python predict.py                          # uses model_emnist.pth
    python predict.py --model model_mnist.pth  # uses MNIST model

Draw a character on the canvas, then click Predict.
"""

import argparse
import sys
import tkinter as tk

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

from model import CharCNN


class App:
    CANVAS_PX = 280      # display canvas size
    MODEL_PX  = 28       # model input size
    BRUSH     = 18       # brush stroke width

    def __init__(self, root: tk.Tk, model: CharCNN, class_names: list, device: torch.device, dataset: str):
        self.root        = root
        self.model       = model
        self.class_names = class_names
        self.device      = device
        self.dataset     = dataset

        root.title("Handwritten Character Recognition — CodeAlpha")
        root.resizable(False, False)
        root.configure(bg="#1e1e2e")

        # ── Canvas ──────────────────────────────────────────────────────
        self.canvas = tk.Canvas(root, width=self.CANVAS_PX, height=self.CANVAS_PX,
                                bg="black", cursor="crosshair", highlightthickness=2,
                                highlightbackground="#7c3aed")
        self.canvas.pack(padx=20, pady=(20, 8))

        self._reset_image()
        self._last = None

        self.canvas.bind("<B1-Motion>",        self._paint)
        self.canvas.bind("<ButtonRelease-1>",  lambda e: setattr(self, "_last", None))

        # ── Buttons ──────────────────────────────────────────────────────
        btn_frame = tk.Frame(root, bg="#1e1e2e")
        btn_frame.pack(pady=4)

        tk.Button(btn_frame, text="  Predict  ", command=self._predict,
                  bg="#7c3aed", fg="white", font=("Arial", 12, "bold"),
                  relief="flat", padx=12, pady=6).pack(side=tk.LEFT, padx=8)

        tk.Button(btn_frame, text="  Clear  ", command=self._clear,
                  bg="#ef4444", fg="white", font=("Arial", 12, "bold"),
                  relief="flat", padx=12, pady=6).pack(side=tk.LEFT, padx=8)

        # ── Result labels ────────────────────────────────────────────────
        self.lbl_result = tk.Label(root, text="Draw a character and click Predict",
                                   font=("Arial", 18, "bold"), bg="#1e1e2e", fg="white")
        self.lbl_result.pack(pady=(10, 2))

        self.lbl_top = tk.Label(root, text="", font=("Courier", 10),
                                bg="#1e1e2e", fg="#a0a0c0", wraplength=340, justify="center")
        self.lbl_top.pack(pady=(0, 16))

    # ── Drawing helpers ──────────────────────────────────────────────────

    def _reset_image(self):
        self._img  = Image.new("L", (self.CANVAS_PX, self.CANVAS_PX), 0)
        self._draw = ImageDraw.Draw(self._img)

    def _paint(self, event):
        x, y = event.x, event.y
        if self._last:
            lx, ly = self._last
            self.canvas.create_line(lx, ly, x, y, width=self.BRUSH,
                                    fill="white", capstyle=tk.ROUND, smooth=True)
            self._draw.line([lx, ly, x, y], fill=255, width=self.BRUSH)
        else:
            r = self.BRUSH // 2
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="")
            self._draw.ellipse([x - r, y - r, x + r, y + r], fill=255)
        self._last = (x, y)

    def _clear(self):
        self.canvas.delete("all")
        self._reset_image()
        self._last = None
        self.lbl_result.config(text="Draw a character and click Predict")
        self.lbl_top.config(text="")

    # ── Prediction ───────────────────────────────────────────────────────

    def _preprocess(self) -> torch.Tensor:
        img = self._img.filter(ImageFilter.GaussianBlur(radius=1))
        img = img.resize((self.MODEL_PX, self.MODEL_PX), Image.LANCZOS)

        arr = np.array(img, dtype=np.float32) / 255.0

        # EMNIST images are stored transposed vs. natural drawing orientation.
        # torchvision corrects this automatically during training, so we mirror
        # that correction here to keep inference consistent with training.
        if self.dataset == "emnist":
            arr = np.transpose(arr)

        arr = (arr - 0.5) / 0.5                              # match training normalisation
        return torch.tensor(arr).unsqueeze(0).unsqueeze(0).to(self.device)

    def _predict(self):
        if self._img.getbbox() is None:
            self.lbl_result.config(text="Canvas is empty — draw something first!")
            return

        tensor = self._preprocess()
        with torch.no_grad():
            logits = self.model(tensor)
            probs  = torch.softmax(logits, dim=1)[0]
            k      = min(5, len(self.class_names))
            top    = torch.topk(probs, k)

        best_char = self.class_names[top.indices[0].item()]
        best_prob = top.values[0].item() * 100

        self.lbl_result.config(text=f'Prediction: "{best_char}"   ({best_prob:.1f}%)')

        lines = [
            f'"{self.class_names[i.item()]}"  {p.item()*100:.1f}%'
            for i, p in zip(top.indices, top.values)
        ]
        self.lbl_top.config(text="Top guesses:  " + "   |   ".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Handwritten character recognition demo")
    parser.add_argument("--model", default="model_emnist.pth",
                        help="Path to saved model checkpoint (default: model_emnist.pth)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        ckpt = torch.load(args.model, map_location=device, weights_only=False)
    except FileNotFoundError:
        print(f"[ERROR] Model file '{args.model}' not found.")
        print("        Run  python train.py  first to train and save the model.")
        sys.exit(1)

    model = CharCNN(num_classes=ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    root = tk.Tk()
    App(root, model, ckpt["class_names"], device, ckpt["dataset"])
    root.mainloop()


if __name__ == "__main__":
    main()
