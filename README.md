# Handwritten Character Recognition

A CNN-based system that recognizes handwritten digits and characters, trained on the MNIST and EMNIST-Balanced datasets. Includes an interactive drawing demo where you can write a character on a canvas and get a real-time prediction.

Built as part of the **CodeAlpha** internship program.

---

## Demo

Draw a character on the canvas and click **Predict** — the model returns the top 5 guesses with confidence scores.

![demo](https://raw.githubusercontent.com/jak004/CodeAlphaHandwrittenCharacterRecognition/main/demo.png)

---

## Model Architecture

`CharCNN` — a lightweight convolutional neural network:

| Layer block | Details |
|---|---|
| Conv Block 1 | Conv2d(1→32) × 2, BatchNorm, ReLU, MaxPool, Dropout2d |
| Conv Block 2 | Conv2d(32→64) × 2, BatchNorm, ReLU, MaxPool, Dropout2d |
| Classifier | Flatten → Linear(3136→512) → ReLU → Dropout → Linear(512→N) |

- Input: 28×28 grayscale image
- Output: N class logits (10 for MNIST, 47 for EMNIST-Balanced)

---

## Datasets

| Dataset | Classes | Description |
|---|---|---|
| MNIST | 10 | Handwritten digits (0–9) |
| EMNIST-Balanced | 47 | Digits + uppercase + visually distinct lowercase letters |

Both datasets are auto-downloaded by PyTorch on first run.

---

## Project Structure

```
.
├── model.py          # CharCNN architecture
├── train.py          # Training script
├── predict.py        # Interactive drawing demo (tkinter)
├── requirements.txt
├── model_mnist.pth   # Pre-trained MNIST weights
└── model_emnist.pth  # Pre-trained EMNIST-Balanced weights
```

---

## Setup

```bash
git clone https://github.com/jak004/CodeAlphaHandwrittenCharacterRecognition.git
cd CodeAlphaHandwrittenCharacterRecognition
pip install -r requirements.txt
```

---

## Usage

### Run the interactive demo (pre-trained model included)

```bash
# EMNIST model (digits + letters) — default
python predict.py

# MNIST model (digits only)
python predict.py --model model_mnist.pth
```

### Train from scratch

```bash
# Train on EMNIST-Balanced (47 classes)
python train.py --dataset emnist --epochs 15

# Train on MNIST (10 classes)
python train.py --dataset mnist --epochs 10

# Full options
python train.py --dataset emnist --epochs 20 --batch-size 128 --lr 0.001
```

Training saves the best checkpoint (by validation accuracy) automatically.

---

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision, numpy, Pillow, matplotlib, scikit-learn

Install all with:
```bash
pip install -r requirements.txt
```

---

## License

This project is open source and available under the [MIT License](LICENSE).
