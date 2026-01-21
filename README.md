# Multi-Label Face Mask & Sunglasses Detection using CNN

## 📌 Project Overview

This project implements a **multi-label Convolutional Neural Network (CNN)** to detect **face mask usage** and **sunglasses presence** simultaneously from facial images.

Unlike traditional single-label classification, this problem is modeled as a **multi-attribute vision task**, where each image can independently belong to multiple categories:

* Mask: Worn / Not Worn
* Sunglasses: Present / Absent

The model is trained and evaluated on a **real-world, noisy dataset** hosted on Kaggle, making it closer to practical computer vision applications.

---

## 🎯 Problem Statement

Given a facial image, predict:

1. Whether the person is **wearing a face mask**
2. Whether the person is **wearing sunglasses**

Each prediction is made **independently**, resulting in four possible combinations:

* Masked + Sunglasses
* Masked + No Sunglasses
* Unmasked + Sunglasses
* Unmasked + No Sunglasses

---

## 🧠 Why Multi-Label Classification?

Treating this as a flat 4-class classification problem would force artificial dependencies between mask and sunglasses.

Instead, this project uses **multi-label learning**, which:

* Reflects real-world conditions more accurately
* Allows independent prediction of attributes
* Scales easily to additional attributes (cap, helmet, etc.)

---

## 🗂️ Dataset Description

**Source:** Kaggle (Face Mask Detection with Sunglasses)

**Folder Structure:**

```
/kaggle/input/face-mask-detection/
├── plain-masked/
│   └── plain-masked/
├── plain-unmasked/
│   └── plain-unmasked/
├── sunglasses-masked/
│   └── sunglasses-masked/
└── sunglasses-unmasked/
    └── sunglasses-unmasked/
```

**Label Encoding:**

| Folder Name         | Mask | Sunglasses |
| ------------------- | ---- | ---------- |
| plain-masked        | 1    | 0          |
| sunglasses-masked   | 1    | 1          |
| plain-unmasked      | 0    | 0          |
| sunglasses-unmasked | 0    | 1          |

Each image is resized to **128×128** and normalized.

---

## 🏗️ Model Architecture

```
Input (128×128×3)
↓
Conv2D (32) → ReLU → MaxPooling
↓
Conv2D (64) → ReLU → MaxPooling
↓
Conv2D (128) → ReLU → MaxPooling
↓
Flatten
↓
Dense (128) → ReLU → Dropout (0.5)
↓
Dense (2) → Sigmoid
```

* **Sigmoid activation** is used to allow independent probabilities for each label
* **Binary Crossentropy** is used as the loss function

---

## ⚙️ Training Configuration

* Image Size: 128 × 128
* Batch Size: 32
* Optimizer: Adam
* Loss Function: Binary Crossentropy
* Epochs: 10
* Train/Validation Split: 80/20 (with fixed random seed)

---

## 📊 Evaluation Strategy

Accuracy alone is insufficient for multi-label problems.

This project evaluates:

* Precision & Recall **per attribute** (Mask / Sunglasses)
* F1-score

Predictions are thresholded at **0.5** for each output neuron.

---

## 🧪 Sample Prediction Output

```
Mask: Yes
Sunglasses: No
```

Each attribute is predicted independently.

---

## 🚀 Key Learnings & Concepts Demonstrated

* Multi-label CNN design
* Proper choice of activation and loss functions
* Handling real-world noisy image data
* Attribute-level evaluation metrics
* Practical dataset handling in Kaggle environment

---

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* NumPy
* Scikit-learn
* Kaggle Notebooks

---

## 🔮 Future Improvements

* Add data augmentation for better generalization
* Use transfer learning (MobileNetV2 / ResNet)
* Extend to object detection using annotations (YOLO)
* Deploy as a real-time webcam application
