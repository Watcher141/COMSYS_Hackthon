# CMOS Face Recognition

This repository contains solutions for **multiclass face recognition** and **binary face classification**, tailored for robust performance even with image distortions.

* `task_a.ipynb`: Binary classification (e.g., male vs female) using a simple CNN.
* `task_b.ipynb`: Multiclass face recognition using frozen FaceNet (InceptionResnetV1) + a custom classifier.

---

##  Setup Instructions

1. **Clone this repository**

```bash
git clone https://github.com/your-username/CMOS-Face-Recognition.git
cd CMOS-Face-Recognition
```

2. **Prepare the dataset** in the following structure:

### For `task_a.ipynb` (Binary Classification):

```
your_dataset/
├── train/
│   ├── male/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   ├── female/
│   │   ├── img3.jpg
│   │   └── img4.jpg
├── val/
│   ├── male/
│   ├── female/
```

### For `task_b.ipynb` (Multiclass Face Recognition):

```
your_dataset/
├── train/
│   ├── Person_1/
│   │   ├── Person_1.jpg
│   │   └── distortion/
│   │       ├── blur.jpg
│   │       └── noise.jpg
│   ├── Person_2/
│   │   ├── Person_2.jpg
│   │   └── distortion/
├── val/
│   ├── Person_1/
│   │   ├── Person_1.jpg
│   │   └── distortion/
│   │       └── noise.jpg
```

3. **Install the required packages:**

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install torch torchvision facenet-pytorch numpy pillow tqdm matplotlib scikit-learn opencv-python jupyter
```

4. **(Optional) If using Google Colab**

* Mount Google Drive to load data and save weights

5. **To run locally:**

* Make sure you have a working GPU (recommended) with CUDA installed
* Execute cells one-by-one in Jupyter Notebook or convert to a Python script using:

```bash
jupyter nbconvert --to script task_a.ipynb
python task_a.py
```

6. **Download Pretrained Weights** (if available):

| Model   | File                    | Description                   |
| ------- | ----------------------- | ----------------------------- |
| Model A | `binary_classifier.pth` | CNN trained for gender task   |
| Model B | `best_classifier.pth`   | FaceNet embeddings classifier |

>  Place weight files in the root or specify the correct path in your notebook.

---

##  Model Architectures

### 🔹 Model A (`task_a.ipynb`)

**Architecture:** Custom CNN for binary classification

* Multiple convolution + pooling layers
* Fully connected output layer for 2 classes
* Simpler baseline compared to FaceNet pipeline

### 🔶 Model B (`task_b.ipynb`)

**Architecture:** Frozen pretrained FaceNet + custom classifier

* **FaceNet (InceptionResnetV1)**:

  * Outputs 512-dimensional embedding per face
  * Pretrained on VGGFace2 dataset
  * Used as a frozen feature extractor

* **Classifier Head:**

```text
512 → Linear → 256
       ReLU
       Dropout(0.4)
256 → Linear → num_classes
```

* **Training Strategy:**

  * CrossEntropyLoss
  * Adam optimizer
  * Early stopping based on validation accuracy
  * Faces cropped via MTCNN

---

##  Inference

Once the best model is trained and saved (`best_classifier.pth` or `binary_classifier.pth`), you can run inference on any image:

```python
def predict_image(image_path):
    # Load and preprocess image
    # Detect face with MTCNN (for task_b)
    # Extract FaceNet embedding (for task_b)
    # Predict class with trained classifier

predict_image('path/to/test/image.jpg')
```

* Automatically detects face (for multiclass model)
* Uses the trained classifier to predict the class label

---

##  Model Weights

Store your pretrained weights in the project directory:

```text
CMOS-Face-Recognition/
├── task_a.ipynb
├── task_b.ipynb
├── binary_classifier.pth       <- Model A (CNN)
├── best_classifier.pth         <- Model B (FaceNet)
```

Ensure that these files are loaded correctly in the respective notebook using `torch.load()`.

---

##  Training Results

### Model A (`task_a.ipynb`) - Binary Classification

| Metric    | Train | Validation |
| --------- | ----- | ---------- |
| Accuracy  | 93.5% | 88.7%      |
| Precision | 92.0% | 87.0%      |
| Recall    | 94.0% | 89.1%      |
| F1-Score  | 93.0% | 88.0%      |

### Model B (`task_b.ipynb`) - Multiclass Face Recognition

| Metric    | Train | Validation |
| --------- | ----- | ---------- |
| Accuracy  | 96.2% | 91.8%      |
| Precision | 95.4% | 90.1%      |
| Recall    | 96.0% | 91.0%      |
| F1-Score  | 95.7% | 90.5%      |

*Note: These results are placeholders. Replace them with actual values after training.*

---

##  Repository Structure

| File           | Description                                  |
| -------------- | -------------------------------------------- |
| `task_a.ipynb` | Binary CNN for gender or other binary labels |
| `task_b.ipynb` | Multiclass face recognition using FaceNet    |
| `README.md`    | This documentation                           |
| `*.pth`        | Saved PyTorch model weights                  |

---

##  Notes

* Face detection is handled via `MTCNN` from `facenet-pytorch`
* Image distortions (blur, noise) are part of the training data
* Designed for high accuracy and robustness in face recognition tasks

---

##  Contact

For issues, questions, or contributions, feel free to open an issue or PR.

---
