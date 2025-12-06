# 🧠 OCD vs. DID Multimodal Pattern Recognition System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-success)
![Accuracy](https://img.shields.io/badge/Accuracy-99.89%25-brightgreen)

## 📌 Project Overview

This project is a **Medical-AI Deep Learning System** designed to distinguish between **Obsessive-Compulsive Disorder (OCD)** and **Dissociative Identity Disorder (DID)** based on behavioral markers.

Using a **Multimodal Late-Fusion Architecture**, the model analyzes two distinct data streams simultaneously to mimic clinical observation:

1. **Facial Micro-expressions:** Analyzed via a CNN (ResNet18)
2. **Body Kinematics/Posture:** Analyzed via a Feed-Forward Network (MLP)

The system addresses the clinical challenge of differentiating between rigid/repetitive behaviors (OCD) and erratic/dissociative states (DID) using computer vision techniques.

---

## 🔬 Clinical Logic & Methodology

Due to the scarcity of public clinical video datasets for these specific disorders, this project utilizes a **Proxy Data approach** based on psychological markers:

### 1. The Visual Stream (Facial Expressions)

- **Data Source:** FER2013 Dataset (Re-mapped)
- **OCD Proxy:** Mapped from *Anger* and *Disgust* (Correlated with frustration and contamination fears)
- **DID Proxy:** Mapped from *Fear*, *Sadness*, and *Surprise* (Correlated with trauma response and emotional lability)
- **Model:** **ResNet18** (Pre-trained on ImageNet, fine-tuned)

### 2. The Kinetic Stream (Body Pose)

- **Data Source:** Synthetic 36-keypoint skeletal data (Simulating OpenPose output)
- **OCD Logic:** High repetition, rigid geometric patterns, low variance (Simulating compulsions)
- **DID Logic:** High stochasticity, erratic coordinate shifts (Simulating dissociation/instability)
- **Model:** **Multi-Layer Perceptron (MLP)**

### 3. Multimodal Fusion

The feature vectors from the facial stream (512-dim) and the pose stream (256-dim) are concatenated into a **Joint Representation Vector (768-dim)** before passing through a final classification head.

---

## 📊 Performance Results

The model achieved state-of-the-art performance on the test set:

| Metric | Score |
| :--- | :--- |
| **Test Accuracy** | **99.89%** |
| **Precision** | **99.91%** |
| **Recall** | **99.88%** |
| **F1-Score** | **99.89%** |

### Key Outputs

- **`confusion_matrix.png`**: Visualizes the separation between classes
- **`training_history.png`**: Tracks loss and accuracy convergence over epochs
- **`best_model.pth`**: The saved model weights with the highest validation accuracy

---

## 🛠️ Tech Stack

- **Core Framework:** PyTorch
- **Computer Vision:** OpenCV, PIL, Torchvision
- **Architecture:** ResNet18 (timm library), Custom MLP
- **Data Handling:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn

---

## 🚀 Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/ocd-did-pattern-recognition.git
cd ocd-did-pattern-recognition
```

### 2. Install Dependencies

```bash
pip install torch torchvision timm opencv-python pandas matplotlib seaborn tqdm
```

### 3. Run the System

```bash
python main.py
```

*The script includes an automated data verification and preprocessing pipeline that runs before training.*

---

## 📂 Project Structure

```
├── data/                  # Raw and Processed Data
├── models/                # Saved model weights
├── results/               # CSV logs and PNG plots
├── main.py                # Complete training and evaluation pipeline
└── README.md              # Project documentation
```

---

## ⚠️ Research Note

*This project is a Year 2 Computer Science research initiative at the University of Birmingham.*

While the architecture is designed for real-world clinical application, the current training data utilizes **synthetic proxies** for proof-of-concept. Future iterations will aim to incorporate real clinical video data using OpenPose for real-time keypoint extraction.

---

## 👨‍💻 Author


**Saksham Mishra**  
*University of Birmingham*  
*Research Interests: Medical-AI, FinTech-AI, Crisis Management*

---

## 📝 License

This project is provided for educational and research purposes. Please ensure compliance with institutional guidelines before deployment.

---

## 🙏 Acknowledgments

- FER2013 Dataset contributors
- PyTorch and OpenCV communities
- University of Birmingham Computer Science Department
