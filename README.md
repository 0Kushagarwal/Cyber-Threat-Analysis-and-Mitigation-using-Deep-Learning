# 🛡️ Malware Detection using Deep Learning

A binary classification project that detects malware from system-level features using a Deep Neural Network (DNN) and compares performance with standard machine learning metrics.

---

## 📌 Overview

This project trains a deep learning model to classify processes or executables as either **benign** or **malware** based on a set of system/memory features extracted from process metadata. It uses TensorFlow/Keras for modeling and scikit-learn for preprocessing, evaluation, and metrics.

---

## 📂 Dataset

- **File:** `database.csv`
- **Target column:** `classification` — binary label with values `benign` and `malware`
- **ID column:** `hash` — dropped before training (not a feature)
- Several noisy/irrelevant columns are dropped during preprocessing:
  - `usage_counter`, `normal_prio`, `policy`, `vm_pgoff`, `task_size`, `cached_hole_size`, `hiwater_rss`, `nr_ptes`, `cgtime`, `signal_nvcsw`

---

## 🧠 Model Architecture

A fully connected feedforward neural network built with Keras:

| Layer | Units | Activation |
|-------|-------|------------|
| Input | 23    | —          |
| Dense | 16    | ReLU       |
| Dense | 8     | ReLU       |
| Dense | 4     | ReLU       |
| Output | 1   | Sigmoid    |

- **Loss:** Binary Crossentropy  
- **Optimizer:** RMSprop  
- **Epochs:** 5  
- **Batch Size:** 32

---

## ⚙️ Pipeline

1. Load dataset from `database.csv`
2. Drop irrelevant columns
3. Encode target: `benign → 0`, `malware → 1`
4. Split into train/test sets (60% / 40%)
5. Standardize features using `StandardScaler`
6. Train deep learning model
7. Evaluate using accuracy, F1 score, and confusion matrix

---

## 📊 Evaluation Metrics

- **Training Accuracy**
- **Test Accuracy**
- **F1 Score** (on test set)
- **Confusion Matrix** (visualized as a heatmap)

---

## 🗂️ Project Structure

```
├── database.csv          # Dataset (not included in repo)
├── malware_detection.py  # Main script
└── README.md
```

---

## 🔧 Requirements

Install dependencies with:

```bash
pip install pandas numpy scikit-learn tensorflow seaborn matplotlib imbalanced-learn
```

| Library | Purpose |
|--------|---------|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | Preprocessing, splitting, metrics |
| `tensorflow` / `keras` | Deep learning model |
| `seaborn` / `matplotlib` | Visualization |
| `imbalanced-learn` | SMOTE (imported, available for use) |

---

## 🚀 Usage

```bash
python malware_detection.py
```

Make sure `database.csv` is present in the same directory before running.

---

## 📈 Results

After training, the script prints:

```
Number of benign samples: ...
Number of malware samples: ...
Deep Learning - Training data accuracy: ...
Deep Learning - Test data accuracy: ...
F1 score: ...
```

And displays a confusion matrix heatmap.

---

## 📝 Notes

- `SMOTE` from `imbalanced-learn` is imported and available for handling class imbalance if needed.
- The model uses a fixed `random_state=42` for reproducibility.
- `input_dim=23` assumes 23 features after dropping columns — adjust if your dataset differs.

---
