# <div align="center">

# 🧠 Bias-Aware Multi-View Complementary Label Learning for Image Classification

![Python](https://img.shields.io/badge/Python-3.x-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red)
![Scikit-learn](https://img.shields.io/badge/ML-Scikit--learn-orange)
![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-green)
![Status](https://img.shields.io/badge/Status-Initial%20Results%20Available-yellow)
![Type](https://img.shields.io/badge/Type-Independent%20Research%20Project-lightgrey)

An experimental framework for image classification under **biased complementary-label supervision**, combining CNN features, Histogram of Oriented Gradients (HOG) and RGB colour histograms.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/azmattanveer409/bias-aware-multiview-learning/blob/main/notebooks/Bias_Aware_Multi_View_Learning.ipynb)

</div>

---

<!-- PROJECT BANNER -->
<p align="center">
  <img width="1340" height="1024" alt="Bias-Aware Multi-View README Infographic" src="https://github.com/user-attachments/assets/01d471d6-8790-4dc3-8bc4-c6de30361440" />
</p>

---

## 📌 1. Overview

This project investigates **bias-aware complementary-label learning** for image classification through multiple visual feature representations.

In conventional supervised learning, each training image is associated with its correct class label. **Complementary-label learning** instead provides information indicating that an image does *not* belong to a particular class (for example, "this image is not a truck").

The project studies how a **non-uniform (biased)** complementary-label distribution influences learning, and compares a **naive** objective, a **bias-aware** objective that uses a transition matrix, and a **supervised reference** model. Each image is described through three views: learned CNN features, HOG descriptors and RGB colour histograms.

> **Scope:** this is an independent academic experiment on a controlled synthetic setting. It is **not** a peer-reviewed publication and **not** a claim of a novel algorithm.

---

## 🎯 2. Research Objectives

1. Implement complementary-label learning for CIFAR-10 image classification.
2. Simulate biased complementary labels through a predefined transition matrix.
3. Implement naive and bias-aware (transition-matrix-aware) learning objectives.
4. Combine handcrafted (HOG, colour) and deep (CNN) visual representations.
5. Compare multi-view learning with single-view configurations.
6. Evaluate performance with accuracy, macro precision, macro recall, macro F1-score and confusion matrices.

---

## 🗂️ 3. Dataset

### CIFAR-10

The experiments use **CIFAR-10**, a widely used image-classification benchmark.

| Property | Value |
|---|---|
| Total images | 60,000 |
| Original training set | 50,000 |
| Test set | 10,000 |
| Classes | 10 |
| Image size | 32 × 32 pixels, RGB |

**Splits used in the experiment:** the 50,000 training images are divided into a **45,000-image training subset** and a **5,000-image validation subset**. The **10,000-image test set** is used for the reported results.

<p align="center">
  <img width="1184" height="497" alt="CIFAR-10 sample images" src="https://github.com/user-attachments/assets/26ceeba0-b7b1-4cda-8a7e-05c9d93c09f9" />
</p>

<p align="center">
  <img width="1036" height="800" alt="Additional dataset visualisation" src="https://github.com/user-attachments/assets/cf2248eb-ae48-4328-b4c2-4d65f41eb96f" />
</p>

The dataset is downloaded automatically through `torchvision.datasets.CIFAR10`.

---

## 🧩 4. Multi-View Feature Representations

Three different representations of each image are used to investigate the contribution of complementary information.

<p align="center">
  <img width="800" height="400" alt="Multi-view feature extraction" src="https://github.com/user-attachments/assets/aeeb7739-3772-45b6-9705-f3d0b13dec1e" />
</p>

| View | Representation | Captures |
|---|---|---|
| **CNN** | Learned visual features | Visual patterns and spatial structure |
| **HOG** | 1,764-dimensional descriptor | Local edge orientation, shape |
| **RGB histogram** | 48-dimensional descriptor | Colour distribution (16 bins × 3 channels) |

### 4.1 CNN features
A **Convolutional Neural Network** learns visual representations directly from the image pixels.

### 4.2 Histogram of Oriented Gradients (HOG)
A handcrafted descriptor that summarises local gradient and edge-orientation information, giving a representation of image structure and shape.

### 4.3 RGB colour histograms
The distribution of pixel intensities in the Red, Green and Blue channels, giving information about colour content but not shape.

### 4.4 Multi-view fusion
The three representations are combined by **feature concatenation (early fusion)** into one joint representation. This is a simple baseline for studying the contribution of each view, not a specialised multi-view learning method.

---

## 🏷️ 5. Complementary-Label Generation

Complementary labels are **generated synthetically** from the true CIFAR-10 labels using a predefined **transition matrix** $T$:

$$
T_{y,c} = P(\bar{Y} = c \mid Y = y)
$$

where

- $Y$ is the true class,
- $\bar{Y}$ is the complementary class (a class the image does *not* belong to),
- $T_{y,c}$ is the probability that an image of true class $y$ receives complementary label $c$.

The matrix assigns **zero probability to the true class** and distributes the remaining probability over incorrect classes.

**Bias setting:** the experiment uses a **bias strength of 0.75**. For each true class, one selected incorrect class receives a higher probability of being assigned as the complementary label, so the labels are systematically non-uniform. The exact construction of the matrix is defined in the notebook.

> **Important:** the transition matrix is predefined and **known** to the bias-aware model. The experiment therefore evaluates learning under *known synthetic bias*. It does **not** show that an unknown real-world transition matrix can be estimated.

---

## 📐 6. Learning Objectives

### 6.1 Naive complementary-label learning

The naive objective penalises the probability the model assigns to the observed complementary class:

$$
\mathcal{L}_{\mathrm{naive}} = -\log\left(1 - p(c \mid x)\right)
$$

- $x$: input image
- $c$: observed complementary class
- $p(c \mid x)$: predicted probability of class $c$

This corresponds to assuming that annotators choose wrong classes **uniformly**.

### 6.2 Bias-aware complementary-label learning

The bias-aware objective uses the transition matrix to model the probability of observing a complementary label:

$$
P(\bar{Y} = c \mid x) = \sum_{y} T_{y,c} \cdot P(Y = y \mid x)
$$

$$
\mathcal{L}_{\mathrm{bias}} = -\log P(\bar{Y} = c \mid x)
$$

This objective accounts for the specified label-generation distribution. It follows the general idea of forward correction with a transition matrix (see References).

### 6.3 Supervised reference

The supervised multi-view model is trained with standard cross-entropy on the **true** labels. It is a **reference**, not a directly equivalent complementary-label method, because it uses more informative supervision.

---

## 🔬 7. Experimental Methodology

<p align="center">
  <img width="700" alt="Research methodology flowchart" src="https://github.com/user-attachments/assets/5ccc4e77-5058-4425-835d-ec6f7f9624aa" />
</p>

1. **Dataset acquisition:** download and prepare CIFAR-10.
2. **Feature extraction:** extract CNN, HOG and RGB colour representations.
3. **Complementary-label generation:** generate labels with the predefined biased transition matrix (bias strength 0.75).
4. **Model development:** implement the naive and bias-aware objectives.
5. **Multi-view learning:** concatenate the view representations (early fusion).
6. **Training:** train the models for **5 epochs** (single recorded configuration).
7. **Evaluation:** compare models on the 10,000-image test set.
8. **Robustness analysis (planned):** repeat over random seeds and several bias strengths.

---

## 🧪 8. Experimental Comparisons

| Model | Views | Labels / objective |
|---|---|---|
| **Supervised Multi-View** | CNN + HOG + colour | True labels (reference) |
| **Naive Multi-View** | CNN + HOG + colour | Complementary, naive objective |
| **Bias-Aware Multi-View** | CNN + HOG + colour | Complementary, bias-aware objective |
| **Bias-Aware CNN-Only** | CNN | Complementary, bias-aware objective |
| **Bias-Aware HOG-Only** | HOG | Complementary, bias-aware objective |
| **Bias-Aware Colour-Only** | RGB histogram | Complementary, bias-aware objective |

---

## 📊 9. Results

Test-set results from the notebook (**one configuration, 5 epochs, bias strength 0.75**):

| Model | Test Accuracy | Macro F1 |
|---|---:|---:|
| Supervised Multi-View *(reference, true labels)* | 68.10% | 0.6706 |
| **Bias-Aware Multi-View** | **65.45%** | **0.6563** |
| Bias-Aware HOG-Only | 57.05% | 0.5675 |
| Bias-Aware CNN-Only | 55.88% | 0.5561 |
| Bias-Aware Colour-Only | 34.07% | 0.3337 |
| Naive Multi-View | 20.95% | 0.1445 |

### 9.1 Model performance comparison

<p align="center">
  <img width="1200" height="957" alt="Model performance comparison" src="https://github.com/user-attachments/assets/a7ed7d84-09fc-4a13-a57e-6a63380eb130" />
</p>

*Figure: Test accuracy of the evaluated models.*

### 9.2 Confusion matrix

<p align="center">
  <img width="625" height="547" alt="Confusion matrix" src="https://github.com/user-attachments/assets/76915c78-a8cf-4e3e-bde1-5e0932c3f4d3" />
</p>

*Figure: Confusion matrix of the selected model on the test set.*

### 9.3 Observations

- The bias-aware multi-view model reached **65.45%** accuracy, **2.65 percentage points below** the supervised reference (68.10%).
- It was **8.40 points above** the best single-view model (HOG-only, 57.05%) and **9.57 points above** the CNN-only model (55.88%), which suggests that combining views helped in this setting.
- The **naive** multi-view model reached only **20.95%**, far below the bias-aware model. This is consistent with the naive objective's uniform-annotator assumption being violated by the biased labels.
- Colour alone was the weakest bias-aware single view (34.07%), as expected for a descriptor that ignores shape.

### 9.4 How to read these results

- They come from **one run and one bias strength**; no variance across random seeds is reported yet, and no significance test was performed.
- Training lasted **only 5 epochs**, so the CNN branch is likely under-trained; the ranking of CNN-only versus HOG-only should **not** be read as CNN being inferior.
- Labels are **synthetic** and the transition matrix is **known** to the bias-aware model.
- The supervised model uses true labels and is shown only as a reference.

> These are initial results. Robustness across seeds and bias strengths is still to be evaluated.

---

## 🛠️ 10. Technologies and Tools

| Technology | Application |
|---|---|
| **Python** | Programming language |
| **PyTorch** | Neural networks and training |
| **NumPy** | Numerical computing |
| **Pandas** | Data handling and result tables |
| **Scikit-learn** | Metrics and evaluation |
| **Scikit-image** | HOG feature extraction |
| **Matplotlib** | Visualisation |
| **Google Colab** | Development and experimentation |

---

## 📁 11. Repository Structure

```text
bias-aware-multiview-learning/
│
├── README.md
├── Bias_Aware_Multi_View_Learning.ipynb
├── requirements.txt
│
├── data/
│ ├── X_COLOR.npy
│ ├── X_HOG.npy
│ └── Y_TRAIN.npy
│
└── results/
    └── model_comparison.csv

```

---

## ♻️ 12. Reproducibility

1. Open the notebook in Google Colab (badge above) or run it locally.
2. Install the requirements: `pip install -r requirements.txt`.
3. Run all cells in order. CIFAR-10 is downloaded automatically.

The notebook documents the dataset preparation, feature extraction, label generation, transition matrix, model architecture, training settings and evaluation. CIFAR-10 and large intermediate feature arrays are **not** stored in this repository.

---

## ⚠️ 13. Limitations

1. **Synthetic complementary labels** generated from the true labels.
2. **Known transition matrix:** the work does not estimate an unknown matrix from real data.
3. **Single recorded configuration:** one bias strength (0.75), 5 training epochs, no repeated seeds.
4. **Short training:** the CNN branch may be under-trained.
5. **Fusion:** early fusion by concatenation is a simple baseline.
6. **Dataset:** CIFAR-10 images are small and low-resolution; results should not be generalised to other datasets or real annotators.
7. The supervised model uses true labels and is not directly comparable to the complementary-label models.

---

## 🚀 14. Future Work

- [ ] Evaluate several bias strengths.
- [ ] Repeat all experiments over multiple random seeds (mean ± standard deviation).
- [ ] Test a misspecified transition matrix (assumed ≠ true bias).
- [ ] Investigate estimating an unknown transition matrix.
- [ ] Explore other fusion strategies.
- [ ] Test other datasets and real human-annotated complementary labels.
- [ ] Extract the notebook into reusable modules and write a short research report.

---

## 📚 15. Related Work and References

The bias-aware objective follows the idea of learning with a transition matrix for complementary labels. This project is a small empirical study and does not claim to extend these works.

- Ishida, T., Niu, G., Hu, W., Sugiyama, M. *Learning from Complementary Labels.* NeurIPS 2017.
- Yu, X., Liu, T., Gong, M., Tao, D. *Learning with Biased Complementary Labels.* ECCV 2018.
- Tang, L., Yan, Y., Tian, Y., Pardalos, P. M. *Self-adaptive label discovery and multi-view fusion for complementary label learning.* Neural Networks 181, 2025, 106763.
- Tang, L., Liu, Y., Tian, Y., Pardalos, P. M. *Complementary label learning with multi-view data and a semi-supervised labeling mechanism.* Pattern Recognition 165, 2025, 111651.
- Krizhevsky, A. *Learning Multiple Layers of Features from Tiny Images.* 2009 (CIFAR-10).

---

## 👤 16. Author

**Azmat Tanveer**
BS Mathematics, Abbottabad University of Science and Technology, Pakistan

🔗 GitHub: [@azmattanveer409](https://github.com/azmattanveer409)

---

## 🔎 17. Research Areas

`Artificial Intelligence` · `Machine Learning` · `Computer Vision` · `Image Processing` · `Image Classification` · `Multi-View Learning` · `Complementary Label Learning` · `Statistical Learning`

---

*This repository documents an academic experimental implementation. It is not presented as a peer-reviewed publication or a claim of a novel algorithm.*
ical computing |
| **Pandas** | Data processing |
| **Scikit-learn** | Machine learning and evaluation |
| **Scikit-image** | HOG feature extraction |
| **Matplotlib** | Data visualisation |
| **Google Colab** | Development and experimentation |

---

## 11. Repository Structure

```text
bias-aware-multiview-learning/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── images/
│   ├── project-banner.png
│   ├── cifar10-samples.png
│   ├── feature-views.png
│   ├── methodology.png
│   │
│   └── results/
│       ├── model-comparison.png
│       └── confusion-matrix.png
│
├── notebooks/
│   └── complete_experiment.ipynb
│
├── src/
│   ├── feature_extraction.py
│   ├── complementary_labels.py
│   ├── models.py
│   ├── losses.py
│   ├── training.py
│   └── evaluation.py
│
├── results/
│   └── model_comparison.csv
│
└── reports/
    └── research_report.pdf
```

---

## 12. Reproducibility

The complete experimental notebook will document:

- Dataset acquisition and preparation.
- Feature extraction procedures.
- Complementary-label generation.
- Transition matrix configuration.
- Model architectures.
- Training configurations.
- Evaluation procedures.

Random seeds, experimental settings and model configurations should be documented to support reproducibility.

The CIFAR-10 dataset and large intermediate feature arrays are **not** included in this repository. They are downloaded or generated during execution.

---

## 13. Experimental Limitations

The current experimental framework has several limitations:

1. **Synthetic complementary labels:** Complementary labels are generated using a predefined transition matrix.
2. **Known bias distribution:** The current implementation does not establish the ability to estimate an unknown transition matrix from real-world data.
3. **Dataset limitations:** CIFAR-10 contains relatively small, low-resolution images.
4. **Feature fusion:** The current multi-view approach uses feature concatenation as an early-fusion baseline.
5. **Generalisation:** Results from one dataset and one simulation setting should not be generalised to all complementary-label learning problems.

These limitations will be considered when interpreting the experimental findings.

---

## 14. Research Status

**Current status:** *Experimental implementation and evaluation in progress.*

The project aims to develop a reproducible experimental framework for investigating bias-aware complementary-label learning with multiple visual feature representations.

Further experiments, comparative analysis and robustness evaluation will be documented as they are completed.

---

## 15. Author

**Azmat Tanveer**

BS Mathematics
Abbottabad University of Science and Technology, Pakistan

GitHub: [@azmattanveer409](https://github.com/azmattanveer409)

---

## 16. Research Areas

- Artificial Intelligence
- Machine Learning
- Computer Vision
- Image Processing
- Image Classification
- Multi-View Learning
- Complementary Label Learning
- Statistical Learning

---

*This repository documents an academic experimental implementation. It is not presented as a peer-reviewed publication or a claim of a novel algorithm.*
