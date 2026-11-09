# Bias-Aware Multi-View Complementary Label Learning for Image Classification



![Python](https://img.shields.io/badge/Python-3.x-blue)




![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red)




![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-green)




![Status](https://img.shields.io/badge/Status-Research%20in%20Progress-orange)



---

<!-- PROJECT BANNER: Upload your banner as images/project-banner.png -->


![Project Banner]

<img width="1340" height="1024" alt="Bias-Aware Multi-View README Infographic" src="https://github.com/user-attachments/assets/01d471d6-8790-4dc3-8bc4-c6de30361440" />



---

## 1. Overview

This project investigates **bias-aware complementary-label learning** for image classification through multiple visual feature representations.

In conventional supervised learning, each training image is associated with its correct class label. **Complementary-label learning** instead provides information indicating that an image does *not* belong to a particular class.

The project investigates how non-uniform complementary-label distributions influence learning and examines the role of a bias-aware probabilistic objective.

The proposed experimental framework combines deep learning and handcrafted visual representations to investigate the potential benefits of multi-view learning.

---

## 2. Research Objectives

The primary objectives of this project are:

1. Implement complementary-label learning for image classification.
2. Investigate the effect of biased complementary-label distributions.
3. Implement a transition-matrix-aware learning objective.
4. Combine handcrafted and deep visual feature representations.
5. Compare multi-view and single-view learning configurations.
6. Evaluate classification performance using standard evaluation metrics.

---

## 3. Dataset

### CIFAR-10

The experiments use the **CIFAR-10** dataset, a widely used benchmark for image classification.

**Dataset characteristics:**

- **Total images:** 60,000
- **Training images:** 50,000
- **Test images:** 10,000
- **Number of classes:** 10
- **Image dimensions:** 32 × 32 pixels
- **Colour channels:** RGB

<!-- DATASET IMAGE: Upload as images/cifar10-samples.png -->


![CIFAR-10 Dataset Samples]<img width="1184" height="497" alt="download" src="https://github.com/user-attachments/assets/26ceeba0-b7b1-4cda-8a7e-05c9d93c09f9" />
<img width="1036" height="800" alt="XoYbaOlVEQyjrvS5B0sIBKHPWXZEZle8mMSaJVxPlJDfbUDE_GzokKm7bRvyuWSSdPgpTGSOilWtDZOkfbIpL5WzVti4UMIlaQet7nMBH7Lv3WjQ4i52cVlmSPW-mRUNqtNO1TP7kKQEzbt0TkDUkE-WFqEzVg0x3hlLOeW_Ztxx92awajbySfwjb8T-h6EY" src="https://github.com/user-attachments/assets/cf2248eb-ae48-4328-b4c2-4d65f41eb96f" />




The dataset is downloaded through the `torchvision.datasets.CIFAR10` interface.

---

## 4. Multi-View Feature Representations

The framework combines three different visual feature representations to investigate the contribution of complementary information.

<!-- FEATURE EXTRACTION IMAGE: Upload as images/feature-views.png -->


![Multi-View Feature Extraction]<img width="800" height="400" alt="pz5MNSuqfbwqSsWsayLCbn50P4om7upPD_t-o-JvOcS-ACT8q0_aWuVUJIKjoGUc9NNRz5PxxgB9soGcEPmL1aHgIyDZ6Kci3_HggMkeZRqB-4g9eWusE8aRN3RCizYBa03b1hhbwp4chxQsFx4hz66dQ_fFhQ2AypEhvwQoL_EcRlfjrI_VcUyX3WnbM97P" src="https://github.com/user-attachments/assets/aeeb7739-3772-45b6-9705-f3d0b13dec1e" />


### 4.1 CNN Features

A **Convolutional Neural Network (CNN)** learns visual representations directly from image data.

The CNN branch extracts learned features that capture visual patterns and spatial information.

### 4.2 Histogram of Oriented Gradients (HOG)

**Histogram of Oriented Gradients (HOG)** is a handcrafted feature extraction technique that captures local gradient and edge-orientation information.

It provides a representation of image structure and shape characteristics.

### 4.3 RGB Colour Histograms

**RGB colour histograms** represent the distribution of pixel intensities across the three colour channels:

- Red
- Green
- Blue

They provide information about the colour distribution of an image.

### 4.4 Multi-View Feature Fusion

The three feature representations are combined through **feature concatenation (early fusion)** to construct a joint representation.

This provides a baseline for investigating the contribution of different visual feature views.

---

## 5. Complementary-Label Generation

Complementary labels are synthetically generated using a predefined **transition matrix**.

The transition matrix is defined as:

$$
T_{y,c} = P(\bar{Y} = c \mid Y = y)
$$

Where:

- $Y$ represents the true class.
- $\bar{Y}$ represents the complementary class.
- $T_{y,c}$ represents the probability of assigning complementary class $c$ to an image belonging to true class $y$.

The transition matrix assigns zero probability to the true class and distributes the remaining probability across incorrect classes.

Different bias strengths can be investigated to evaluate the effect of non-uniform complementary-label distributions.

> **Important experimental consideration:** The transition matrix is predefined and known in the current controlled simulation. The experiment therefore evaluates learning under *known synthetic bias* rather than estimating an unknown real-world transition matrix.

---

## 6. Learning Objectives

The project investigates two complementary-label learning objectives.

### 6.1 Naive Complementary-Label Learning

The naive objective penalises the predicted probability assigned to the observed complementary class.

Its mathematical formulation is:

$$
\mathcal{L}_{naive} = -\log\left(1 - p(c \mid x)\right)
$$

Where:

- $x$ represents an input image.
- $c$ represents its observed complementary class.
- $p(c \mid x)$ represents the model's predicted probability for class $c$.

### 6.2 Bias-Aware Complementary-Label Learning

The bias-aware objective incorporates the predefined transition matrix to model the probability of observing a complementary class.

The complementary-label probability is:

$$
P(\bar{Y} = c \mid x) = \sum_y T_{y,c} \, P(Y = y \mid x)
$$

The corresponding learning objective is:

$$
\mathcal{L}_{bias} = -\log P(\bar{Y} = c \mid x)
$$

This objective accounts for the specified complementary-label generation distribution.

---

## 7. Experimental Methodology

The experimental workflow consists of the following stages:

<!-- METHODOLOGY FLOWCHART: Upload as images/methodology.png -->


![Research Methodology]<img width="2361" height="2885" alt="AfJ-y-rJ8_L087dhHSEMbYqA5Xuh4XQoig-a_oFT5PflctLKJt0cvZdTt25wMo25ylyQ3EuoukOWgN16RHppJsjljQpwpPYKAa2oK5gY3njG6kRYlvXPzpdt_kxk4nfyvkp698e38m9G_87PaGrGOUTWaXFIzmFf4jywa6rI2aQckyrUabRjaJjFw478T8wU" src="https://github.com/user-attachments/assets/5ccc4e77-5058-4425-835d-ec6f7f9624aa" />





1. **Dataset acquisition:** Download and prepare CIFAR-10.
2. **Feature extraction:** Extract CNN, HOG and RGB colour features.
3. **Complementary-label generation:** Generate complementary labels using a predefined transition matrix.
4. **Model development:** Implement naive and bias-aware learning objectives.
5. **Multi-view learning:** Combine different visual feature representations.
6. **Model training:** Train the models using the selected learning objectives.
7. **Evaluation:** Compare model performance using standard metrics.
8. **Robustness analysis:** Investigate performance under different random seeds and bias strengths.

---

## 8. Experimental Comparisons

The experimental framework includes the following configurations:

| Model | Description |
|---|---|
| **Supervised Multi-View** | Reference model trained using true class labels |
| **Naive Complementary Multi-View** | Multi-view model using the naive complementary-label objective |
| **Bias-Aware Multi-View** | Main experimental model using the bias-aware objective |
| **Bias-Aware CNN-Only** | Uses CNN features |
| **Bias-Aware HOG-Only** | Uses HOG features |
| **Bias-Aware Colour-Only** | Uses RGB colour features |

The comparisons are intended to investigate the contribution of bias-aware learning and different feature representations.

---

## 9. Evaluation Metrics

Model performance is evaluated using the following metrics:

- **Classification Accuracy:** Measures the proportion of correctly classified images.
- **Precision:** Measures the correctness of positive predictions.
- **Recall:** Measures the proportion of actual class instances correctly identified.
- **Macro F1-Score:** Evaluates the balance between precision and recall across classes.
- **Confusion Matrix:** Provides a class-wise comparison of actual and predicted labels.

Additional experiments will investigate robustness across multiple random seeds and complementary-label bias strengths.

### 9.1 Model Performance Comparison

<!-- RESULTS IMAGE: Add your actual accuracy comparison chart here -->


![Model Performance Comparison]<img width="1200" height="957" alt="fhuUvETcwW8_0TcYJwiv3IeiWTheotXlABASU0e2Kz4I9zI7KivqqixHrwOBNpjIjmq6POQ6g02PQsvF-KbZEboM1N_o_sCI1e8FB6eU1Q6rZwkLqOWPnGGIqIyHnat11cS3yEiOnx87ISJCKygNclAZDE-J0rlPYCjVoIZMuuGjt8v1apRGg8okg8KC-lUs" src="https://github.com/user-attachments/assets/a7ed7d84-09fc-4a13-a57e-6a63380eb130" />


*Figure: Comparative classification performance of the evaluated models.*

### 9.2 Confusion Matrix

<!-- CONFUSION MATRIX: Add your actual confusion matrix here -->


![Confusion Matrix]<img width="625" height="547" alt="download" src="https://github.com/user-attachments/assets/76915c78-a8cf-4e3e-bde1-5e0932c3f4d3" />



*Figure: Confusion matrix of the selected model on the test dataset.*

> **Note:** These figures will be added after the corresponding experiments have been completed and their results verified.

---

## 10. Technologies and Tools

The project uses the following technologies:

| Technology | Application |
|---|---|
| **Python** | Programming language |
| **PyTorch** | Deep learning and neural networks |
| **NumPy** | Numerical computing |
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
