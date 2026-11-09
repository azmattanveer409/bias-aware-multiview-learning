Bias-Aware Multi-View Complementary Label Learning for Image Classification

"Python" (https://img.shields.io/badge/Python-3.x-blue)
"PyTorch" (https://img.shields.io/badge/Framework-PyTorch-red)
"Dataset" (https://img.shields.io/badge/Dataset-CIFAR--10-green)
"Status" (https://img.shields.io/badge/Status-Research%20in%20Progress-orange)

Overview

This project investigates bias-aware complementary-label learning for image classification through multiple visual feature representations.

In conventional supervised learning, each training image is associated with its correct class label. Complementary-label learning instead provides information indicating that an image does not belong to a particular class.

The project investigates how non-uniform complementary-label distributions influence learning and examines the role of a bias-aware probabilistic objective.

Research Objectives

1. Implement complementary-label learning for image classification.
2. Investigate the effect of biased complementary-label distributions.
3. Implement a transition-matrix-aware learning objective.
4. Combine handcrafted and deep visual feature representations.
5. Compare multi-view and single-view learning configurations.
6. Evaluate classification performance using standard metrics.

Dataset

CIFAR-10

- 60,000 RGB images.
- 50,000 training images.
- 10,000 test images.
- 10 object categories.
- Image resolution: 32 × 32 pixels.

The dataset is downloaded through the torchvision dataset interface.

Multi-View Feature Representations

1. CNN Features

A convolutional neural network learns visual representations directly from image data.

2. Histogram of Oriented Gradients

HOG descriptors capture local gradient and edge-orientation information.

3. RGB Colour Histograms

Colour histograms represent the distribution of pixel intensities across RGB channels.

The multi-view baseline combines these representations through feature concatenation.

Complementary-Label Generation

Complementary labels are synthetically generated using a predefined transition matrix:

P(complementary label = c | true class = y)

The matrix assigns zero probability to the true class and distributes the remaining probability across incorrect classes.

Different bias strengths can be investigated to evaluate the effect of non-uniform complementary-label distributions.

Learning Objectives

Naive Complementary-Label Learning

The naive objective penalises the predicted probability assigned to the observed complementary class.

Bias-Aware Complementary-Label Learning

The bias-aware objective models the probability of observing a complementary class through a predefined transition matrix.

The two objectives are compared under controlled experimental conditions.

Experimental Comparisons

The experimental framework includes:

- Supervised multi-view reference model.
- Naive complementary-label multi-view model.
- Bias-aware complementary-label multi-view model.
- Bias-aware CNN-only model.
- Bias-aware HOG-only model.
- Bias-aware colour-only model.

Evaluation Metrics

- Classification accuracy.
- Macro precision.
- Macro recall.
- Macro F1-score.
- Confusion matrices.
- Robustness across different random seeds and bias strengths.

Technologies

Python, PyTorch, NumPy, Pandas, Scikit-learn, Scikit-image, Matplotlib and Google Colab.

Experimental Limitations

Complementary labels are synthetically generated using a predefined transition matrix. The current experiment therefore evaluates learning under controlled, known bias. It does not establish the ability to estimate an unknown transition matrix from real-world data.

Furthermore, CIFAR-10 is a relatively small, low-resolution image dataset. Results should be interpreted within this experimental setting.

Repository Structure

bias-aware-multiview-learning/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── notebooks/
│   └── complete_experiment.ipynb
├── src/
│   ├── feature_extraction.py
│   ├── complementary_labels.py
│   ├── models.py
│   ├── losses.py
│   ├── training.py
│   └── evaluation.py
├── results/
└── reports/

Reproducibility

The complete notebook will document dataset acquisition, feature extraction, complementary-label generation, model training, and evaluation.

Experimental settings, random seeds, transition matrices and model configurations should be reported to support reproducibility.

Author

Azmat Tanveer
BS Mathematics
Abbottabad University of Science and Technology, Pakistan

GitHub: https://github.com/azmattanveer409

Research Areas

Artificial Intelligence | Machine Learning | Computer Vision | Image Classification | Multi-View Learning | Complementary Label Learning

---

This repository documents an academic experimental implementation. It is not presented as a peer-reviewed publication or a claim of a novel algorithm.
