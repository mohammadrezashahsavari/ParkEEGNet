# ParkEEGNet: A Transformer-based model for Parkinson's Disease Detection

This repository contains the official implementation of the paper **"ParkEEGNet: A Deep Learning Framework for Parkinson's Disease Diagnosis via EEG Functional Connectivity and Spatio-Temporal Modeling"**.

![Graphical Abstract](https://github.com/mohammadrezashahsavari/ParkEEGNet/blob/main/Images%20&%20Diagrams/ParkEEGNET%20-%20Graphical%20Abstract.jpg?raw=true)

---
## Project Overview

Accurate early-stage diagnosis of Parkinson's Disease (PD) remains a major challenge due to symptom variability and the absence of definitive biomarkers. While electroencephalography (EEG) offers a promising non-invasive diagnostic modality, existing EEG-based methods often rely on handcrafted features or limited spatio-temporal modeling, which restricts their performance and generalizability. We propose **ParkEEGNet**, a novel deep learning framework designed for automated PD detection using raw EEG signals.

ParkEEGNet integrates three key components:
1.  **Cross-Channel Correlation Learning (CCL)** to capture inter-channel dependencies as functional connectivity patterns.
2.  **Attentive Spatio-Temporal Representation Learning (ASTRL)** to extract spatio-temporal features while emphasizing diagnostically relevant temporal segments via additive attention.
3.  A **classification module** for final prediction.

The model was trained and validated using three publicly available datasets: UC San Diego (training and validation), PRED-CT, and UI (used as independent holdout sets for testing generalizability). ParkEEGNet achieved **100% accuracy** on the UC San Diego dataset, **97.4% accuracy** on the PRED-CT dataset, and **84.4% accuracy** on the UI dataset, outperforming existing state-of-the-art methods.

---

## Model Architecture

The proposed ParkEEGNet architecture consists of three main modules: the Cross-Channel Correlation (CCL) module, the Attentive Spatio-Temporal Representation Learning (ASTRL) module, and a final Classifier module.

![Model Architecture](https://github.com/mohammadrezashahsavari/ParkEEGNet/blob/main/Images%20%26%20Diagrams/PD%20-%20Transformer%20-%20Model%20Architecture.jpg)
*Figure 2 from the paper: An overview of the proposed deep learning architecture.*

-   **Cross-Channel Correlation (CCL) Module**: This module uses multi-head self-attention to learn functional connectivity patterns between EEG channels. It processes a 2-second segment of 32-channel EEG data and outputs encoded data that incorporates information from other channels.
-   **Attentive Spatio-Temporal Representation Learning (ASTRL) Module**: This module processes both the raw EEG signals and the encoded signals from the CCL module. It uses two stacked Bi-LSTM layers to capture temporal dependencies and an additive attention mechanism to focus on the most diagnostically relevant time steps.
-   **Classifier Module**: The features from the CCL and ASTRL modules are integrated and passed through two fully connected layers to produce a final classification (PD or Healthy Control).

---

## Datasets

This study utilized three publicly available EEG datasets:

1.  **UC San Diego Dataset**: Includes recordings from 15 PD patients and 16 healthy controls using a 32-channel BioSemi system. This dataset was used for training and validation.
2.  **PRED-CT Dataset**: Contains resting-state EEG recordings from 28 PD patients using a 64-channel Brain Vision system. This was used as a hold-out test set.
3.  **University of Iowa (UI) Dataset**: Includes EEG recordings from 14 PD patients and 14 healthy controls using a 64-channel Brain Vision system. This was also used as a hold-out test set to evaluate generalizability.

---

## Repository Structure
```bash
.
├── data/
│   ├── raw/                  # Directory for raw dataset files before any processing
│   │   ├── uc_san_diego/
│   │   ├── pred_ct/
│   │   └── ui/
│   └── processed/            # Directory for preprocessed, segmented data ready for training
│       ├── uc_san_diego/
│       ├── pred_ct/
│       └── ui/
│
├── src/
│   ├── data_preprocessing.py # (Formerly Preprocessing.py) Handles the initial processing of raw EEG data.
│   ├── datasets.py           # (Formerly part of utils.py) Contains functions for loading and splitting processed datasets.
│   ├── models.py             # (Combined from Models/) Defines the main ParkEEGNet architecture (Transformer + BiLSTM + Attention).
│   ├── legacy_models.py      # (Combined from Models/) Contains other architectures you compared against (VGG, ResNet, etc.).
│   ├── experiment_setup.py   # (Formerly experiments.py) Manages experiment configurations, training loops, and evaluation logic.
│   └── utils.py              # (Formerly utils.py) Holds utility functions like plotting, metrics calculation, etc.
│
├── main.py                   # (Formerly Main.py) The main entry point to run experiments.
├── requirements.txt          # Lists all project dependencies for easy installation.
├── results/
    ├── figures/              # Directory for saved plots, like attention maps and connectivity graphs.
    │   ├── attention_maps/
    │   └── connectivity_maps/
    ├── logs/                 # Directory for storing training logs and performance metrics.
    └── trained_models/       # Directory for saving the final, trained model weights (.h5 files).
```
---

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mohammadrezashahsavari/ParkEEGNet.git
    cd ParkEEGNet
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```
    The main dependencies are `tensorflow`, `numpy`, `scipy`, `matplotlib`, and `pyedflib`.

---

## How to Run

### 1. Data Preprocessing

The `Preprocessing.py` script is designed to segment raw EEG data from the datasets into 2-second epochs. You will need to download the datasets and place them in the `Data` directory. Modify the paths in `Preprocessing.py` accordingly.

### 2. Training and Evaluation

The `Main.py` script is the entry point for all experiments. You can modify this file to run different experiments as described in the paper.

-   **Prepare the data:** The `exp.prepare()` function loads and preprocesses the datasets.

-   **Train the model with 10-fold cross-validation:**
    ```python
    # In Main.py
    exp = Experiment(base_project_dir, 0, 'Transformer', exp_name='10fold-32channel')
    exp.prepare()
    exp.train_10fold()
    ```

-   **Reproduce results and visualize attention:**
    To evaluate a pre-trained model and generate attention plots, use the `reproduce_results_on_10fold` method. Set `plot_attention_weights=True` to visualize the additive attention maps from the ASTRL module, and `plot_self_attention_weights=True` to visualize the functional connectivity maps from the CCL module.
    ```python
    # In Main.py
    exp = Experiment(base_project_dir, 0, 'Transformer', exp_name = '10fold-32channel')
    exp.prepare()
    exp.reproduce_results_on_10fold(evaluation_set='test', plot_attention_weights=True, plot_self_attention_weights=True)
    ```

-   **Evaluate on hold-out datasets:**
    ```python
    # In Main.py
    exp = Experiment(base_project_dir, 0, 'Transformer', exp_name = '10fold-SIT-32channel')
    exp.prepare()
    exp.evalute_on_PRED_CT(model_number=5) # Evaluate the 5th model of the fold
    exp.evalute_on_UI(model_number=5)
    ```

---

## Results

Our proposed ParkEEGNet model demonstrates state-of-the-art performance across all three datasets. The subject-independent 10-fold cross-validation results are summarized below:

| Datasets | AUC | ACC | Sens | Spec | PPV | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **UC San Diego** | 0.998 | 0.990 | 1 | 0.982 | 0.981 | 0.990 |
| **PRED-CT*** | - | 0.974 | - | - | - | - |
| **UI** | 0.925 | 0.844 | 0.781 | 0.906 | 0.891 | 0.832 |
*Table IV from the paper, showing performance on the three datasets.*

### Visualizations

The interpretability of ParkEEGNet is one of its key features.

-   **Functional Connectivity (from CCL module):** The model learns to distinguish between the functional connectivity patterns of healthy controls and PD patients. As shown below, PD patients exhibit reduced connectivity, particularly in the frontal, temporal, and parietal lobes.

    ![Functional Connectivity](https://github.com/mohammadrezashahsavari/ParkEEGNet/blob/main/Images%20%26%20Diagrams/Healthy%20vs%20Parkinson%20Functional%20Connectivity.png)
    *Figure 3 from the paper: Functional connectivity maps for a healthy subject (left) and a PD patient (right).*

-   **Additive Attention (from ASTRL module):** The attention maps highlight the specific time segments in the EEG signals that the model found most important for its diagnosis.

    ![Additive Attention](https://github.com/mohammadrezashahsavari/ParkEEGNet/blob/main/Images%20%26%20Diagrams/Aditive%20Attention%20Maps.png)
    *Figure 4 from the paper: Additive attention maps showing key time intervals for a healthy subject (left) and a PD patient (right).*

