# Automatic Recognition of Microbial Life

## Overview

This repository contains research and development work focused on the real-time identification of bacterial particles in bioaerosols using fluorescence spectroscopy and machine learning.

The project builds upon the work of Alejandro Fontal, Xavier Rodó, and collaborators at ISGlobal, who demonstrated that Laser-Induced Fluorescence (LIF) data collected by the Rapid-E bioaerosol sensor can be used to classify bacterial species in real time using machine learning techniques.

The primary goal of this repository is to reproduce, evaluate, and extend the original classification pipeline while exploring more advanced machine learning and deep learning approaches for microbial identification in airborne particles.

---

## Research Objectives

This work is organized around two major research themes:

### 1. Improved Bacterial Classification

* Reproduction of the original Random Forest baseline classifier
* Evaluation of alternative classical machine learning models
* Development of deep learning approaches for direct analysis of fluorescence spectra
* Development of multimodal models that combine multiple Rapid-E data sources

### 2. Classification Under Mixed-Aerosol Conditions

* Evaluation of classifier robustness under bacterial mixtures
* Assessment of performance under distribution shift
* Estimation of aerosol mixture composition from particle-level predictions
* Investigation of real-world deployment scenarios

---

## Repository Structure

```text
Automatic-Recognition-of-Microbial-Life/

├── data/
│   ├── raw/
│   ├── processed/
│   ├── metadata/
│   └── realtime_mock/
│
├── models/
│   ├── trained/
│   └── model_registry.csv
│
├── reports/
│   ├── notebooks/
│   └── figures/
│
├── results/
│
├── scripts/
│
├── src/
│   └── lif_thesis/
│       ├── data/
│       ├── evaluation/
│       ├── experiments/
│       ├── models/
│       └── realtime/
│
├── dashboard/
│
└── api/
```

---

# Project Components

## 1. Offline Experiments

The offline experimentation framework is used to train, evaluate, and compare machine learning models on previously collected Rapid-E datasets.

Current experiments include:

* Random Forest baseline reproduction
* Classical machine learning model comparisons
* 1D Convolutional Neural Networks (CNNs)
* Multimodal deep learning architectures
* Mixture classification studies

Results from each experiment are stored within the `results/` directory.

---

## 2. Real-Time Mock Pipeline

A simulated real-time inference pipeline is provided to mimic the behavior of a deployed Rapid-E system.

The pipeline performs:

1. Detection of incoming particle files
2. Data validation and preprocessing
3. Model inference
4. Prediction storage
5. Aggregated mixture estimation

This environment allows deployment testing without requiring direct access to a live Rapid-E instrument.

---

## 3. Dashboard

A dashboard interface is being developed to visualize predictions in real time.

Planned functionality includes:

* Live particle counts
* Species-level predictions
* Prediction confidence monitoring
* Mixture composition estimation
* Spectral visualization
* Historical trend analysis

The dashboard is intended to demonstrate how Rapid-E classification models could be integrated into operational monitoring systems.

---

## 4. API and Deployment

The repository is being structured to support future deployment through a REST API.

Planned endpoints include:

* Single-particle prediction
* Batch prediction
* Real-time monitoring
* Mixture composition summaries
* Model health monitoring

This deployment layer is intended to support integration with future environmental monitoring applications.

---

# Baseline vs. New Contributions

## Original ISGlobal Baseline

The original project demonstrated bacterial classification using:

* Rapid-E bioaerosol sensor measurements
* Laser-Induced Fluorescence (LIF)
* Hand-engineered features
* Random Forest classification

The baseline implementation serves as the primary benchmark reproduced within this repository.

## New Contributions

This repository extends the original work through:

* Reproduction and validation of the published baseline
* Exploration of additional machine learning algorithms
* Development of deep learning approaches
* Development of multimodal architectures
* Evaluation under mixed-bacterial aerosol conditions
* Real-time inference pipeline development
* Dashboard and deployment infrastructure
* Investigation of model robustness under distribution shift

---

# Data Access

The datasets used in this project originate from research conducted at ISGlobal and associated collaborators.

Due to data ownership, licensing, and research restrictions, raw datasets are not currently distributed through this repository.

To reproduce experiments, users must obtain access to the appropriate Rapid-E datasets through the original project collaborators or authorized data providers.

Additional instructions regarding data preparation and directory structure will be provided in future releases.

---

# Reproducing Results

## Environment Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

```bash
source .venv/bin/activate
```

or on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -e .
```

---

## Data Preparation

Prepare processed datasets:

```bash
python scripts/prepare_data.py
```

---

## Running Experiments

Run an experiment:

```bash
python scripts/run_experiment.py --experiment exp04_baseline_cnn
```

Evaluate a trained model:

```bash
python scripts/evaluate.py
```

---

## Running the Real-Time Pipeline

```bash
python scripts/run_realtime_mock.py
```

---

## Launching the Dashboard

```bash
streamlit run dashboard/app.py
```

---

# Citation and Attribution

This repository builds upon research conducted by Alejandro Fontal, Xavier Rodó, and collaborators at ISGlobal.

If you use this repository or its derived work, please cite both the original publication and this thesis work where appropriate.

### Original Publication

Fontal, A., Rodó, X., et al.

*Laser-Induced Fluorescence coupled with Machine Learning as an effective approach for real-time identification of bacteria in bioaerosols.*

Atmospheric Measurement Techniques, 2025.

DOI: https://doi.org/10.5194/amt-18-7297-2025

---

# Thesis Information

**Author:** Christopher Perry

**Institution:** Universitat de Barcelona

**Program:** Master's in Data Science

**Research Partner:** ISGlobal

**Thesis Topic:** Real-Time Identification of Bacterial Bioaerosols Using Fluorescence Spectroscopy and Machine Learning

---

# License

Please refer to the LICENSE file for licensing information.

Original datasets, publications, and associated intellectual property remain the property of their respective authors and institutions.
