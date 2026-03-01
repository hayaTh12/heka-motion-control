# Exoskeleton Hip Kinematics Prediction — BioMAT Adaptation

Adaptation of the [BioMAT framework](https://github.com/MohsenSharifi1991/BioMAT) for real-time hip joint kinematics prediction from IMU signals, integrated into the HEKA lower-limb exoskeleton control pipeline at Polytechnique Montréal.

---

## Overview

This project adapts and extends BioMAT — an open-source Transformer-based framework for biomechanical joint angle prediction — to meet the real-time inference constraints of the HEKA powered exoskeleton. The model predicts **hip flexion angle** from wearable IMU data across multiple locomotion modes (level ground, ramp, stairs), enabling the exoskeleton to generate assistive motor torque commands in real time.

The original BioMAT paper: [Sharifi-Renani et al., *Sensors*, 2023](https://www.mdpi.com/1424-8220/23/13/5778)

---

## Key Contributions

- Adapted the BioMAT training pipeline to the HEKA exoskeleton's IMU configuration (thigh + trunk sensors)
- Configured predictive horizon segmentation for real-time forward estimation of hip kinematics
- Tuned Transformer hyperparameters (depth, attention heads, dropout) for exoskeleton deployment constraints
- Integrated a **Temporal Weighted MSE loss** to prioritize near-future predictions critical for torque control
- Evaluated model performance across 6 locomotion activities: level ground, ramp ascent/descent, stair ascent/descent

---

## Model Architecture

The core model is a **Time Series Transformer (TST)** adapted from tsai:
- **Input:** IMU signals (accelerometer + gyroscope) from thigh and trunk sensors
- **Output:** Predicted hip flexion angle (`hip_flexion_r`) over a 20-sample horizon
- **Architecture:** 3-layer Transformer, 16 attention heads, d_model=128, d_ff=256
- **Loss:** Temporal Weighted MSE (decaying weights over prediction horizon)

Supported model variants: `transformertsai`, `bilstm`, `Hernandez2021cnnlstm`

---

## Project Structure

```
Heka-motion-control/
├── configs/               # Training configurations (model, dataset, hyperparameters)
├── model/                 # Transformer, BiLSTM, CNN-LSTM architectures
│   └── core/tst/          # Time Series Transformer core modules
├── preprocessing/         # IMU filtering, segmentation, normalization
├── loading/               # Dataset loading utilities
├── loss/                  # Temporal weighted loss functions
├── visualization/         # W&B and Streamlit plotting
├── train.py               # Training loop
├── test.py                # Evaluation
├── main_universal.py      # Main entry point
└── run_dataset_prepration.py  # Dataset preprocessing pipeline
```

---

## Setup

```bash
git clone https://github.com/<your-username>/exoskeleton-locomotion-classifier.git
cd exoskeleton-locomotion-classifier
pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.7 · PyTorch ≥ 1.8.1 · CUDA-enabled GPU

---

## Usage

**1. Prepare the dataset**
```bash
python run_dataset_prepration.py
```

**2. Hyperparameter tuning**

Set `"tuning": true` in `configs/camargo_config.json`, then run:
```bash
python main_universal.py
```
Alternatively, use a W&B sweep for a more systematic search:
```bash
python run_sweep.py
```

**3. Train with best hyperparameters**

Set `"tuning": false` in `configs/camargo_config.json`, update the hyperparameters with the best values found, then run:
```bash
python main_universal.py
```

Dataset source: [Camargo et al. — Georgia Tech Epic Lab](https://www.epic.gatech.edu/opensource-biomechanics-camargo-et-al/)

---

## Tech Stack

**Python · PyTorch · scikit-learn · W&B · Streamlit · NumPy**

---

## Credits

Built on top of [BioMAT](https://github.com/MohsenSharifi1991/BioMAT) by Sharifi-Renani et al. and the [tsai](https://github.com/timeseriesAI/tsai) Time Series Transformer library.

Developed as part of the **HEKA Exoskeleton Project** — Polytechnique Montréal.
