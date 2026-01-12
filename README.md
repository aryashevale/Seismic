# 🌍 Next-Gen Seismic & Structural Health Monitoring System

![Project Status](https://img.shields.io/badge/Status-Prototype_Complete-success)
![Tech Stack](https://img.shields.io/badge/Tech-Python_%7C_PyTorch_%7C_Unity_%7C_C%23-blue)

## 📖 Overview
This project presents a novel **Non-Destructive Testing (NDT)** framework designed to detect underground anomalies and structural faults using Deep Learning. 

Addressing the scarcity of real-world fault data, the system utilizes a **Quantum-inspired synthetic data generator** to train a Convolutional Neural Network (CNN). The results are visualized in real-time using an **interactive 3D holographic dashboard** built in Unity, effectively creating a "Digital Twin" of the subsurface environment.

## 🚀 Key Features
* **Quantum-Inspired Data Generation:** Synthesized 2,000+ seismic traces using randomized wave propagation parameters to simulate rare fault scenarios.
* **AI-Powered Detection:** Custom 1D CNN model achieved **78.6% validation accuracy** in classifying signal anomalies.
* **3D Holographic Dashboard:** A Unity-based visualization tool that renders sensor data as a color-coded, interactive 3D grid.
* **Real-World Validation:** Framework designed to integrate with MEMS vibration sensors and drone flight logs (validated on NIT Rourkela campus data).

## 🛠️ Tech Stack
* **Data Processing & AI:** Python, NumPy, PyTorch (CNN), Scikit-Learn.
* **Visualization:** Unity Game Engine (2022/6000 LTS), C#, TextMeshPro.
* **Data Bridge:** JSON serialization between Python and Unity.

## 📊 Methodology
1.  **Signal Synthesis:** A Python script generates synthetic seismic waves (Ricker wavelets) to mimic fault reflections.
2.  **Model Training:** A PyTorch CNN is trained to distinguish between "Safe" layers (Green) and "Fault" layers (Red).
3.  **Export:** The model predicts outcomes for new sensor data and exports a `unity_data.json` file.
4.  **Visualization:** The Unity dashboard parses the JSON and dynamically spawns a 3D grid, elevating "Fault" blocks as red glowing spikes for easy identification.

## 💻 Installation & Usage

### Prerequisites
* Python 3.8+
* Unity Hub & Editor (2021.3 or later)

### Step 1: Generate Data & Run AI
Navigate to the `Seismic` folder and run the Jupyter Notebook/Script:
```bash
python CNN_Model.py

## 📸 Screenshots
Here is the 3D Holographic Dashboard visualizing the fault lines (Red) vs safe zones (Green):

![Unity Dashboard View](dashboard_view.jpg)
