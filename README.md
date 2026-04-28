# Polymer Coil–Globule Transition Analysis using Simulation and Machine Learning

## 📌 Overview

This project explores the **coil–globule transition of polymer chains** using molecular simulation and machine learning techniques. By varying the interaction parameter (ε), the system transitions between expanded (*coil*) and compact (*globule*) states.

The project aims to:

* Classify polymer configurations into coil or globule states
* Identify the **θ-point (critical transition value)**

---

## 🎯 Objectives

* Perform polymer simulations using LAMMPS input scripts
* Analyze structural properties such as:

  * Radius of gyration (Rg)
  * End-to-end distance (Re)
* Train ML models to classify polymer states
* Determine the θ-point using model predictions

---

## ⚙️ Project Workflow

### 1. Simulation

* Input file: `polymer_coil_globule.in`
* Job execution: `run_job.sh`
* Simulations performed across varying ε values

### 2. Data Processing

* Extracted features:

  * Radius of Gyration (Rg)
  * End-to-End Distance (Re)

### 3. Machine Learning

* Logistic Regression
* Support Vector Machine (RBF Kernel)

---

## 📊 Results

### Logistic Regression

* Accuracy: **67.36%**
* High recall for globule, poor detection of coil states

### SVM (RBF Kernel)

* Accuracy: **79.35%**
* Balanced performance across both classes

✅ **Best Model:** SVM (RBF Kernel)

---

## 🔬 θ-Point Results

* θ-point ε = **0.928**
* Below ε → **Coil state**
* Above ε → **Globule state**

Additional:

* θ-point Rg = 8.0746
* Mean Re used = 4.131

---

## 📁 Project Structure

```id="r9xk2a"
iiche_projects/
│
├── polymer_project/
│   ├── polymer_coil_globule.in   # LAMMPS input script
│   └── run_job.sh                # Script to run simulation
│
└── polymer_results/
    ├── epsilon_results/          # Results for different ε values
    ├── ml_outputs/               # Model outputs (metrics, logs)
    └── polymer_ml/               # ML scripts / processed data
```

---

## ▶️ How to Run

### Step 1: Run Simulation

```id="f82k1d"
cd polymer_project
bash run_job.sh
```

### Step 2: Process Data & Run ML

```id="c1v8pw"
cd ../polymer_results/polymer_ml
polymer_ml.py
```

---

## 📦 Dependencies

* Python 3.x
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* LAMMPS (for simulation)

---

## 📌 Key Insights

* Logistic Regression struggled with class imbalance (poor coil recall)
* SVM significantly improved classification balance
* θ-point accurately identified at ε = 0.928

---

## 🚀 Future Work

* Explore deep learning models
* Increase dataset size for better generalization
* Include additional structural features
* Perform temperature-dependent analysis

---

## 👨‍💻 Author

Samyak Kothari
Smita Singh
Daksh Vaya
Bharti Sehra

---
