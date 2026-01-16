# Neural Networks & Machine Learning — Coursework + Projects 🧠⚡

A curated collection of homework assignments and side projects from a Neural Networks / Machine Learning course.

The focus is **clarity, reproducibility, and clean results** (because the only thing worse than a buggy model is a buggy model with no README).

---

## Contents

### Homework
- **HW_1 — LIF Neuron (Leaky Integrate-and-Fire)** (`homework/tarea1_lif/`)
  - How parameters affect LIF neuron dynamics.
- **HW_2 — Frog Visual Stimulus Simulation** (`homework/tarea2_frog_visual/`)
  - Numerical integration (Euler), stimulus selection, hysteresis, and a simple motor scheme.
- **HW_3 — Hopfield Network (Digits 0–9)** (`homework/tarea3_hopfield/`)
  - Associative memory: store and recover noisy digit patterns.
- **HW_4 — Perceptron (Iris)** (`homework/tarea4_perceptron/`)
  - Linear separability and perceptron learning.
- **HW_5 — Hill Climbing & Reference-Point Learning** (`homework/tarea5_hill_climbing/`)
  - Heuristic optimization vs. reference-point learning.

### Extras
- **Liquid Neural Networks** (`extras/liquid_neural_networks/`)
- **Deep Linear Networks** (`extras/deep_linear_networks/`)
- **Kolmogorov–Arnold Networks (KAN)** (`extras/kan/`)
- **Liquid Time-Constant Networks (LTC)** (`extras/ltc/`)

---

## Setup

### 1) Create a virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# or
.venv\Scripts\activate      # Windows
```

### 2) Install dependencies
Core:
```bash
pip install -r requirements.txt
```
Optional (deep learning / neuro extras):
```bash
pip install -r requirements-dl.txt
```

---

## How to run

### Python scripts
```bash
python homework/tarea2_frog_visual/Tarea2_IBS.py
python homework/tarea3_hopfield/Tarea3_IBS.py
```

### Notebooks
```bash
jupyter notebook
```
Then open any notebook under `homework/` or `extras/`.

---

## Notes

- Datasets (if any) are not included unless explicitly stated inside each task folder.
- Some notebooks may require heavier dependencies (e.g., TensorFlow/PyTorch). Use `requirements-dl.txt` if needed.

---

## Academic Integrity
This repo is for learning and portfolio purposes. If you’re taking a similar class: **do not copy**—use it to understand concepts and workflow.

---

## Author
Juan Daniel Rosales (with collaborators listed inside each assignment).
