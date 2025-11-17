# 🪶 Perch-Scale Manuscript Repository

This repository contains the **data, analysis scripts, and figure-generation notebooks** accompanying the manuscript describing the **NeuralSyntaxLab Perch-Scale** — an open-source system for continuous and automated weight monitoring of birds during neurophysiology experiments.

It provides all materials required to reproduce the results and figures presented in the study, and serves as an accessible example of how to organize, process, and visualize weight data collected from multiple perch-scale sensors.

<p align="left">
    <img src="https://github.com/user-attachments/assets/27e0cfaa-9621-4f2f-849d-2f27458e2df8" alt="Description" style="width:30%;">
</p>

---

## 📖 Overview

The repository includes:

- **Analysis scripts and notebooks** (`scripts/`):  
  Self-contained Python notebooks that reproduce all manuscript figures and demonstrate the data-processing workflow.  
  The code is modular and relies on helper functions defined in `_paths.py` and `helpers.py`.

- **Data** (`data/`):  
  - `birds/`: 8 compressed `.csv.gz` files with continuous weight recordings from individual birds.  
  - `controls/`: 5 one-day control recordings of idle objects used for validation.  
  - `metadata/`: further experimental information used in the analysis (daily manual weights, bird names, summary tables, etc...).  
  All files are directly usable; scripts automatically read them using `pandas` with gzip compression.

- **Figures** (`figures/`):  
  Final versions of the manuscript figures as .png's.

- **Environment file** (`requirements.txt`):  
  Python dependencies for reproducing the analysis in a clean local environment.

---

## ⚙️ Local environment setup

It is recommended to use a **dedicated virtual environment** to run the analysis:

```bash
# create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# install required packages
pip install -r requirements.txt
```
---

## 🔗 Companion repository

This manuscript repository focuses exclusively on the **data analysis and figure generation** accompanying the paper.

The **Perch-Scale System Repository** contains all resources required to build and operate the perch-scale hardware, including:
- Mechanical design files (CAD, drawings)
- Arduino firmware for scale sensors
- Raspberry Pi host software
- User setup and calibration guides

👉 **Access it here:** [NeuralSyntaxLab/perch-scale-system](https://github.com/NeuralSyntaxLab/Bird-Scale-Methods-Article.git)  
*(If not yet public, replace with the official link once available.)*

---

