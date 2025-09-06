# Disease Diagnosis Project

*Built by **Team BioHackers** for ALTSchool Africa Hackathon 2025*

---

## Overview

A machine learning project that predicts diseases from symptoms and patient data.
This project demonstrates how data science can support early diagnosis and healthcare
decision-making in low-resource settings.

---

## Problem Statement

In low-resource healthcare settings, limited infrastructure, understaffing, and
inequitable access to care remain persistent challenges. Nigeria, Africa’s most
populous country, exemplifies this struggle: despite a high disease burden, its
health system is underfunded and overstretched, leading to gaps in diagnosis and
treatment ([WHO, 2023](https://www.afro.who.int/sites/default/files/2023-08/Nigeria.pdf)).

Nigeria faces a dual burden of communicable and non-communicable diseases. Malaria,
tuberculosis, HIV/AIDS, and cholera remain widespread, while hypertension, diabetes,
and cardiovascular disease are sharply rising
([WHO, 2023](https://www.afro.who.int/sites/default/files/2023-08/Nigeria.pdf);
[AHA Journals, 2023](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.123.063671)).
Malaria alone contributes substantially to mortality and morbidity ([Reuters, 2024](https://www.reuters.com/business/healthcare-pharmaceuticals/nigeria-rolls-out-new-oxford-r21-malaria-vaccine-2024-10-17/)).

At the same time, access to diagnostic tools and timely medical advice is limited
outside major cities, worsening preventable illness and death ([PMC, 2013](https://pmc.ncbi.nlm.nih.gov/articles/PMC3560225/); [2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7436948/)).

Machine learning (ML) offers a way forward. Research has shown its potential in:

- Predicting under-five mortality in Nigeria with up to 89.47% accuracy
([BMC Med Inform Decis Mak, 2024](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-024-02476-5))
- Supporting HIV treatment adherence in Nigerian patients ([PMC, 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC12021289/))
- Enhancing outbreak prediction for diseases like cholera
([Exploration Journals, 2023](https://www.explorationpub.com/Journals/edht/Article/101140))

Full Reference List: `citations.md`

### Scope & Goal

This project addresses the urgent need for accessible, privacy-safe, and
context-aware disease prediction tools. Using **domain-informed synthetic Nigerian
hospital data**, we trained a machine learning predictor tailored to local healthcare
realities.

⚠️ While not a substitute for clinical expertise, this system demonstrates how
AI-driven models can serve as decision support tools for healthcare workers in
low-resource settings—helping flag likely diseases earlier, guide triage, and
ultimately improve patient outcomes.

---

---

## Features

- Confidence intervals: Bootstrap resampling estimates uncertainty for model
predictions.  
- Interactive demo: Streamlit for real-time predictions.

---

## Dataset

- **Synthetic dataset:** Generated using `src/generator.py`.
- **Generation process:** `data/README.md`
- **File:** `data/synthetic_dataset.csv`
- **10,000 patient records with features:**
  `age_band, gender, setting, region, season, fever, headache, cough, fatigue, chills,
  sweats, nausea, vomiting, diarrhea, abdominal_pain, loss_of_appetite, sore_throat,
  runny_nose, dysuria`. confirm symptoms all represented
- **Classes (10 diseases + healthy):**  
  healthy, malaria, typhoid, gastroenteritis, pneumonia, diabetes, measles,
  tuberculosis, HIV, hypertension, peptic ulcer.
- **Epidemiological basis:** `data/domain_review.md`

---

## Workflow

1. **Preprocessing**
   - Categorical encoding (`OneHotEncoder`) for non-numeric columns.
   - Feature selection: symptoms + basic patient data.

2. **Modeling**
   - Random Forest & XGBoost classifiers.
   - Balancing via sample weights to handle class imbalance.

3. **Evaluation**
   - Metrics: Accuracy, classification report.
   - Visualization: Confusion matrices, calibration plots.
   - Real-world add-on: Bootstrap confidence intervals to estimate uncertainty.

---

## Notebooks

- `notebooks/disease_diagnosis_demo.ipynb`: Full demo with preprocessing, model
training, evaluation, and confidence intervals.

---

## Scripts

- `src/generator.py`: Synthetic dataset generator module.

---

## Results

- Confusion matrices, calibration plots, and evaluation plots are saved in `/plots`.
- Trained model pipelines are saved in `/models` as `.pkl` files.
- Class balance and sample weight summaries are printed during training.

---

## Getting Started

```bash
### Clone the repository
git clone <repo-url>
cd disease-diagnosis

### Install dependencies
pip install -r requirements.txt

### Run notebook
jupyter notebook notebooks/disease_diagnosis_demo.ipynb

---

## Tools

- Libraries: `scikit-learn`, `xgboost`, `pandas`, `numpy`, `matplotlib`, `seaborn`.

- Deployment: TBD

## Team

- Dorcas Jola-Moses
- Ayoku Alaran

## 📂 Repo Structure

disease-diagnosis/
├── data/
│   ├── synthetic_dataset.csv
│   ├── README.md
│   └── domain_review.md
├── models/
│   └── *.pkl
├── notebooks/
│   └── disease_diagnosis_demo.ipynb
├── plots/
│   ├── confusion_matrix_*.png
│   ├── calibration_plot_*.png
├── src/
│   └── generator.py
├── requirements.txt
├── README.md
├── citations.md              <-- full references & citations
└── .gitignore
