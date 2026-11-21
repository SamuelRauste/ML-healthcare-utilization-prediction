# Predicting Healthcare Utilization with Machine Learning

**A small supervised-learning study comparing logistic regression and decision trees on survey data from older adults.**

This project began as a 2024 university machine learning assignment. It investigates whether a respondent's self-reported health, age, employment status, and sleep-related factors can predict how many different doctors they visited in a year.

The key takeaway is **not** a high-performing predictor: the original logistic regression achieved **50.48% test accuracy**, highlighting the difficulty of making useful predictions from a small survey dataset. The project demonstrates a complete classification workflow and honest evaluation of its limitations.

## Problem and data

The [National Poll on Healthy Aging (NPHA) dataset](https://archive.ics.uci.edu/dataset/936/national+poll+on+healthy+aging+npha) contains **714 survey records**, 14 original input features, and a three-class target:

| Target label | Number of different doctors visited |
| --- | --- |
| 1 | 0–1 |
| 2 | 2–3 |
| 3 | 4 or more |

In the original analysis, removing 16 records with refused responses left **698 observations**. Two features (`Trouble_Sleeping` and `Race`) were excluded, leaving **12 input features**. The train/validation/test split was approximately **70% / 15% / 15%**.

## Methods

- Clean refused/missing survey responses and select input features.
- Train **multinomial logistic regression** and a **decision tree** using scikit-learn.
- Evaluate training and validation accuracy to detect overfitting.
- Assess the logistic regression on the held-out test set.
- Compare against a **majority-class baseline** and produce a confusion matrix using the included analysis script.

The repository's Python script is a cleaned-up implementation of the original notebook appendix. It retains the original split logic and feature choices; it also adds a deterministic tree seed, baseline comparison, and optional saved plots. Consequently, rerun metrics can differ slightly from those in the original report.

## Original results (2024 report)

| Model | Train accuracy | Validation accuracy | Test accuracy |
| --- | ---: | ---: | ---: |
| Logistic regression | 52.05% | 54.29% | **50.48%** |
| Decision tree | 89.96% | 28.57% | Not reported |

*Values above are transcribed from the report's **Results** section, not newly verified runs. The report's conclusion gives a different decision-tree validation figure; rerun the script for an independently calculated value.*

The decision tree substantially overfit the training data. Logistic regression generalized better but produced weak held-out performance. A majority-class strategy represents an essential reference point: the largest class accounts for roughly half of the full dataset, so overall accuracy alone is insufficient evidence of predictive value.

## Getting started

**Requirements:** Python 3.11–3.13 recommended, with `pip`.

1. Clone this repository and create a virtual environment:

   ```bash
   git clone https://github.com/YOUR-USERNAME/healthcare-utilization-prediction.git
   cd healthcare-utilization-prediction
   python -m venv .venv
   ```

2. Activate it:

   - **Git Bash on Windows:** `source .venv/Scripts/activate`
   - **PowerShell on Windows:** `.venv\Scripts\Activate.ps1`
   - **Linux/macOS:** `source .venv/bin/activate`

3. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Download `NPHA-doctor-visits.csv` from the [UCI dataset page](https://archive.ics.uci.edu/dataset/936/national+poll+on+healthy+aging+npha) and save it as `data/NPHA-doctor-visits.csv`.

5. Run the analysis:

   ```bash
   python train.py --data data/NPHA-doctor-visits.csv
   ```

The script prints model accuracy, majority-class baseline accuracy, and a classification report. It saves calculated metrics and a confusion matrix into `figures/` (generated locally and not committed by default).

To run preprocessing tests:

```bash
python -m unittest discover -s tests -v
```

## Project structure

```text
healthcare-utilization-prediction/
├── train.py                 # Data cleaning, models, evaluation and plots
├── requirements.txt         # Python dependencies
├── data/
│   └── README.md            # Data download instructions
├── docs/
│   └── original_report.pdf  # Original 2024 course report and code appendix
├── tests/
│   └── test_preprocessing.py
└── figures/                 # Generated plots and metrics (on local runs)
```

## Limitations and possible improvements

- **Small sample:** 698 observations after filtering can lead to unstable performance estimates.
- **Class imbalance:** accuracy can be misleading without a majority-class baseline and per-class metrics.
- **Feature representation:** some coded categories, such as employment status, are not genuinely ordinal and could benefit from one-hot encoding.
- **Validation strategy:** stratified splits and cross-validation would make performance estimates more robust.
- **Prediction, not causation:** self-reported survey variables should not be interpreted as causing healthcare visits.
- **Not for clinical use:** this is an educational experiment, not a tool for medical decisions or resource allocation.

An interesting next step would be to compare encoded features, tuned regularization, and repeated stratified cross-validation against a dummy classifier, without using the held-out test set for model selection.

## Data attribution

National Poll on Healthy Aging (NPHA), UCI Machine Learning Repository, dataset ID 936. [Dataset and citation information](https://archive.ics.uci.edu/dataset/936/national+poll+on+healthy+aging+npha); [original study DOI](https://doi.org/10.3886/ICPSR37305.v1).

The UCI dataset is distributed under [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/). The dataset is **not included** in this repository; download it from the source above. Any use or redistribution must preserve attribution.
