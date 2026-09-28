# Breast Cancer Classification: Logistic Regression Cross-Validation

A comparison of two Logistic Regression pipelines on the Breast Cancer Wisconsin dataset, evaluated with 5-fold Stratified Cross-Validation. The goal is to understand how regularization strength (`C`) affects model performance and to justify a deployment choice using metrics.

## Objective

- Build a Logistic Regression pipeline with feature scaling
- Evaluate it with 5-fold Stratified Cross-Validation
- Train a second, heavily regularized model (`C=0.001`) and compare both
- Decide which model to deploy, backed by metric numbers

## Dataset

- `sklearn.datasets.load_breast_cancer` (569 samples, 30 numeric features)
- Target: `0` = malignant, `1` = benign

## Methodology

1. Loaded the data and did basic inspection (nulls, duplicates, shape)
2. Built a `Pipeline`: `StandardScaler` + `LogisticRegression`
3. Used `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` so each fold keeps the same class ratio
4. Ran `cross_validate` with 5 metrics: accuracy, precision, recall, F1, ROC-AUC
5. Repeated the same process for `LogisticRegression(C=0.001)`

Scaling is inside the Pipeline, so it is fitted only on the training part of each fold. This avoids data leakage.

## Results (mean ± std over 5 folds)

| Metric | Default (C=1.0) | Strong Reg (C=0.001) |
|---|---|---|
| Accuracy | 0.9737 ± 0.0166 | 0.8946 ± 0.0318 |
| Precision | 0.9683 ± 0.0290 | 0.8605 ± 0.0324 |
| Recall | 0.9916 ± 0.0112 | 0.9944 ± 0.0113 |
| F1 | 0.9794 ± 0.0127 | 0.9224 ± 0.0225 |
| ROC-AUC | 0.9953 ± 0.0053 | 0.9877 ± 0.0060 |

## Conclusion

The **default model (C=1.0)** is the better choice to deploy.

- It is better on accuracy, precision, F1 and ROC-AUC.
- Recall is almost the same for both models (difference is within one standard deviation).
- `C=0.001` over-regularizes the model (underfitting), which lowers precision by about 11 points and produces more false alarms.
- The default model is also more stable across folds (lower std).

## Key Learnings

- Cross-validation gives a more reliable estimate than a single train/test split
- Putting preprocessing inside a `Pipeline` prevents data leakage
- Stronger regularization is not always better; it can cause underfitting

## Tech Stack

Python, pandas, scikit-learn, matplotlib, seaborn

## How to Run

```bash
git clone https://github.com/shabanamalik/breast-cancer-logreg-cross-validation.git
cd breast-cancer-logreg-cross-validation
pip install -r requirements.txt
jupyter notebook breast_cancer_cv_comparison.ipynb
```

## Author

Shabana Malik | [GitHub](https://github.com/shabanamalik)
