# PSC Multimodal Machine Learning Models

This repository contains code for training and evaluating machine-learning models
to predict primary sclerosing cholangitis (PSC) using multiple data modalities,
including genetics, laboratory measurements, serology, and clinical variables.

The framework supports:
- Single-modality Random Forest models
- Early integration (feature-level fusion)
- Late integration (model-level fusion via stacking)

All models are evaluated using nested cross-validation with consistent data splits
to enable fair comparison across modeling strategies.

---

## Data availability

Patient-level data are not included in this repository due to privacy and
institutional restrictions. The code is provided to enable methodological
reproducibility.

To run the scripts, users must supply their own input data with equivalent structure.

---

## Repository structure
PSC_GitHub/
├── modeling/
│ ├── single_modal.py # Single-modality RF models
│ └── multi_modal.py # Multi-modal (early + late integration)
│
├── evaluation/
│ └── predict_with_saved_models.py
│
├── hpc/
│ ├── run_multi_modal.slurm
│ └── run_prediction.slurm
│
├── requirements.txt
├── README.md
└── LICENSE
---

## Example usage

### Run multimodal evaluation
```bash
python modeling/multi_modal_integration.py \
  --input-dir /path/to/imputed_data \
  --pheno-path /path/to/phenotype.csv \
  --out-dir /path/to/output \
  --n-jobs 16


#Run a single-modality model
python modeling/single_modal_rf_nestedcv.py \
  --modality serology

Methods overview

Random Forest classifiers are used as base learners.

Hyperparameters are optimized via inner cross-validation.

Early integration concatenates features across modalities.

Late integration combines modality-specific models using a logistic regression
meta-learner.

Performance metrics include AUC, precision, recall, and balanced accuracy,
computed on out-of-fold predictions.


Requirements

Install dependencies using:
pip install -r requirements.txt

License

This project is released under the MIT License.
