### Anonymization guidelines
- Remove **EXIF** metadata
- Ensure **no faces** and **no sensitive/private locations** are visible
- Crop tightly to the **keris object** (minimize background context)
- Prefer **white background** (post-rembg) when possible

## Reproducing results without public data

Even if `data/sample_anonymized/` is not included, the training/evaluation scripts in this repository are designed to output artifacts needed for manuscript tables and auditing, such as:
- Metrics CSV files (for manuscript tables)
- Confusion matrix (CSV and/or image)
- Per-image prediction CSV (for audit and error analysis)

## Data access

If you need access to the full dataset, contact the project owner/maintainer. Access is granted only under appropriate authorization and data-use agreements.