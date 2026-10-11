# Evidence and reporting boundaries

## What was found in the private project folder

The inspected folder contains code for binary classification, lesion detection, manifest preparation and crop preparation. It also contains three private ResNet50 checkpoint files and one training-curve image.

It does **not** contain a formal PDF/DOCX report, a completed detector-evaluation report, a result registry for eight trained models, or saved VGG16/DenseNet121 checkpoints. The public showcase therefore explains implemented architecture and available artefacts without inventing an experiment report.

## What the artefacts demonstrate

| Evidence | Demonstrates | Does not demonstrate |
| --- | --- | --- |
| ResNet50 checkpoint files | Checkpoint persistence for a private ResNet50 run | Clinical accuracy or generalisation |
| Training-curve image | Loss/accuracy were tracked during one exploratory run | End-to-end validation or clinical performance |
| `train_cv.py` | Three classifier candidate backbones and a validation loop | That all candidates completed comparable experiments |
| `train_detector.py` | Faster R-CNN implementation with coordinate-aware flip augmentation | Held-out detection performance |
| Crop helper | 35% contextual crop expansion is implemented | A completed detector-to-classifier hand-off |

## Why this still creates portfolio value

The project demonstrates transferable ML engineering skills:

- converting metadata, images and masks into model-ready records
- handling class imbalance through weighted loss
- using pretrained computer-vision architectures
- writing target-aware geometric augmentation
- separating classification and detection objectives
- checkpointing, resuming runs and recording evaluation metrics
- identifying reproducibility gaps instead of overstating results

## Reporting plan for a future approved release

When the private run inventory is available, the report should include a table with model name, input type, split version, seed, transform recipe, optimizer, checkpoint selection rule and held-out metrics. Detection should be evaluated separately from classification with a defined IoU threshold and lesion-level metrics. Only approved aggregate metrics should be released.
