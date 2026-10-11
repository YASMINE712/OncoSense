# OncoSense: Breast-Cancer Computer-Vision Showcase

> Public-safe portfolio of an internship exploration using mammography images for binary benign-versus-malignant experiments. The original implementation, images, annotations, model weights, and clinical records remain private.

## Why this project matters

OncoSense shows the engineering work needed before a computer-vision model can be useful: turn image metadata into traceable manifests, align full mammograms with lesion masks, build model-ready loaders, train pretrained vision backbones, track runs, and keep experimental limits explicit.

It explores **two complementary computer-vision paths**:

1. **Binary image classification**: a pretrained CNN receives an image selected by the train/test manifest and predicts the experimental binary pathology label.
2. **Lesion localisation and crop-focused analysis**: a detector learns lesion boxes derived from masks on full mammograms; a crop helper can expand a lesion box with surrounding context before downstream classification.

The project is an exploratory research workflow. It is not a clinical decision system and it makes no diagnostic-performance claim.

![OncoSense technical architecture](assets/ml-workflow.svg)

## What the data pipeline does

The code implements an ETL-style vision pipeline. It is not an Airflow or cloud ETL deployment; it is Python data preparation for ML experiments.

| Stage | Actual work in the project | Output |
| --- | --- | --- |
| Extract | Read CSV manifests carrying patient/view metadata, image paths, pathology and experimental labels | Train/test rows that can be traced back to an approved image asset |
| Transform | Resolve paths, open PNG files, convert to RGB, resize to `224 x 224`, convert to tensors and apply ImageNet normalization | Consistent tensors for pretrained CNNs |
| Transform for detection | Match patient ID, side and view to a full mammogram; read a lesion mask; derive a bounding box from non-zero mask pixels | Detection targets: full image, box and lesion label |
| Load | Use PyTorch `Dataset` and `DataLoader`, batching and shuffled training batches | Batches for classifier or detector training |
| Record | Save checkpoints and an exploratory learning curve | Reproducible run artefacts, without publishing weights |

The private train manifest contains 6,123 rows and the held-out manifest contains 1,487 rows. The binary experiment maps `MALIGNANT` to label `1` and `BENIGN` / `BENIGN_WITHOUT_CALLBACK` to label `0`. The showcase publishes only aggregate counts, never rows or images.

## Computer-vision approach A: binary classification

The classifier factory supports three pretrained CNN backbones:

| Backbone | Role |
| --- | --- |
| ResNet50 | Primary saved-run configuration in the available artefacts |
| VGG16 | Alternative transfer-learning candidate |
| DenseNet121 | Alternative transfer-learning candidate |

The classification path uses:

- Torchvision pretrained weights and a replaced two-class prediction head
- Image resizing, RGB conversion and ImageNet normalization
- Class-weighted cross-entropy to reduce label-imbalance effects
- Adam with learning rate `1e-4`
- CUDA detection and optional automatic mixed precision
- Early stopping after 10 non-improving validation epochs
- Checkpoint resume and best-checkpoint selection by validation F1
- Accuracy, precision, recall, F1 and loss tracking

The available private artefacts show ResNet50 checkpoints and one learning-curve image. The code supports VGG16 and DenseNet121, but this folder does not contain their saved checkpoint evidence. It also does not prove eight separate trained model runs, so the public portfolio does not claim that number.

![Exploratory ResNet50 training curve](assets/training-curve.png)

This curve shows optimization progress in one private run. It is not clinical accuracy, external validation, or proof of diagnostic safety.

## Computer-vision approach B: lesion localisation, augmentation and crops

The detector path uses **Faster R-CNN with a ResNet50-FPN backbone** on full mammograms. Bounding boxes are created from lesion-mask extents at training time. The detector uses:

- Horizontal flip with probability `0.5`, while updating the x-coordinates of the box
- Vertical flip with probability `0.5`, while updating the y-coordinates of the box
- SGD with momentum `0.9` and weight decay `1e-4`
- Two-epoch learning-rate warm-up followed by cosine annealing
- Optional mixed precision on CUDA
- Best-checkpoint saving by mean training loss

The crop utility expands a lesion bounding box by **35% context** before saving a lesion-centred image. This supports the intended detector-to-crop-to-classifier workflow: localise the suspicious region, preserve surrounding tissue context, then pass the crop to a classifier.

The public description labels this route as **exploratory**. The available scripts do not yet provide a complete detector evaluation protocol or a verified end-to-end detector-to-classifier run. In addition, the crop helper expects explicit box columns while the current manifest-builder script writes image and mask references; that interface should be aligned before presenting the two-stage path as reproducible.

## Architecture, reporting and next engineering step

See:

- [Architecture and implementation report](docs/technical-overview.md)
- [Safe pseudocode for both CV paths](docs/pseudocode.md)
- [Evidence and reporting boundaries](docs/technical-report.md)
- [Publication boundaries](docs/publication-boundaries.md)

The next credible improvement is a versioned manifest-validation stage, an aligned detector/crop interface, patient-level splitting, a held-out detector evaluation protocol, and a model registry that records every architecture, seed, configuration and result.

## Technology stack

`Python` · `PyTorch` · `Torchvision` · `Pandas` · `NumPy` · `Pillow` · `OpenCV` · `scikit-learn` · `Matplotlib` · `TensorBoard`

## What remains private

- Internship source code and notebooks
- Mammography images, masks, metadata and row-level labels
- Trained weights, checkpoints and full experiment logs
- Private paths, credentials and environment files
