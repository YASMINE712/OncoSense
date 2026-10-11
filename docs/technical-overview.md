# Technical overview

## Data engineering layer

The project used tabular case metadata and image paths to prepare model inputs. Each record carried the information needed to locate an image, associate its pathology-derived label, and assign it to a training or evaluation split.

The preparation work centered on making image files usable in ML experiments:

- Resolving image locations from tabular manifests
- Converting inputs into tensors through a repeatable preprocessing path
- Resizing images to a model-compatible size
- Applying ImageNet normalization for pretrained backbones
- Mapping pathology categories to a binary experimental label
- Using class weights to reduce the influence of label imbalance

This is the bridge between raw imaging assets and a reproducible training dataset. In a production or publication release, this layer should emit versioned manifests and validation reports before any model training begins.

## Deep-learning layer

The classifier design used transfer learning. A pretrained convolutional backbone supplies general image features, while its final layer is replaced with a two-output classification head for the experiment.

| Candidate backbone | Role in the exploration |
| --- | --- |
| ResNet50 | Primary transfer-learning classifier |
| VGG16 | Alternative convolutional baseline |
| DenseNet121 | Alternative dense-feature baseline |
| Faster R-CNN with ResNet50-FPN | Exploratory lesion-region detection path |

The training workflow included weighted cross-entropy, Adam optimization, checkpoint saving, early stopping, CUDA detection, and an optional automatic mixed-precision path. Run-level metrics included loss, accuracy, F1, precision, and recall.

## Evaluation discipline

This showcase avoids a clinical-performance claim. A training curve only demonstrates how one optimization run progressed. It does not establish diagnostic effectiveness, safety, generalization, or real-world clinical utility.

A mature evaluation stage would report a locked dataset split, model-selection rules, threshold selection, calibration, sensitivity and specificity, PR/ROC curves, confusion matrices, and confidence intervals. Those materials are not distributed here because the underlying internship assets are private.

## Engineering perspective

The project demonstrates an end-to-end pattern that is useful beyond medical imaging:

1. Gather metadata and files from source systems.
2. Convert them into model-ready, traceable records.
3. Apply consistent preprocessing in the data loader.
4. Train and checkpoint transfer-learning models.
5. Store metrics and visual artifacts for comparison.
6. Review outcomes, then improve the manifest, data checks, or modeling approach.

That combination of tabular data handling, data-quality thinking, computer vision, and ML experimentation is the central value of this portfolio piece.
