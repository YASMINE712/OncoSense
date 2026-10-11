# Architecture and implementation report

## Problem framing

The private project explores binary image-based experiments on mammography records. Its experimental labels are benign-like (`0`) and malignant (`1`). This is not equivalent to a clinical diagnosis product: the available project artefacts do not establish patient-level split control, external validation, threshold selection, calibration or clinical review.

## Architecture

```text
Restricted PNG images + CSV metadata + lesion masks
                         |
                         v
             Manifest and path-resolution layer
                    |                    |
                    |                    |
                    v                    v
       Direct classifier path       Detection-first path
       ----------------------       --------------------
       RGB -> 224 resize ->         Full mammogram + mask
       ImageNet normalization       -> mask-derived box
       -> CNN classifier            -> Faster R-CNN / ResNet50-FPN
       -> benign/malignant label    -> optional 35% context crop
                    |                    |
                    +--------- experiment artefacts --------+
                                      |
                                      v
                    checkpoints, loss, accuracy, F1,
                    precision and recall
```

## Data engineering and traceability

The classifier reads `roi_train.csv` and `roi_test.csv`. The records contain a patient identifier, laterality, image view, abnormality number, pathology text, type, a binary label and a PNG path. The training code reads this schema with Pandas, opens the referenced PNG through Pillow, and emits `(tensor, label)` pairs through a custom PyTorch `Dataset`.

For the detection route, the manifest builder uses patient ID, side and view to find the matching full mammogram. It retains the full-image path, lesion-mask path and binary label. The detector dataset reads the mask, finds all non-zero mask pixels, and converts the extrema to `[xmin, ymin, xmax, ymax]` coordinates.

This is the valuable data-engineering contribution: the model does not receive an unexplained image directory. It receives image assets connected to metadata, image role and labels through tabular manifests.

## Classifier path

The code exposes ResNet50, VGG16 and DenseNet121. It initializes each from Torchvision pretrained weights, replaces the final classifier with two outputs and uses a common input recipe:

- RGB conversion
- resize to `224 x 224`
- tensor conversion
- ImageNet mean/std normalization

The training loop calculates class weights from the training labels and passes them to cross-entropy loss. It trains with Adam, can resume from a checkpoint, optionally uses automatic mixed precision on GPU, and retains the best checkpoint according to validation F1. Validation reports loss, accuracy, F1, precision and recall.

There is no image-flip augmentation in the classification transform currently visible in the code. Adding it to a later release would require clinical and data-governance review, because image orientation can be meaningful in mammography.

## Detector and crop path

The detector is a Faster R-CNN ResNet50-FPN model with a two-class predictor: background and lesion. Its spatial augmentation is explicit and target-aware:

- `0.5` probability horizontal flip, with x box coordinates mirrored
- `0.5` probability vertical flip, with y box coordinates mirrored

The crop helper separately expands a supplied lesion box by `35%` on width and height, clipped to image borders. It saves the lesion-centred crop and its experimental label.

The two scripts are not yet connected by one shared manifest contract: the crop helper consumes columns named `image_path`, `xmin`, `ymin`, `xmax`, `ymax` and `bin_label`, while the current builder stores `full_img_path`, `mask_path` and `label`. The architecture shows the intended two-stage approach; this interface difference is a concrete engineering task, not something hidden by the portfolio.

## Evidence currently available

| Item | Evidence | What can be claimed |
| --- | --- | --- |
| ResNet50 run | `resnet50_best.pt`, `resnet50_fold1.pt`, `resnet50_latest.pt` | A ResNet50 training run produced saved private checkpoints |
| Learning curve | `metrics_curve_fold1.png` | An exploratory run tracked loss/accuracy over epochs |
| Candidate classifiers | `train_cv.py` | ResNet50, VGG16 and DenseNet121 are implemented candidates |
| Detection implementation | `train_detector.py` | Faster R-CNN training and flip augmentation are implemented |
| Detector performance | No detector test/evaluation artefact found | No detector quality claim should be made |
| Eight trained models | No corresponding inventory/report found in this folder | Do not claim eight until the run registry is recovered |

## What a publishable technical report needs next

1. Validate every manifest record and produce a row-count, missing-file and duplicate report.
2. Split at patient level before any preprocessing or augmentation.
3. Define one versioned contract for masks, boxes, crops and labels.
4. Evaluate detector localisation on a held-out set using a defined IoU protocol.
5. Record architecture, seed, transform recipe, split version, optimizer and metrics in a model registry.
6. Publish only approved aggregate metrics and synthetic or authorised visuals.
