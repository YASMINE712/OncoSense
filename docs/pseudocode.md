# Safe pseudocode: two computer-vision paths

This describes the implementation design without publishing private source code, images, masks, row-level data, weights or filesystem paths.

## Path A: manifest-driven binary classification

```text
INPUT: train manifest and held-out manifest

read CSV records containing image reference and binary pathology label
for each record:
    open referenced PNG image
    convert image to RGB
    resize to 224 x 224
    convert image to tensor
    apply ImageNet normalization
    return tensor and label

select a pretrained CNN candidate:
    ResNet50, VGG16 or DenseNet121
replace final layer with a two-output classifier
compute class weights from the training labels

for each epoch:
    train with weighted cross-entropy and Adam
    use mixed precision when CUDA is available
    evaluate against held-out manifest
    record loss, accuracy, F1, precision and recall
    save latest checkpoint
    save best checkpoint when validation F1 improves
    stop after 10 validation epochs without improvement
```

## Path B: lesion localisation and crop preparation

```text
INPUT: full mammogram reference, lesion-mask reference, experimental label

locate full mammogram using patient ID, side and view
read the lesion mask
find non-zero mask pixels
derive xmin, ymin, xmax and ymax from mask extents

for each detector training image:
    with 0.5 probability, flip image horizontally and mirror x box coordinates
    with 0.5 probability, flip image vertically and mirror y box coordinates
    train Faster R-CNN with a ResNet50-FPN backbone
    retain checkpoint with lowest mean training loss

for crop preparation when explicit box columns are available:
    expand the lesion box by 35 percent for contextual tissue
    clip crop bounds to image dimensions
    save crop reference and binary label for a downstream classifier
```

## Technology mapping

| Stage | Applied implementation |
| --- | --- |
| Manifest preparation | Python, Pandas, CSV records, path resolution |
| Image and mask handling | Pillow, OpenCV helper, NumPy |
| Classification batching | PyTorch `Dataset` and `DataLoader` |
| Classification models | Torchvision ResNet50, VGG16 and DenseNet121 |
| Detection model | Torchvision Faster R-CNN with ResNet50-FPN |
| Detection augmentation | Coordinate-aware horizontal and vertical flips |
| Optimisation | Adam for classification; SGD with momentum, warm-up and cosine scheduling for detection |
| Tracking | Checkpoints, learning curve and classification metrics |

The pseudocode is a portfolio explanation, not an assertion of clinical validation or an end-to-end production deployment.
