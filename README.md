# Brain MRI Tumor Classification with CNN + Grad-CAM

A convolutional neural network (CNN) built from scratch in PyTorch to classify
brain MRI scans into four tumor categories, with Grad-CAM explainability
overlays showing which brain regions the model attends to when making each
prediction.


## Dataset

**Source:** Brain Tumor MRI Dataset (Masoud Nickparvar) via Kaggle
([link](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset))

- 7,200 MRI images (5,600 training / 1,600 testing), pre-split by dataset creator
- 4 balanced classes (1,400 training images each):

| Class | Description |
|---|---|
| Glioma | Malignant tumor arising from glial cells |
| Meningioma | Tumor arising from the meninges (brain lining) |
| Pituitary | Tumor in the pituitary gland region |
| No tumor | Healthy brain scan |


## Model architecture

Three-layer CNN built from scratch — no pretrained weights used:

```python
class BrainMRI_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 4)
```

- Input: grayscale MRI images resized to 128×128
- Three Conv2d layers with ReLU activation and MaxPool2d (halves dimensions each time)
- Final feature map: 16×16×64 → flattened → two fully connected layers
- Data augmentation: random rotation (±10°) and horizontal flipping on training images only


## Methodology

### Iterative improvement process

| Attempt | Configuration | Test accuracy |
|---|---|---|
| Baseline | No augmentation, 5 epochs | 82.50% |
| + Augmentation | Data augmentation, 5 epochs | 85.00% |
| + More training | Data augmentation, 10 epochs | **90.00%** |

Training for more epochs resolved a tradeoff discovered during augmentation
(improved 3 of 4 classes, but initially hurt glioma recall, which recovered
with continued training).


## Results

**Final test accuracy: 90.00%** (n = 1,600 test images, GPU: T4)

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Glioma | 0.90 | 0.79 | 0.84 |
| Meningioma | 0.90 | 0.86 | 0.88 |
| No tumor | 0.87 | 0.98 | 0.92 |
| Pituitary | 0.95 | 0.98 | 0.97 |

### Key finding — glioma/meningioma confusion

The model's primary source of error is between glioma and meningioma
specifically (33 + 28 mix-ups), consistent with known radiological difficulty
in distinguishing these two tumor types on MRI. This is a clinically
meaningful, not an arbitrary, failure pattern.


## Grad-CAM explainability

Gradient-weighted Class Activation Mapping (Grad-CAM) was implemented from
scratch using PyTorch forward and backward hooks, producing heatmap overlays
showing which brain regions drove each prediction:

```python
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
```

**Key Grad-CAM finding:** On glioma images misclassified as meningioma, the
model's attention was correctly focused on the tumor region itself — not on
irrelevant background. This indicates the confusion arises from genuine visual
similarity between the two tumor types (a known clinical challenge), rather
than the model attending to uninformative image features.

## Tools & libraries

Python · PyTorch · torchvision · scikit-learn · OpenCV · Matplotlib · Kaggle API

## Files

| File | Description |
|---|---|
| `brain_mri_cnn.ipynb` | Full notebook with architecture, training loop, and Grad-CAM |
