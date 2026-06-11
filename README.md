A simple Python-based image classification application built as part of a machine learning and computer vision learning exercise.

The project uses a pre-trained ResNet-50 model from PyTorch/TorchVision to classify images against the ImageNet dataset and display the top-5 predicted classes.

Features

* Load a pre-trained ResNet-50 model
* Use ImageNet-trained weights
* Select images using a native macOS file picker
* Generate random placeholder images for testing
* Apply standard ImageNet preprocessing and normalization
* Run inference using PyTorch
* Display top-5 ImageNet predictions with probabilities
* Optional Core ML model export for macOS/iOS experimentation

Image
  ↓
Load Image
  ↓
Apply Transform
  ↓
Tensor [1, 3, 224, 224]
  ↓
ResNet-50
  ↓
Softmax
  ↓
Top-K Selection
  ↓
Human-Readable Labels

Requirements

* Python 3.12+
* PyTorch
* TorchVision
* Pillow
* NumPy

Optional:

* coremltools (for Core ML export)

Create a virtual environment
uv venv
source .venv/bin/activate