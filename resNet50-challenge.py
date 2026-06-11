# -*- coding: utf-8 -*-
#   Project: misc-cise File: resNet50-challenge.py.py  Created: 6/8/26 22:04 Author: etyrer & his robot dog™
import numpy as np
import os
import platform
import subprocess
import torch
import torchvision.models as models

# Core ML conversion is optional and only relevant if you want to export the model for use on macOS/iOS.
# It is not required for loading the model or making predictions in Python.

try:
    import coremltools as ct
except ImportError:
    ct = None

from torchvision.models import ResNet50_Weights
from torchvision import transforms
from PIL import Image  # we'll use Pillow to handle image loading and processing

# use tkinter to open a file dialog to select an image file
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# method to use file dialog to select an image file
def select_image_file():
    """
    Select an image file.

    On macOS, prefer the native Apple file chooser because Tk dialogs can behave oddly
    when launched from PyCharm or uv. Keep Tk as a fallback for other platforms.
    """
    if platform.system() == "Darwin":
        script = '''
        set chosenFile to choose file with prompt "Select an image file"
        POSIX path of chosenFile
        '''
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                check=False,
            )
            file_path = result.stdout.strip()
            return file_path or None
        except Exception as exc:
            print(f"macOS file picker failed, falling back to Tk: {exc}")

    root = Tk()
    root.withdraw()  # Hide the root window.

    # macOS/Tk is happier when each extension is provided as a separate pattern.
    file_path = askopenfilename(
        parent=root,
        title="Select an image file",
        filetypes=[
            ("Image files", ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif")),
            ("All files", "*.*"),
        ],
    )

    root.destroy()
    return file_path or None

def load_resnet50(pretrained=True):
    """
    Load a ResNet-50 model.

    If pretrained=True, use ImageNet pre-trained weights.
    Put the model in evaluation mode before returning it.
    """
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    model.eval()
    return model

def get_transform():
    """
    Get the image transformation pipeline for ResNet-50.

    This includes resizing, center cropping, converting to tensor, and normalizing
    using ImageNet mean and std.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def generate_random_image(width, height):
    array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(array)

def load_image(image_path=None):
    """
    If image path is none, generate a random image. Otherwise, load the image from the specified path and apply the
    necessary transformations to prepare it for input into the ResNet-50 model.

    """
    if image_path is None:
        image = generate_random_image(224, 224)
    else:
        image = Image.open(image_path).convert("RGB")

    transform = get_transform()
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

def predict(model, image_tensor, topk=5):
    """
    Predict the top-k classes for an already-transformed image tensor.
    """
    with torch.no_grad():
        logits = model(image_tensor)

    probabilities = torch.softmax(logits, dim=1)
    top_probabilities, top_classes = torch.topk(probabilities, k=topk, dim=1)

    results = []
    for prob, cls in zip(top_probabilities[0], top_classes[0]):
        results.append((cls.item(), prob.item()))

    return results


def convert_resnet50_to_coreml(model, output_path="resnet50.mlpackage"):
    """
    Convert the PyTorch ResNet-50 model to a Core ML model package.

    Core ML conversion happens after the PyTorch model is loaded.
    It does not belong in load_image() or get_transform().
    """
    if ct is None:
        raise ImportError("coremltools is not installed. Run: uv add coremltools")

    example_input = torch.rand(1, 3, 224, 224)

    traced_model = torch.jit.trace(model, example_input)

    coreml_model = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input", shape=example_input.shape)],
        convert_to="mlprogram",
    )

    coreml_model.save(output_path)
    return output_path

def load_imagenet_labels(labels_path=None):
    """Load ImageNet class labels from a file."""
    if labels_path is not None:
        with open(labels_path, "r") as f:
            labels = [line.strip() for line in f.readlines()]
    else:
        labels = ResNet50_Weights.IMAGENET1K_V2.meta["categories"]

    return {
        idx: label for idx, label in enumerate(labels)
    }

if __name__ == "__main__":
    """
    main function to load the ResNet-50 model, load an image (either from a file or generate a random one), and predict 
    the top-5 classes for that image using the model. The predicted classes and their probabilities will be printed to
    the console.
    """

    print("Loading ResNet-50 model... ImageNet pre-trained weights will be used.")
    model = load_resnet50(pretrained=True)

    # ------------------------------------------------------------------
    # Learning Notes / Previous Attempts
    # ------------------------------------------------------------------
    # Optional Core ML export path:
    # Uncomment this if you want to create a Core ML model package for macOS/iOS use.
    # coreml_path = convert_resnet50_to_coreml(model)
    # print(f"Saved Core ML model to: {coreml_path}")
    # transform = get_transform


    # Learning notes / previous attempts intentionally preserved for reference.
    # Keeping these while working through the exercise is perfectly fine.
    labels = load_imagenet_labels()
    print("Opening image picker...")
    image_path = select_image_file()  # Open file dialog to select an image, or set to None to generate a random image
    print(f"selected file: {image_path}")

    if image_path:
        image_tensor = load_image(image_path)
        print(image_tensor.shape)
    else:
        image_tensor = load_image()  # Generate a random image if no path is provided

    print(f"Loading image... If you want to test with a random image, set the image_path variable to None.")
    # image_tensor = load_image("goldfish.jpg")

    print("Predicting based on inference from the model...")
    predictions = predict(model, image_tensor)

    print("Top-5 Predictions:")
    for cls, prob in predictions:
        print(f"{labels[cls]}: {prob:.4f}")

    print(image_tensor.shape)


