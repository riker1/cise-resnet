# -*- coding: utf-8 -*-
#   Project: misc-cise File: resNet50-challenge.py.py  Created: 6/8/26 22:04 Author: etyrer & his robot dog™
import numpy as np
import os
import torch
import coremltools as ct
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from torchvision import transforms
from PIL import Image

# libary to open a file dialog to select an image file on macOS

from tkinter import Tk
from tkinter.filedialog import askopenfilename

# python to use macOS file dialog to select an image file
def select_image_file():
    root = Tk()
    root.withdraw()  # Hide the root window
    root.update()

    # macOS/Tk is happier when each extension is provided as a separate pattern.
    file_path = askopenfilename(
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
    Predict the top-k classes for the given already-transformed image tensor.
    """
    with torch.no_grad():
        logits = model(image_tensor)

    probabilities = torch.softmax(logits, dim=1)
    top_probabilities, top_classes = torch.topk(probabilities, k=topk, dim=1)

    results = []
    for prob, cls in zip(top_probabilities[0], top_classes[0]):
        results.append((cls.item(), prob.item()))

    return results

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
    transform = get_transform()
    labels = load_imagenet_labels()
    image_path = select_image_file()  # Open file dialog to select an image, or set to None to generate a random image

    if image_path:
        image_tensor = load_image(image_path)
    else:
        image_tensor = load_image()  # Generate a random image if no path is provided

    # image_path = with open ("goldfish.jpg", "r") as file:
    # image_tensor = file.read()
    # Set to None to generate a random image, or specify a path to an actual image file
    # image_tensor = load_image("german-shep.jpg")

    # at some point add code to load an image from file using os dialog.
    # print("Opening file dialog to select an image... If you want to test with a random image, simply cancel the dialog.")
    print(f"Loading image... If you want to test with a random image, set the image_path variable to None.")
    # image_tensor = load_image("goldfish.jpg")

    print("Predicting based on inference from the model...")
    predictions = predict(model, image_tensor)

    print("Top-5 Predictions:")
    for cls, prob in predictions:
        print(f"{labels[cls]}: {prob:.4f}")

    print(image_tensor.shape)


