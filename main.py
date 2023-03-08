import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Load the model
model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)
model.eval()

# Set the input and output folder paths
input_folder = "testing/image_2"
output_folder = "output"
gt_folder = "semantic_rgb"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Define the preprocessing steps
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Loop over all the images in the input folder and process each image
for file_name in os.listdir(input_folder):
    print(file_name)

    # Load the input image
    input_path = os.path.join(input_folder, file_name)
    input_image = Image.open(input_path)

    # Preprocess the input image
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

    # Move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    # Get the model's output
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    # Create a color palette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # Plot the semantic segmentation predictions of 21 classes in each color
    output_image = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
    output_image.putpalette(colors)
    output_path = os.path.join(output_folder, file_name)
    output_image.save(output_path, "png")

iou_score = 0
for file_name in os.listdir(output_folder):
        # Check if the file is a PNG image
        if not file_name.endswith(".png"):
            continue
        # Load the segmented image
        segmented_path = os.path.join(output_folder, file_name)
        segmented_image = Image.open(segmented_path).convert("RGB")
        segmented_array = np.array(segmented_image)

        # Load the corresponding ground truth image
        gt_path = os.path.join(gt_folder, file_name)
        gt_image = Image.open(gt_path)
        gt_array = np.array(gt_image)

        intersection = np.logical_and(gt_image, segmented_image)
        union = np.logical_or(gt_image, segmented_image)
        iou_score = np.sum(intersection) / np.sum(union) + iou_score

print(iou_score)

# Show the output images
import matplotlib.pyplot as plt

plt.show()
