import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True
)


img1, _ = dataset[0]
img2, _ = dataset[1]
transform_pipeline = transforms.Compose([
    transforms.Resize((128, 128)),                 # Resize
    transforms.Grayscale(num_output_channels=3),   # Grayscale
    transforms.RandomRotation(30),                 # Rotation
    transforms.RandomHorizontalFlip(p=1.0),        # Flip
    transforms.ToTensor(),                         # Tensor
    transforms.Normalize(mean=[0.5,0.5,0.5],
                         std=[0.5,0.5,0.5])        # Normalize
])
processed_img1 = transform_pipeline(img1)
processed_img2 = transform_pipeline(img2)
def print_info(img, name):
    print(f"{name} Shape:", img.shape)
    print(f"{name} Pixel Range:", img.min().item(), "to", img.max().item())


print_info(processed_img1, "Image 1")
print_info(processed_img2, "Image 2")
def show_images(img1, img2, title):
    fig, axes = plt.subplots(1, 2, figsize=(8,4))
    axes[0].imshow(img1.permute(1,2,0))
    axes[1].imshow(img2.permute(1,2,0))
    axes[0].set_title("Image 1")
    axes[1].set_title("Image 2")
    for ax in axes:
        ax.axis("off")
    plt.suptitle(title)
    plt.show()


show_images(processed_img1, processed_img2, "Processed Images")
batch_tensor = torch.stack([processed_img1, processed_img2])
print("Final Tensor Shape:", batch_tensor.shape)
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])


def apply_sharpen(img):
    img_np = img.permute(1,2,0).numpy()
    filtered = cv2.filter2D(img_np, -1, kernel)
    return torch.tensor(filtered).permute(2,0,1)


sharpened_img1 = apply_sharpen(processed_img1)
sharpened_img2 = apply_sharpen(processed_img2)


show_images(sharpened_img1, sharpened_img2, "Sharpened Images")
