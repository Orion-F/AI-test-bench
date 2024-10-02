import numpy as np
import matplotlib.pyplot as plt
import os

# Generate a sample image (simple gradient image)
def create_sample_image(size=(128, 128, 3)):
    img = np.zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
            img[i, j] = [(i / size[0]), (j / size[1]), 0.5]
    return img

# Load an image and preprocess it
def load_image(image_path, size=(128, 128)):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file '{image_path}' not found. Please ensure the image is in the same directory as the script.")
    from PIL import Image
    img = Image.open(image_path).convert('RGB')  # Ensure image is in RGB format
    img = img.resize(size)
    img = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    return img

# Auto-Regression: Remove chunks from the image
def autoregression_degrade(image, steps):
    degraded_images = []
    mask = np.ones_like(image)
    total_pixels = image.shape[0] * image.shape[1]
    pixels_per_step = total_pixels // steps

    for step in range(steps):
        indices = np.where(mask[:, :, 0].flatten() == 1)[0]
        if len(indices) > 0:
            remove_indices = np.random.choice(indices, size=min(pixels_per_step, len(indices)), replace=False)
            mask.reshape(-1, 3)[remove_indices] = 0
        degraded_image = image * mask
        degraded_images.append(degraded_image)
    return degraded_images

# Auto-Regression: Reconstruct to a different image
def autoregression_reconstruct(target_image, steps):
    reconstructed_images = []
    mask = np.zeros_like(target_image)
    total_pixels = target_image.shape[0] * target_image.shape[1]
    pixels_per_step = total_pixels // steps

    for step in range(steps):
        indices = np.where(mask[:, :, 0].flatten() == 0)[0]
        if len(indices) > 0:
            add_indices = np.random.choice(indices, size=min(pixels_per_step, len(indices)), replace=False)
            mask.reshape(-1, 3)[add_indices] = 1
        reconstructed_image = target_image * mask
        reconstructed_images.append(reconstructed_image)
    return reconstructed_images

# Diffusion: Add noise to the image
def diffusion_degrade(image, steps):
    degraded_images = []
    for step in range(steps):
        noise_level = (step + 1) / steps
        noisy_image = image * (1 - noise_level) + np.random.randn(*image.shape) * noise_level
        noisy_image = np.clip(noisy_image, 0, 1)
        degraded_images.append(noisy_image)
    return degraded_images

# Diffusion: Denoise to a different image
def diffusion_reconstruct(target_image, steps):
    reconstructed_images = []
    noisy_image = np.random.randn(*target_image.shape)  # Start from pure noise
    for step in range(steps):
        noise_level = (steps - step) / steps
        denoised_image = noisy_image * noise_level + target_image * (1 - noise_level)
        denoised_image = np.clip(denoised_image, 0, 1)
        reconstructed_images.append(denoised_image)
    return reconstructed_images

# Create the infographic
def create_infographic():
    steps = 5
    img_dir = 'img'

    # Ensure the img directory exists
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    original_image = load_image('puppy1.jpg')
    plt.imsave(f"{img_dir}/original.png", original_image)
    
    target_image = load_image('puppy2.png')
    plt.imsave(f"{img_dir}/target.png", target_image)
    
    blank_image = np.zeros_like(original_image)
    plt.imsave(f"{img_dir}/blank.png", blank_image)

    # Auto-Regression Process
    ar_degraded = autoregression_degrade(original_image, steps)
    ar_reconstructed = autoregression_reconstruct(target_image, steps)

    # Diffusion Process
    diff_degraded = diffusion_degrade(original_image, steps)
    diff_reconstructed = diffusion_reconstruct(target_image, steps)

    # Save degraded and reconstructed images
    for i, img in enumerate(ar_degraded):
        plt.imsave(f"{img_dir}/ar_degraded_{i+1}.png", img)

    for i, img in enumerate(ar_reconstructed):
        plt.imsave(f"{img_dir}/ar_reconstructed_{i+1}.png", img)

    for i, img in enumerate(diff_degraded):
        plt.imsave(f"{img_dir}/diff_degraded_{i+1}.png", img)

    for i, img in enumerate(diff_reconstructed):
        plt.imsave(f"{img_dir}/diff_reconstructed_{i+1}.png", img)

    # Plotting
    fig, axes = plt.subplots(4, steps + 1, figsize=(15, 10))

    # Titles and labels
    axes[0, 0].set_ylabel('Auto-Regression\nDegradation', fontsize=12)
    axes[1, 0].set_ylabel('Diffusion\nDegradation', fontsize=12)
    axes[2, 0].set_ylabel('Auto-Regression\nReconstruction', fontsize=12)
    axes[3, 0].set_ylabel('Diffusion\nReconstruction', fontsize=12)

    # Original Images
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original', fontsize=10)
    axes[0, 0].axis('off')

    axes[1, 0].imshow(original_image)
    axes[1, 0].set_title('Original', fontsize=10)
    axes[1, 0].axis('off')

    axes[2, 0].imshow(blank_image)
    axes[2, 0].set_title('Blank', fontsize=10)
    axes[2, 0].axis('off')

    axes[3, 0].imshow(np.random.randn(*original_image.shape))
    axes[3, 0].set_title('Noise', fontsize=10)
    axes[3, 0].axis('off')

    # Auto-Regression Degradation
    for i, img in enumerate(ar_degraded):
        axes[0, i + 1].imshow(img)
        axes[0, i + 1].axis('off')

    # Diffusion Degradation
    for i, img in enumerate(diff_degraded):
        axes[1, i + 1].imshow(img)
        axes[1, i + 1].axis('off')

    # Auto-Regression Reconstruction
    for i, img in enumerate(ar_reconstructed):
        axes[2, i + 1].imshow(img)
        axes[2, i + 1].axis('off')

    # Diffusion Reconstruction
    for i, img in enumerate(diff_reconstructed):
        axes[3, i + 1].imshow(img)
        axes[3, i + 1].axis('off')

    # Adjust layout and save the final infographic
    plt.tight_layout()
    plt.savefig(f'{img_dir}/infographic.png')
    plt.show()

if __name__ == "__main__":
    create_infographic()
