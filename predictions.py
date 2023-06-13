from tensorflow.keras.models import load_model
import PIL
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array


image_folder = './images/test'  # Folder containing the images for prediction
crop_size = (128, 128)
# Load the trained model
model = load_model('best_model', compile=False)


def crop_image(image, crop_size):
    width, height = image.size
    crop_width, crop_height = crop_size
    cropped_images = []

    # Iterate over the image to create multiple crops
    for x in range(0, width, crop_width):
        for y in range(0, height, crop_height):
            # Calculate the crop coordinates
            left = x
            upper = y
            right = x + crop_width
            lower = y + crop_height

            # Crop the image
            cropped_image = image.crop((left, upper, right, lower))

            # Append the cropped image and mask to the lists
            cropped_images.append(np.array(cropped_image))

    return cropped_images

def join_masks(masks, crop_size, image_size):
    x_steps = int(np.ceil(image_size[1] / crop_size[1]))
    y_steps = int(np.ceil(image_size[0] / crop_size[0]))
    image = np.zeros((image_size[0], image_size[1]), dtype=np.uint8)  # Set the dtype to uint8
    for x in range(x_steps):
        for y in range(y_steps):
            x_start = x * crop_size[1]
            y_start = y * crop_size[0]
            mask = masks[x * y_steps + y]
            mask = np.squeeze(mask, axis=-1)  # Remove the last axis if it exists
            mask = (mask * 255).astype(np.uint8)  # Scale the mask and convert to uint8
            image[y_start:y_start + crop_size[0], x_start:x_start + crop_size[1]] = mask
    return image

# Iterate over each image in the folder
for filename in os.listdir(image_folder):
    image_path = os.path.join(image_folder, filename)
    
    try:
        image = Image.open(image_path)
    except (OSError, PIL.UnidentifiedImageError):
        # Skip files that are not valid image files
        continue
    if filename.endswith('_mask.jpg'):
        continue
    crop_images = crop_image(image, crop_size)
    a = np.array(np.array(crop_images))
    print (a.shape)
    masks = model.predict(np.array(crop_images))
    mask = join_masks(masks, crop_size, image.size)
    
    # Save the masked image
    mask_image = Image.fromarray(mask)
    mask_image_path = os.path.join(image_folder, f"{os.path.splitext(filename)[0]}_mask.jpg")
    mask_image.save(mask_image_path)
    print(f"Saved masked image: {os.path.basename(mask_image_path)}")
