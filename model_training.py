import numpy as np
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.losses import binary_crossentropy

# Define paths
train_images_dir = './images/train_v2'
train_csv_file = './train_ship_segmentations_v2.csv'

# Define hyperparameters
image_size = (768, 768)
crop_size = (128, 128)
batch_size = 16
epochs = 10

# Preprocess the data
raw_df = pd.read_csv(train_csv_file)
raw_df['EncodedPixels'].fillna('', inplace=True)
df = raw_df.drop_duplicates(subset='ImageId').copy()
df['ships'] = df['EncodedPixels'].apply(lambda x: 0 if x == '' else 1)
# Remove empty images
no_ships = df['ships'] == 0
num_rows = no_ships.sum()
rows_to_remove = int(1.0 * num_rows)
np.random.seed(42)
rows_to_remove_indices = np.random.choice(df[no_ships].index, size=rows_to_remove, replace=False)
df.drop(rows_to_remove_indices, inplace=True)

# Split the data into training and validation sets
df_train_split, df_val_split = train_test_split(df,test_size=200, train_size=1000,
                                                stratify=df['ships'],
                                                random_state=42)
# Define the UNet model

def unet_model(input_shape):
    inputs = keras.Input(shape=input_shape)

    # Contracting path
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    # Bottom
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)

    # Expanding path
    up4 = layers.UpSampling2D(size=(2, 2))(conv3)
    up4 = layers.Conv2D(128, 2, activation='relu', padding='same')(up4)
    merge4 = layers.concatenate([conv2, up4], axis=3)
    conv4 = layers.Conv2D(128, 3, activation='relu', padding='same')(merge4)
    conv4 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv4)

    up5 = layers.UpSampling2D(size=(2, 2))(conv4)
    up5 = layers.Conv2D(64, 2, activation='relu', padding='same')(up5)
    merge5 = layers.concatenate([conv1, up5], axis=3)
    conv5 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge5)
    conv5 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv5)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv5)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Create the model
model = unet_model(crop_size + (3,))

# Callbacks for controling proccess of training
checkpoint = ModelCheckpoint('./best_model',
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')
                             #saves best result of training

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                   patience=1, verbose=1, mode='min',
                                   min_delta=0.0001, cooldown=0, min_lr=1e-8) 
                                   #reduces learning rate when a metric has stopped improving

early_stopping = EarlyStopping(monitor="val_loss", mode="min", 
                               verbose=1,patience=10)
                               #responses for stopping of model when a quality of model has stopped improving

callbacks_list = [checkpoint, reduceLROnPlat, early_stopping] # list of callbacks

def get_img_mask(df, row, train_images_dir, image_size):
    image_path = os.path.join(train_images_dir, row['ImageId'])
    mask = np.zeros(image_size[0] * image_size[1], dtype=np.uint8)
    if row['EncodedPixels'] != '':
        pixels = [int(x) for x in row['EncodedPixels'].split()]
        coordinates = [(pixels[i], pixels[i + 1]) for i in range(0, len(pixels), 2)]
        for coord in coordinates:
            start, length = coord
            mask[start:start + length] = 1
    mask = mask.reshape(image_size).T

    image = load_img(image_path, target_size=(image_size))
    image = img_to_array(image)
    return image, mask

def crop_image_and_mask(image, mask, crop_size):
    image = Image.fromarray(image.astype(np.uint8))
    mask = Image.fromarray(mask.astype(np.uint8))

    width, height = image.size
    crop_width, crop_height = crop_size

    cropped_images = []
    cropped_masks = []

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

            # Crop the mask with the same coordinates
            cropped_mask = mask.crop((left, upper, right, lower))

            # Append the cropped image and mask to the lists
            cropped_images.append(np.array(cropped_image))
            cropped_masks.append(np.array(cropped_mask))

    return cropped_images, cropped_masks

def find_best(cropped_images, cropped_masks):
    max_area_index = max(range(len(cropped_masks)), key=lambda i: sum(sum(row) for row in cropped_masks[i]))
    return cropped_images[max_area_index], cropped_masks [max_area_index]

# Prepare data generator

class DataGenerator:
    def __init__(self, df, batch_size, image_size):
        self.df = df
        self.batch_size = batch_size
        self.image_size = image_size
        self.datagen = ImageDataGenerator(rescale=1.0/255.0)

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def generate(self):
        while True:
            batch_indices = np.random.choice(len(self.df), size=self.batch_size, replace=False)
            batch_df = self.df.iloc[batch_indices]
            images = []
            masks = []
            for _, row in batch_df.iterrows():
                image, mask = get_img_mask(batch_df, row, train_images_dir, self.image_size)
                cropped_images, cropped_masks = crop_image_and_mask(image, mask, crop_size)
                best_image, best_mask = find_best(cropped_images, cropped_masks)
                images.append(best_image)
                masks.append(best_mask.astype(float))

            images = np.array(images)
            masks = np.array(masks)

            # Apply data augmentation using ImageDataGenerator
            images = self.datagen.flow(images, shuffle=False).next()

            yield images, masks

def dice_score(pred_masks, true_masks, smooth=1e-7):
    intersection = K.sum(pred_masks * true_masks)
    union = K.sum(pred_masks) + K.sum(true_masks)

    dice = (2.0 * intersection + smooth) / (union + smooth)

    return dice


def bce_dice_loss(y_pred, y_target):
    # dice loss metric
    return (1-dice_score(y_pred, y_target))

# Compile the model

model.compile(optimizer=Adam(1e-3, decay=1e-6), loss=bce_dice_loss, metrics=['binary_accuracy', dice_score])

# Create data generator
train_generator = DataGenerator(df_train_split, batch_size, image_size).generate()
val_generator = DataGenerator(df_val_split, batch_size, image_size).generate()

history = model.fit(train_generator,
                    epochs=epochs,
                    steps_per_epoch=len(df_train_split) // batch_size,
                    validation_data=val_generator,
                    validation_steps=len(df_val_split) // batch_size,
                    callbacks=callbacks_list,
                    verbose=1)
model.save("./full_model")