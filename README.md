Semantic Segmentation Model for Ship Detection

This repository contains the code for building a semantic segmentation model for ship detection. The model is trained using the UNet architecture and can be used to identify ships in images.

Dataset

The model is trained on a ship detection dataset, which consists of images and corresponding ship masks. The train_v2 directory contains the ship images, and the train_ship_segmentations_v2.csv file provides the annotations in the form of encoded pixels.

Requirements

To run the code and deploy the model, you need to have the following dependencies installed:

Python 3.10
TensorFlow 2.10.0
pandas
numpy
matplotlib
PIL

You can install the required packages by running the following command:

`conda create --name <env> --file requirements.txt`

Usage

Follow the steps below to deploy the semantic segmentation model for ship detection:

1. Download the ship detection dataset and organize it as mentioned above.
2. Open the model_training.ipynb notebook.
3. Update the paths to the train images directory and the train CSV file in the notebook.
4. Optionally, adjust the hyperparameters such as image size, crop size, batch size, and number of epochs according to your requirements.
5. Run the notebook to train the model. The trained model will be saved as full_model in the current directory.
6. Once the model is trained, you can use it for ship detection.


To perform ship detection on a new image, follow these steps:

1. Open the predictions.ipynb notebook.
2. Update the path to the image you want to test in the notebook.
3. Run the notebook to load the trained model and perform ship detection on the input image.
4. The result will be displayed, showing the original image and the predicted ship mask.

Feel free to modify the code and experiment with different settings to improve the performance of the ship detection model.

For any questions or further information, please feel free to contact me.

Happy ship detection!