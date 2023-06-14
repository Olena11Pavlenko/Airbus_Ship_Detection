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

Model inference

Semantic segmentation model for ship detection is not performing satisfactorily. There could be several reasons why this is happening. Here are some common problems and potential ways to address them:

1. Limited Training Data 
Training a deep learning model with only 1,000 images can lead to limited generalization ability. It's possible that the model is not exposed to enough variations in ship appearances, backgrounds, and lighting conditions.
Solution: Collecting or acquiring a larger and more diverse dataset that includes a variety of ship images, as well as images containing clouds and docks. Augmenting the existing dataset by applying transformations like rotations, flips, and brightness adjustments can also help diversify the training examples.
2. Class Imbalance
The number of ship pixels is significantly smaller than the number of non-ship pixels in the training dataset, the model might be biased towards predicting the majority class (non-ship) more frequently. This can lead to poor ship detection performance.
Solution: Using data augmentation techniques to balance the class distribution during training. This can involve oversampling the ship pixels or undersampling the non-ship pixels to create a more balanced training set. Another approach is to use loss functions that can handle class imbalance, such as focal loss or weighted cross-entropy.
3. Limited Contextual Information: 
By cropping the images to focus only on ships, might been removed valuable contextual information that could aid in ship detection. Ships are often surrounded by clouds, docks, or other objects, and the model needs to understand the relationships between ships and their surroundings.
Solution: Including images with clouds, docks, and other contextual elements during training. By providing the model with a broader range of training examples, it can learn to recognize ships in different contexts and improve its ability to distinguish between ships and similar objects.
4. Loss Function: 
While the dice score is commonly used for semantic segmentation, it may not be the most suitable choice for this specific problem. Dice score primarily measures the overlap between predicted and ground truth masks, but it might not capture all aspects of ship detection accuracy.
Solution: Experiment with different loss functions that are tailored to ship detection tasks. For instance, consider using a combination of dice loss and binary cross-entropy loss to balance the objectives of capturing accurate boundaries and correctly classifying ship pixels. Or even advanced loss functions such as focal loss or Lovász loss, which are designed to handle imbalanced classes and encourage better pixel-level predictions.
5. Model Architecture: 
It's possible that the current architecture and its depth is not able to effectively capture the necessary ship features or contextual information.
Solution: Experimenting with different architectures that have shown success in semantic segmentation tasks, such as DeepLab or Mask R-CNN. These architectures often have specialized modules or skip connections that can capture fine-grained details and improve segmentation accuracy.
6. Model Regularization: 
Adding regularization techniques to the network architecture to control model complexity and avoid over-reliance on specific features.
7. Model Ensemble: 
Ensemble methods, such as averaging or voting, can help mitigate individual model weaknesses and enhance the robustness of your ship detection system.



Feel free to modify the code and experiment with different settings to improve the performance of the ship detection model.

For any questions or further information, please feel free to contact me.

Happy ship detection!