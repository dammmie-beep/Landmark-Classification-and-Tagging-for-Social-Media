# Landmark-Classification-and-Tagging-for-Social-Media

This project focuses on building and deploying a Convolutional Neural Network (CNN) to classify landmarks in images. The project is divided into three main parts: creating a CNN from scratch, using transfer learning, and deploying the best model in a simple application. The following outlines the steps taken and the criteria met in each stage of the project.

## 1. CNN to Classify Landmarks from Scratch (`cnn_from_scratch.ipynb`)

### Data Preparation
- **Data Loading:** In `src/data.py`, the `get_data_loaders` function is implemented to load the train, validation, and test datasets using `ImageFolder`. The `data_transforms` dictionary defines transformations for each dataset, including resizing, cropping, converting to tensor, normalizing, and data augmentation.
- **Data Visualization:** A function, `visualize_one_batch`, is implemented to visualize a batch of images and their labels from the training dataset. This helps in verifying that the data loading and transformations are working correctly.

### Model Definition
- **Model Architecture:** The CNN model is implemented in `src/model.py` within the `MyModel` class. The architecture includes multiple layers, and the output layer is designed to handle the specified number of classes (`num_classes`). Dropout is applied where necessary, controlled by the `dropout` parameter.
- **Loss and Optimizer:** In `src/optimization.py`, functions `get_loss` and `get_optimizer` are defined to return the CrossEntropy loss and initialize optimizers (SGD and Adam) with the provided parameters.

### Training and Validation
- **Training Loop:** The `train_one_epoch` function in `src/train.py` is implemented to handle the training process, including forward passes, loss computation, and backpropagation. The `valid_one_epoch` function manages the validation process, evaluating the model without updating gradients.
- **Optimization:** The `optimize` function implements learning rate scheduling and model checkpointing based on validation loss improvements. The model is trained until the validation loss converges.

### Model Testing and Export
- **Testing:** The model is tested on the test set, achieving an accuracy of at least 50%.
- **Exporting the Model:** The best-performing model is exported using TorchScript. In `src/predictor.py`, the `forward` method is implemented to apply transforms, compute logits, and return softmax predictions. The exported model is tested, and a confusion matrix is generated to verify its performance.

## 2. Transfer Learning for Landmark Classification (`transfer_learning.ipynb`)

### Transfer Learning Architecture
- **Model Setup:** In `src/transfer.py`, the transfer learning model is created by loading a pre-trained network, freezing its parameters, and adding a new linear layer for the specific classification task. The `get_model_transfer_learning` function is used to instantiate this model.

### Training, Validation, and Testing
- **Training Process:** The model is trained with reasonable hyperparameters, ensuring that both training and validation losses decrease over epochs.
- **Model Evaluation:** The transfer learning model achieves a test accuracy of at least 60%.
- **Model Export:** The trained model is exported using TorchScript and saved as `checkpoints/transfer_exported.pt`.

## 3. Deploying the Model in an Application (`app.ipynb`)

### Application Development
- **App Setup:** The notebook contains all necessary code to load the exported TorchScript model (from either the scratch-built CNN or the transfer learning model) and use it for inference in a simple app.
- **App Testing:** The app is tested with an image that is not part of the training or test set. The app successfully displays the image along with the model's predictions.
