# ANN Implementation with 3 Layers from Scratch

## 1. Task to be Performed

This repository contains a detailed implementation of a neural network with three layers designed for classification tasks. The primary focus is to provide insights into each aspect of the neural network, including architecture, training, and evaluation.

## 2. Dataset Used

The neural network is trained on the "Dry Bean Dataset" obtained from the UCI Machine Learning Repository (Dataset ID: 602). The dataset contains various features related to dry beans, and the network aims to classify these beans based on their characteristics.

**Dataset Description:**
Seven different types of dry beans were used in this research, taking into account the features such as form, shape, type, and structure by the market situation. A computer vision system was developed to distinguish seven different registered varieties of dry beans with similar features in order to obtain uniform seed classification. For the classification model, images of 13,611 grains of 7 different registered dry beans were taken with a high-resolution camera. Bean images obtained by a computer vision system were subjected to segmentation and feature extraction stages, and a total of 16 features; 12 dimensions and 4 shape forms, were obtained from the grains.

## 3. Libraries Used and Their Purpose

- **NumPy (np):** Used for efficient numerical operations and array manipulations, providing the foundational support for mathematical computations in the neural network.
- **scikit-learn (train_test_split, StandardScaler, LabelEncoder):** Enables convenient data preprocessing. `train_test_split` is used for dataset splitting, `StandardScaler` standardizes input features, and `LabelEncoder` encodes categorical labels.
- **Keras (to_categorical):** Facilitates one-hot encoding of categorical labels, ensuring proper interpretation by the neural network during training.
- **ucimlrepo (fetch_ucirepo):** Allows easy retrieval of datasets from the UCI Machine Learning Repository. In this context, it fetches the specific dataset (`id=602`) related to dry bean classification.
- **Matplotlib (plt):** Utilized for creating visualizations. In this code, it plots the training and validation costs over epochs, offering insights into the neural network's learning progress.
- **Warnings:** The `warnings` module is used to suppress unnecessary warnings during execution, ensuring a cleaner and more readable output.

These libraries collectively contribute to efficient data processing, neural network modeling, and result visualization.

## 4. Working of the Neural Network Architecture

### Architecture Overview

The neural network architecture is a three-layer feedforward network:

 ![sample SVG image](https://github.com/aamash1wnl/3-layer-ANN-scratch/blob/main/extras/nn(1).svg)
 
1. **Input Layer:**
   - Nodes: The number of input nodes is determined by the features of the dataset.

2. **Hidden Layer:**
   - Activation Function: Leaky ReLU (Rectified Linear Unit).
   - Weights and Biases: Initialized with random values.
   - Purpose: Extracts complex patterns and features from the input data.

3. **Output Layer:**
   - Activation Function: Sigmoid.
   - Purpose: Produces the final classification probabilities.

### Process Overview

1. **Data Preprocessing:**
   - Data is split into training and testing sets.
   - Standard scaling is applied to the features.
   - Categorical labels are one-hot encoded.

2. **Initialization:**
   - Weights and biases for each layer are initialized with random values.

3. **Forward Propagation:**
   - Input data passes through the network to make predictions.
   - Leaky ReLU is used as the activation function for the hidden layer.
   - Sigmoid activation is used in the output layer.

4. **Cost Function:**
   - Cross-entropy loss is used as the cost function.
   - Measures the difference between predicted and true labels.

5. **Backpropagation:**
   - Gradients are calculated for each layer.
   - Weights and biases are updated using gradient descent.

6. **Training Loop:**
   - Iterates for a specified number of epochs.
   - Learning rate is adjusted during training.

### Neural Network Performance

- **Accuracy on Validation Set:** 0.6449559255631734

### Code Structure

The code is organized into a class (`ANN3layer`) with methods for data preprocessing, forward and backward propagation, cost computation, and gradient descent.

## 5. Results

The final training and validation costs are printed, and a plot of the training and validation costs over epochs is provided. Additionally, the accuracy on the validation dataset is calculated.

## 6. Guidelines and Tips

- Adjust hyperparameters like learning rate and the number of epochs for better performance.
- Experiment with different activation functions and network architectures.
- Visualize metrics like accuracy, precision, and recall for a comprehensive evaluation.

## 7. Link to Dataset and Its Paper

- **Dataset Link:** [Dry Bean Dataset (UCI)](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)
- **Dataset Paper:** [Multiclass classification of dry beans using computer vision and machine learning techniques](https://www.semanticscholar.org/paper/Multiclass-classification-of-dry-beans-using-vision-Koklu-%C3%96zkan/e84c31138f2f261d15517d6b6bb8922c3fe597a1)

Feel free to explore, experiment, and contribute to the improvement of this detailed neural network implementation!
