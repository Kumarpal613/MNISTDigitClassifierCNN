# MNIST Handwritten Digit Classification using CNN


## Overview

This project implements a Convolutional Neural Network (CNN) to classify handwritten digits (0-9) from the widely used **MNIST dataset**. The MNIST dataset consists of grayscale images of handwritten digits, and this project demonstrates the application of deep learning techniques, specifically CNNs, to achieve accurate image classification.

## Technologies Used

* **Python:** The primary programming language.
* **[Specify Framework: TensorFlow,Keras]:** Deep learning framework used for building and training the CNN model.
* **NumPy:** For numerical computations.

## Dataset

The **MNIST Handwritten Digit Dataset** is used for this project. It contains a training set of 60,000 examples and a test set of 10,000 examples of handwritten digits (0-9). The dataset is readily available through popular deep learning libraries like [e.g., TensorFlow Keras, Torchvision].

## Model Architecture

> The CNN model consists of the following layers:
>
> 1.  A convolutional layer with [Number] filters of size [Kernel Size] and [Activation Function] activation.
> 2.  A [Pooling Type] pooling layer with a pool size of [Pool Size].
> 3.  Another convolutional layer with [Number] filters of size [Kernel Size] and [Activation Function] activation.
> 4.  Another [Pooling Type] pooling layer with a pool size of [Pool Size].
> 5.  A flattening layer to convert the 2D feature maps to a 1D vector.
> 6.  A fully connected (dense) layer with [Number] units and [Activation Function] activation.
> 7.  An output fully connected (dense) layer with 10 units (one for each digit) and [Softmax] activation for probability distribution.

## Training Details


> The model was trained using the following parameters:
> 
> * **Optimizer:** [ Adam, SGD]
> * **Loss Function:** [ Categorical Cross-entropy]
> * **Batch Size:** 1313
> * **Number of Epochs:** 5

## Evaluation

*(Provide details about the evaluation and the performance achieved. )*

> The trained CNN model was evaluated on the MNIST test set. The evaluation metrics used include:
>
> * **Accuracy:** 0.9869
> * **Loss:** 0.0395
>


## Usage

*(Optional: Briefly explain how someone can run the code and potentially test it with new digit images if you've included that functionality.)*

> After running the main script, the model will train on the MNIST dataset and then evaluate its performance on the test set. [If applicable, mention how to test with new images.]


