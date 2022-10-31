# Predicting Heart-disease

> GD Main.ipynb is the main program in jupyter notebook.

> GD.py is just the copy of whole program in python file.

## Business Problem

To predict whether a patient has heart disease or not by analyzing various features (BP, BodyFat, etc.,) using a Machine learning approach *(Logistic Regression using Gradient Descent in this case)*.

## Objectives

* Implement an optimization method (gradient descent) that allows to train logistic regression models. 
* Understand the relationship between the optimization objective (cross-entropy error), training set error, and test set error, and how these relate to generalization.
* Understand the impact of parameters and design choices on the convergence and performance of the gradient descent algorithm and the learned models.
* Effectively communicate technical results in writing. 

## Dataset

Train and test the model on the *Cleveland* dataset. 

> learn more about dataset here: https://archive.ics.uci.edu/ml/datasets/Heart+Disease

## Detailed Description

First task is to encode a gradient descent algorithm for learning a logistic regression model. 

The function should:
  - *Take Inputs*
    - as a Matrix X, where each row corresponds to a training example, and, 
    - a column vector y where each row corresponds to a label.
  - *Return*
    - the learned weight vector w (which should have one more element than each training example, with the convention that this first element represents the intercept/bias), 
    - the cross-entropy error, and,
    - the classification (0/1) error it achieved on the training set. 
  - Use a learning rate *eta = 10^{-5}*.
  - Automatically terminate the algorithm if the magnitude of each term in the gradient is below *10^{-3}* at any step.

Run experiments on the training set when using three different bounds on the maximum number of iterations: 
- Ten thousand (10,000) iterations, 
- One hundred thousand (100,000) iterations, and,
- One million (1,000,000) iterations. 

Report 
  - The *Cross-entropy error* on the training set. 
  - The *Classification error* on the training data.
  - The *Time* it took for the model to train. 

In your report, discuss the generalization properties (the difference between training and test set classification errors) of the model. How does this relate to the cross-entropy error on the training set?

Now train and test a logistic regression model using whichever library you used for Homework 1. Compare the results with the best ones you achieved and also compare the time taken to achieve the results.

Finally, scale each of the features in your training data by subtracting the mean and dividing by the standard deviation for each of the features in advance of calling the training function. Experiment with the learning rate eta (you may want to start by trying different orders of magnitude), this time using a tolerance (how close to zero you need each element of the gradient to be in order to terminate) of 10^{-6}. Report the results in terms of number of iterations until the algorithm terminates, and also the final cross-entropy error. Spend some time discussing and analyzing your results in your writeup.
