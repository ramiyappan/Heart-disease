# Predicting Heart-disease

The objectives of this assignment are the following:

-- Implement an optimization method (gradient descent) that allows you to train logistic regression models.
-- Better understand the relationship between the optimization objective (cross-entropy error), training set error, and test set error, and how these relate to generalization.
-- Understand the impact of parameters and design choices on the convergence and performance of the gradient descent algorithm and the learned models.
-- Effectively communicate technical results in writing. 

Detailed Description:

Your first task is to encode a gradient descent algorithm for learning a logistic regression model. Your function should take as input a matrix X, where each row corresponds to a training example, and a column vector y where each row corresponds to a label, and return the learned weight vector w (which should have one more element than each training example, with the convention that this first element represents the intercept), the cross-entropy error, and the classification (0/1) error it achieved on the training set. Use a learning rate eta = 10^{-5} and automatically terminate the algorithm if the magnitude of each term in the gradient is below 10^{-3} at any step.

You will train and test your model on the "Cleveland" dataset, which you can learn more about here: https://archive.ics.uci.edu/ml/datasets/Heart+Disease

Run experiments where you learn logistic regression models on the training set when using three different bounds on the maximum number   of iterations: ten thousand, one hundred thousand, and one million.  In your writeup, report all four of the following: (1) the cross-entropy error on the training set (2) the classification error on the training data (3) an estimate of the classification error on the test data obtained using Miner (4) the time it took to train your model. In your report, discuss the generalization properties (the difference between training and test set classification errors) of the model. How does this relate to the cross-entropy error on the training set?

Now train and test a logistic regression model using whichever library you used for Homework 1. Compare the results with the best ones you achieved and also compare the time taken to achieve the results.

Finally, scale each of the features in your training data by subtracting the mean and dividing by the standard deviation for each of the features in advance of calling the training function. Experiment with the learning rate eta (you may want to start by trying different orders of magnitude), this time using a tolerance (how close to zero you need each element of the gradient to be in order to terminate) of 10^{-6}. Report the results in terms of number of iterations until the algorithm terminates, and also the final cross-entropy error. Spend some time discussing and analyzing your results in your writeup.
