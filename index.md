# Project 4

My name is Bryce Whitney and this page details my experiments for project 4 in Advanced Applied Machine Learning. This page includes testing different regressor combinations in boosting and analyzing the effectiveness of each method, along with a theoretical discussion and application of LightGBM.

## Concrete Dataset

For all the analysis I used the [Concrete Compressive Strength Data Set](https://archive.ics.uci.edu/ml/datasets/concrete+compressive+strength) to train the  models and compare the results. The structure of the data can be seen below:

![]("ConcreteDataFrame.png")

In this dataset, the dependent variable we are trying to model is the `Concrete compressive strength`. All the other features are used as the input variables in X. As you can see, the features have a wide range of scales that they are measured on. Some features such as `Coarse Aggregate` have measurements over 1000, while other variables such as `Superplasticizer` don't reach the double digit threshold. To account for this, I normalized the training and testing data before passing it into the regressors.

## Regressor Boosting Combinations

# LightGBM

## Theorteical Discussion

## Application
