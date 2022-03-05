# Project 4

My name is Bryce Whitney and this page details my experiments for project 4 in Advanced Applied Machine Learning. This page includes testing different regressor combinations in boosting and analyzing the effectiveness of each method, along with a theoretical discussion and application of LightGBM.

## Concrete Dataset

For all the analysis I used the [Concrete Compressive Strength Data Set](https://archive.ics.uci.edu/ml/datasets/concrete+compressive+strength) to train the  models and compare the results. The structure of the data can be seen below:
  
![](ConcreteDataFrame.png)

In this dataset, the dependent variable we are trying to model is the `Concrete compressive strength`. All the other features are used as the input variables in X. As you can see, the features have a wide range of scales that they are measured on. Some features such as `Coarse Aggregate` have measurements over 1000, while other variables such as `Superplasticizer` don't reach the double digit threshold. To account for this, I normalized the training and testing data before passing it into the regressors.

## Regressor Boosting Combinations

# LightGBM

## Theorteical Discussion

Light Gradient Boosted Machine, or LightGBM for short, ...

## Application

Similar to the Boosted Regressions conducted above, I did a corssvalidation with 5 splits to measure the effectiveness of the LightGBM algorithm for predicting the compressive strength of concrete. I measured both the MSE and MAE and recorded the crossvalidated means at the end. 

```python
import lightgbm as lgb

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=410)

mse = []
mae = []

for idxTrain, idxTest in kf.split(X):
    Xtrain, Xtest = X[idxTrain], X[idxTest]
    ytrain, ytest = y[idxTrain], y[idxTest]
    
    # Scale the data
    scale = StandardScaler()
    Xtrain_ss = scale.fit_transform(Xtrain)
    Xtest_ss = scale.transform(Xtest)
    
    # Fit the model and make predictions
    model = lgb.LGBMRegressor(random_state = 410)
    model.fit(Xtrain_ss, ytrain)
    y_pred = model.predict(Xtest_ss)

    mse.append(mean_squared_error(ytest, y_pred))
    mae.append(mean_absolute_error(ytest, y_pred))
    
print("Crossvalidated MSE for LightGBM: ", np.mean(mse))
print("Crossvalidated MAE for LightGBM: ", np.mean(mae))
```
The results from the experiment are as follows: \
**Crossvalidated MSE: 20.389** \
**Crossvalidated MAE: 2.977**

TODO: Compare to the results above
