# Report: Predict Bike Sharing Demand with AutoGluon Solution
#### COSMAS UCHENNA NWEKE

## Initial Training
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?
TODO: I realized that kaggle errors occurs when there are negative values in the dataset and when the dataframe is having only one feature. The changes I made was to set all negative values in the predictions data to Zero, confirmed the dataset has two columns (datetime & count) and also exported the data to csv file. 

### What was the top ranked model that performed?
TODO: The top ranked model that performed is WeightedEnsemble_L3 Model.

## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?
TODO: Plotting histogram of all the available features on the train dataset, I found out that, datetime population is filled and mostly at the peak or close, season feature has 4 distant peak populations, holiday has 2 distant populations (1 peak and 1 very little) at descending order, workingday has 2 distant populations (1 peak and 1 almost half) at ascending order, weather has 3 distant populations in descending order, the temp, atemp and humidity has slightly normal distribution, the windspeed is slightly chi-square distribution and count has an F-distribution.
Adding additional features, I parsed the datetime feature, extracted and created 6 new features which are year,month,day,hour,minute and second.

### How much better did your model preform after adding additional features and why do you think that is?
TODO: My model performed extremely better, say over 80% better after adding new features because the model got additional data features that provided relevant information relating to the target variable. 

## Hyper parameter tuning
### How much better did your model perform after trying different hyper parameters?
TODO: My model performed slightly worse after hyperparameter tuning.

### If you were given more time with this dataset, where do you think you would spend more time?
TODO: I will spend more time at the hyperparameter tuning.

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.
|model|hpo1|hpo2|hpo3|score|
|--|--|--|--|--|
|initial|default|default|default|1.79857|
|add_features|default|default|default|0.62281|
|hpo|"RF: n_estimators=300"|"XGB: n_estimators=100"|"XGB: learning_rate=0.1"|0.49398|

### Create a line plot showing the top model score for the three (or more) training runs during the project.

TODO: Replace the image below with your own.
![model_train_score.png](c:\Users\OKKKKK\Downloads\project 1\model_train_score.png)

### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.

TODO: Replace the image below with your own.
![model_test_score.png](c:\Users\OKKKKK\Downloads\project 1\model_test_score.png)

## Summary
TODO: For the initial phase, the model had very high error, both in validation and test.
For the add feature phase, the model had massive improvement from both validation and test. It also got the overall best validation score.
For the HPO, the model had slightly worse validation, but best Kaggle test score.

Conclusively, the model with the best validation score isn't always the one that performs best on the test set. I need to work more on HPO, there is potential in improving the model performance
