# HousingPricePrediction
Predict house prices using Ames dataset.

Initial 11/22/21
This is a first attempt at using the Ames dataset, as accessed from Kaggle, to predict housing prices.  The dataset comprises a mixture of numerical, as well as nominal and ordinal categorical data.  The XGBoost regressor was implemented.

For purposes of model validation, validation and learning curves were plotted.  The validation curves vary the number of estimators and the learning rate, and plot the resulting accuracies of the training set and validation set via 5-fold cross-validation.  These are subsets of the full dataset provided in the 'train.csv' file (75% training, 25% validation).  n_estimators=150 and learning_rate=0.11 yielded best results.  The model suffers from low accuracy, as apparent from a plateauing in the learning curve and validation curves at ~80% for the validation curves.  Additionally, there is significant overfitting, as indicated by the wide gap between the training and validation curves in all plots.

A plot of the residuals shows no discernable pattern, indicating relationships between individual features and the target have been accounted for in the model.  The mean absolute error and r^2 values for the training set were $7706 and 0.98, respectively, whereas the same values for the validation set were $23194 and 0.78, respectively.  Hopefully, prudent selection of appropriate features will improve model performance.  Also, it was noted that the scikit-learn OrdinalEncoder yielded Unknown categories errors, and the only workaround was to delete the offending labels, which is less than ideal.  This may have contributed to the poor model performance.

Update 11/23/21
Uploaded files pertaining to numerical features only.  Tested a variety of regressors including 4 linear models (Linear Regression, Lasso, Elastic Net, Ridge), linear support vector regressor, and the XGBoost and Random Forest ensemble regressors, using default parameters within scikit-learn.  The ensemble models far out-performed the others, with the best performance exhibited by XGBoost.

Model                               MAE ($)
XGBoost                             5597
RandomForest                        7198
Ridge                               22585
Lasso                               22592
LinearRegression                    22593
ElasticNet (50% L1, 50% L2)         23110
LinearSVR                           28532

Validation curves show little change in accuracy with varying # estimators and learning rate within XGBoost model.  n_estimators=100 and learning_rate=0.113 were used for model.  An improved MAE and R2 of $19163 and 0.85, respectively, for the validation set exhibited when only the numerical features were fitted.  This suggests further improvement can be achieved with careful feature engineering.  

Also included is a bar chart plotting the Mutual Importance (MI) of the numerical features.  After the top 20 features, the importance degrades rapidly, suggesting these can be neglected in future attempts.  Furthermore, features related to the garage ('GarageArea', 'GarageCars', 'GarageYrBlt') are in the top 10 of feature importance.  However, the 'GarageYrBlt' relationship with 'SalePrice' is problematic since 81 features appear as outliers with values of 0, indicating no garage, thereby skewing the trend.  Because these represent a relatively small sample, it seems reasonable to remove samples without a garage.

An examination of pair plots shows the majority of the features in the top 20 MI features exhibit a linear trend.  Those that don't, but show relatively flat histograms for a small number features, will be excluded ('Fireplaces', 'Halfbath', 'BedroomAbvGr').  This should yield a set of features with a relatively linear trend vs. SalePrice, which may improve the performances of the linear models, in addition to careful tuning of their input parameters, which may yield improvements over the XGBoost model.
