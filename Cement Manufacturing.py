# %% [markdown]
# *Importing necessary libraries*

# %%
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
plt.style.use('fivethirtyeight')
from sklearn.model_selection import train_test_split 
import warnings
warnings.filterwarnings('ignore')

# %%
# load the dataset
data = pd.read_csv('concrete_data.csv')
data.head()

# %%
# check the shape of the dataset
data.shape

# %%
# checking any duplicate value
data.duplicated().sum()

# %%
# drop duplicate values
data = data.drop_duplicates()

# %%
data.duplicated().sum()


# %%
data.info()

# %%
# check the datatype
data.dtypes

# %%
# checking  missing values
data.isnull().sum()

# %%
sns.heatmap(data.isnull(), cbar=False, yticklabels=False, cmap='viridis');

# %%
# checking the 5 number summary
data.describe().T

# %%
# Quartiles of cement
from scipy import stats

Q1 = data['cement'].quantile(q=0.25)
Q3 = data['cement'].quantile(q=0.75)
print('Quartile 1 is:', Q1)
print('Quartile 3 is:', Q3)
print('Inter Quartile Range is:', stats.iqr(data['cement']))

# %%
# outliers detection from IQR in original data
Lwr_outliers = Q1 - 1.5 * (Q3 - Q1)
upr_outliers = Q3 + 1.5 * (Q3 - Q1)
print('Lower limit Outliers in cement: ', Lwr_outliers)
print('Upper limit Outliers in cement: ',upr_outliers)

# %%
# checking the presence of outliers in cement
print("Number of outliers in cement upper: ", data[data['cement'] > 586.45]['cement'].count())
print("Number of outliers in cement lower: ", data[data['cement'] < -46.75000000000003]['cement'].count())

# %%
# checking outliers in cement using boxplot
sns.boxplot(data=data, x='cement', orient='h');

# %% [markdown]
# - here we see there is no outliers in cement attribute.

# %%
# Quartiles of Age
Q1 = data['age'].quantile(q=0.25)
Q3 = data['age'].quantile(q=0.75)
print('Quartile 1 is:', Q1)
print('Quartile 3 is:', Q3)
print('Inter Quartile Range of age:', stats.iqr(data['age']))

# %%
# Outliers detection in age
lwr_outliers = Q1 - 1.5 * (Q3 - Q1)
Upr_outliers = Q3 + 1.5 * (Q3 - Q1)
print('Lower limit:', lwr_outliers)
print('Upper limit:', Upr_outliers)

# %%
print('The number of outliers above upper limit:', data[data['age'] > 129.5]['age'].count())
print('The number of outliers below lower limit:', data[data['age'] < -66.5]['age'].count())

# %%
# visualize the boxplot of age 
sns.boxplot(data=data, x='age', orient='h');

# %% [markdown]
# *Histplot*

# %%
plt.figure(figsize=(10, 10))
col_list = data.columns

for i in range(len(data.columns)):
    plt.subplot(3, 3, i+1)
    plt.hist(data[col_list[i]], edgecolor='k', color='blue', bins=20)
    plt.title(col_list[i], color='k', fontsize=15)
    plt.tight_layout()

# %% [markdown]
# *Boxplot*

# %%
# plot the boxplot is there any outliers in this dataset
plt.figure(figsize=(15,10))
sns.boxplot(data=data, orient='v');

# %% [markdown]
# observing the boxplot we see that some outliers in this dataset, so we must deal with that.

# %%
# replacing outliers with median value
for cols in data.columns[:-1]:
    Q1 = data[cols].quantile(q=0.25)
    Q3 = data[cols].quantile(q=0.75)
    iqr = Q3 - Q1
    
    below = Q1 - 1.5 * iqr
    above = Q3 + 1.5 * iqr
    data.loc[(data[cols] < below) | (data[cols] > above), cols] = data[cols].median()

# %%
plt.figure(figsize=(15, 10))
sns.boxplot(data=data, orient='v')

# %% [markdown]
# Now most of the outliers are gone.

# %%
# pairplot
sns.pairplot(data=data, diag_kind='kde');

# %%
# checking the correlation 
data.corr()

# %%
# plotting correlation heatmap
plt.figure(figsize=(10,10))
sns.heatmap(data.corr(), annot=True, square=True, cmap='viridis', fmt='.2f', linewidths=0.5, linecolor='k');

# %%
data.head()

# %%
# separate our data into dependent and independent.
X = data.iloc[:, :-1]
y = data.iloc[:,-1]

# %%
# Scale the data
from scipy.stats import zscore

Xscaled = X.apply(zscore)
Xscaled_df = pd.DataFrame(Xscaled, columns=data.columns)

# %%
# split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(Xscaled, y, test_size=0.2, random_state=42)

# %% [markdown]
# ### **Building Models**

# %% [markdown]
# *1. Random Forest*

# %%
from sklearn.ensemble import RandomForestRegressor

Rfr = RandomForestRegressor()
Rfr.fit(X_train, y_train)

# %%
# make prediction
Rf_pred = Rfr.predict(X_test)

# %%
# model performance on training data 
print("Training Score is : {0:.2f}%".format(Rfr.score(X_train, y_train)*100))

# %%
# model performance on testing data
print("Testing score is: {0:.2f}%".format(Rfr.score(X_test, y_test)*100))

# %%
# Calculating the error
from sklearn import metrics 

print("The Mean Squared Error is: ", metrics.mean_squared_error(y_test, Rf_pred))

# %%
Rf_accuracy = (metrics.r2_score(y_test, Rf_pred)*100)
print(Rf_accuracy)

# %% [markdown]
# The error is little bit huge.

# %%
# Store each model accuracy to doing final comparison.
result_1 = pd.DataFrame({'Algorithm' : ['Random Forest'], 'Accuracy' : Rf_accuracy}, index={'1'})
table = result_1[['Algorithm', 'Accuracy']]
table

# %%
# using KFold cross validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

K = 20

kf = KFold(n_splits=K, shuffle=True, random_state=45)
cv = cross_val_score(Rfr, X, y, cv=K)
cv_score = np.mean(abs(cv))*100

# %%
cv

# %%
cv_score

# %%
Rf_cv = pd.DataFrame({'Algorithm' : ['Random Forest KFold'], 'Accuracy' : cv_score}, index={'2'})
table = pd.concat([table, Rf_cv])
table = table[['Algorithm', 'Accuracy']]
table

# %% [markdown]
# *2. Gradient Boost Regressor*

# %%
from sklearn.ensemble import GradientBoostingRegressor 

params = {'n_estimators' : 100, 'learning_rate' : 1, 'criterion' : 'mse'}
gbr = GradientBoostingRegressor(**params)
gbr.fit(X_train, y_train)

# %%
# prediction
gbr_pred = gbr.predict(X_test)

# %%
# Training Accuracy
print('Training Accuracy of GradientBoostingRegressor is: {0:.2f}%'.format(gbr.score(X_train, y_train)*100))

# %%
# Testing Accuracy
print('Training Accuracy of GradientBoostingRegressor is: {0:.2f}%'.format(gbr.score(X_test, y_test)*100))

# %%
gbr_accuracy = metrics.r2_score(y_test, gbr_pred)*100
gbr_accuracy

# %%
# Calculating the error
print("The Mean Squared Error is: {}".format(metrics.mean_squared_error(y_test, gbr_pred)))

# %% [markdown]
# the error higher than random forest regressor.

# %%
gb_df = pd.DataFrame({'Algorithm' : ['Gradient Boosting Regressor'], 'Accuracy' : gbr_accuracy}, index={'3'})
table = pd.concat([table, gb_df])
table = table[['Algorithm', 'Accuracy']]
table

# %%
# using KFold cross validation
gb_K = KFold(n_splits=20, shuffle=True, random_state=1)
gb_cv = cross_val_score(gbr, X, y, cv=gb_K)
gbcv_score = np.mean(abs(gb_cv))*100

# %%
gb_cv

# %%
gbcv_score

# %%
gbK_df = pd.DataFrame({'Algorithm' : ['Gradient BoostingR KFold'], 'Accuracy' : gbcv_score}, index={'4'})
table = pd.concat([table, gbK_df])
table = table[['Algorithm', 'Accuracy']]
table

# %% [markdown]
# ### **AdaBoost Regressor**

# %%
from sklearn.ensemble import AdaBoostRegressor 

Abr = AdaBoostRegressor(random_state=42)
Abr.fit(X_train, y_train)

# %%
# prediction
Abr_pred = Abr.predict(X_test)

# %%
# Training Accuracy
print('Training Accuracy of AdaBoostRegressor is: {0:.2f}%'.format(Abr.score(X_train, y_train)*100))

# %%
print('Testing Accuracy of AdaBoostRegressor is: {0:.2f}%'.format(Abr.score(X_test, y_test)*100))

# %%
Abr_accuracy = metrics.r2_score(y_test, Abr_pred)*100
Abr_accuracy

# %%
# Calculating the error
print("The Mean Squared Error is: {}".format(metrics.mean_squared_error(y_test, Abr_pred)))

# %%
Abr_df = pd.DataFrame({'Algorithm': ['AdaBoostRegressor'], 'Accuracy': Abr_accuracy}, index={'5'})
table = pd.concat([table, Abr_df])
table = table[['Algorithm', 'Accuracy']]
table

# %%
# hyper parameter tuning in AdaBoostRegressor
adr = AdaBoostRegressor(random_state=42, base_estimator=Rfr,n_estimators=100, learning_rate=0.1)
adr.fit(X_train, y_train)

# %%
adr_pred = adr.predict(X_test)

# %%
# Training Accuracy
print('Training Accuracy of AdaBoostRegressor Hyper is: {0:.2f}%'.format(adr.score(X_train, y_train)*100))

# %%
print('Testing Accuracy of AdaBoostRegressor Hyper is: {0:.2f}%'.format(adr.score(X_test, y_test)*100))

# %%
adr_accuracy = metrics.r2_score(y_test, adr_pred)*100
adr_accuracy

# %%
# Calculating the error
print("The Mean Squared Error is: {}".format(metrics.mean_squared_error(y_test, adr_pred)))

# %%
adr_df = pd.DataFrame({'Algorithm': ['AdaBoostRegressor Hyper'], 'Accuracy': adr_accuracy}, index={'6'})
table = pd.concat([table, adr_df])
table = table[['Algorithm', 'Accuracy']]
table

# %% [markdown]
# ### **KNN Regressor**

# %%
from sklearn.neighbors import KNeighborsRegressor
# Checking different values for k 
error_rate = []

for i in range(1,50):
    KNN = KNeighborsRegressor(n_neighbors=i)
    KNN.fit(X_train, y_train)
    knn_pred = KNN.predict(X_test)
    error_rate.append(np.mean(knn_pred != y_test))

# %%
# plot the figure
plt.figure(figsize=(12,6))
plt.plot(range(1,50), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Different K values')
plt.xlabel('K values')
plt.ylabel('Mean Error');

# %%
# take k=3
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train, y_train)

# %%
knn_pred = knn.predict(X_test)

# %%
print('Training Score of KNeighborsRegressor is {0:.2f}%'.format(knn.score(X_train, y_train)*100))

# %%
print('Testing Score of KNeighborsRegressor is {0:.2f}%'.format(knn.score(X_test, y_test)*100))

# %%
knn_accuracy = metrics.r2_score(y_test, knn_pred)*100
knn_accuracy

# %%
# Calculating the error
print("The Mean Squared Error is: {}".format(metrics.mean_squared_error(y_test, knn_pred)))

# %%
knn_df = pd.DataFrame({'Algorithm': ['KNeighborsRegressor'], 'Accuracy': knn_accuracy}, index={'7'})
table = pd.concat([table, knn_df])
table = table[['Algorithm', 'Accuracy']]
table

# %%
KNeighborsRegressor().get_params()

# %%
# Hyper Parameter Tuning of KNN
from sklearn.model_selection import GridSearchCV

param_grid = {'n_neighbors' : [3, 5, 7, 9, 11, 13],
              'weights' : ['uniform', 'distance'],
              'metric' : ['minkowski', 'euclidean', 'manhattan']}

# %%
gs = GridSearchCV(KNeighborsRegressor(), param_grid=param_grid, verbose=1, cv=5, n_jobs=-1)

# %%
gs.fit(X_train, y_train)

# %%
# find the best score
gs.best_score_

# %%
# get the hyper parameters with best score
gs.best_params_

# %%
# use the best HyperParameters
KNN = KNeighborsRegressor(n_neighbors=5, metric='minkowski', weights='distance')
KNN.fit(X_train, y_train)

# %%
ky_pred = KNN.predict(X_test)

# %%
print('Training Accuracy is: {0:.2f}%'.format(KNN.score(X_train, y_train)*100))

# %%
print('Testing Accuracy is: {0:.2f}%'.format(KNN.score(X_test, y_test)*100))

# %%
Hknn_accuracy = metrics.r2_score(y_test, ky_pred)*100
Hknn_accuracy

# %%
# Calculating the error
print("The Mean Squared Error is: {}".format(metrics.mean_squared_error(y_test, ky_pred)))

# %%
HK_df = pd.DataFrame({'Algorithm': ['KNeighborsRegressor Hyper'], 'Accuracy': Hknn_accuracy}, index={'8'})
table = pd.concat([table, HK_df])
table = table[['Algorithm', 'Accuracy']]
table

# %% [markdown]
# ### **Bagging Regressor**

# %%
from sklearn.ensemble import BaggingRegressor 

bg = BaggingRegressor()
bg.fit(X_train, y_train)

# %%
# make prediction
bg_pred = bg.predict(X_test)

# %%
print('Training Score is: {0:.2f}%'.format(bg.score(X_train, y_train)*100))
print('Testing Score is: {0:.2f}%'.format(bg.score(X_test, y_test)*100))

# %%
bg_accuracy = metrics.r2_score(y_test, bg_pred)*100
bg_accuracy

# %%
# calculating error
print("Mean Squared Error is: {}".format(metrics.mean_squared_error(y_test, bg_pred)))

# %%
bg_df = pd.DataFrame({'Algorithm': ['BaggingRegressor'], 'Accuracy': bg_accuracy}, index={'9'})
table = pd.concat([table, bg_df])
table = table[['Algorithm', 'Accuracy']]
table

# %%
# cross validation of BaggingRegressor
K = 20

bg_K = KFold(n_splits=K, random_state=42, shuffle=True)
bg_cv = cross_val_score(bg, X, y, cv=bg_K)
bg_rslt = np.mean(abs(bg_cv))*100

# %%
bg_cv

# %%
bg_rslt

# %%
bg_KFold = pd.DataFrame({'Algorithm' : ['BaggingRegressorKFold'], 'Accuracy' : [bg_rslt]}, index={'10'})
table = pd.concat([table, bg_KFold])
table = table[['Algorithm', 'Accuracy']]
table

# %% [markdown]
# ### **Support Vector Regressor**

# %%
from sklearn.svm import SVR 

svr = SVR(kernel='linear')
svr.fit(X_train, y_train)

# %%
# make prediction
svr_pred = svr.predict(X_test)

# %%
print('Training Score of SVR is: {0:.2f}%'.format(svr.score(X_train, y_train)*100))
print('Testing Score of SVR is: {0:.2f}%'.format(svr.score(X_test, y_test)*100))

# %%
svr_accuracy = metrics.r2_score(y_test, svr_pred)*100
svr_accuracy

# %% [markdown]
# The performance is very very bad.Need to improve that Using some hyper parameter tuning.

# %%
print('Mean Squared Error is: {}'.format(metrics.mean_squared_error(y_test, svr_pred)))

# %% [markdown]
# and the error is very huge.This model is  actually not good.

# %%
svr_df = pd.DataFrame({'Algorithm': ['SupportVectorRegressor'], 'Accuracy': [svr_accuracy]}, index={'11'})
table = pd.concat([table, svr_df])
table = table[['Algorithm', 'Accuracy']]
table

# %%
# Hyper parameter tuning in SVR()
SVR().get_params()

# %%
param_grid = {'kernel' : ['poly', 'linear', 'rbf', 'sigmoid'],
              'degree' : list(range(1,11)),
              'C' : [1.0, 1.2, 1.5, 0.5]}

# %%
gs_svr = GridSearchCV(SVR(), param_grid=param_grid, verbose=1, cv=5)

# %%
gs_svr.fit(X_train, y_train)

# %%
# finding best score
gs_svr.best_score_

# %%
gs_svr.best_params_

# %%
Svr = SVR(C=1.5, degree=1, kernel='rbf')
Svr.fit(X_train, y_train)

# %%
Svr_pred = Svr.predict(X_test)

# %%
print('Training score of svr_hyper is: {0:.2f}%'.format(Svr.score(X_train, y_train)*100))
print('Testing score of svr_hyper is: {0:.2f}%'.format(Svr.score(X_test, y_test)*100))

# %%
Svr_accuracy = metrics.r2_score(y_test, Svr_pred)*100
Svr_accuracy

# %%
print('Mean Squared Error is: {}'.format(metrics.mean_squared_error(y_test, Svr_pred)))

# %%
Svr_df = pd.DataFrame({'Algorithm': ['SupportVectorRegressorHyper'], 'Accuracy': [Svr_accuracy]}, index={'12'})
table = pd.concat([table, Svr_df])
table = table[['Algorithm', 'Accuracy']]
table

# %% [markdown]
# ### **XGBoost Regressor**

# %%
from xgboost import XGBRegressor 

xgb = XGBRegressor()
xgb.fit(X_train, y_train)

# %%
xgb_pred = xgb.predict(X_test)

# %%
print('Training Score of XGBRegressor is: {0:.2f}%'.format(xgb.score(X_train, y_train)*100))
print('Testing Score of XGBRegressor is: {0:.2f}%'.format(xgb.score(X_test, y_test)*100))

# %%
xgb_accuracy = metrics.r2_score(y_test, xgb_pred)*100
xgb_accuracy

# %%
print('Mean Squared Error is: {}'.format(metrics.mean_squared_error(y_test, xgb_pred)))

# %%
xgb_df = pd.DataFrame({'Algorithm': ['XGBRegressor'], 'Accuracy': [xgb_accuracy]}, index={'13'})
table = pd.concat([table, xgb_df])
table = table[['Algorithm', 'Accuracy']]
table

# %% [markdown]
# ### **DecisionTree Regressor**

# %%
from sklearn.tree import DecisionTreeRegressor 

dtr = DecisionTreeRegressor()
dtr.fit(X_train, y_train)

# %%
dtr_pred = dtr.predict(X_test)

# %%
print('Training score of DecisionTreeRegressor is: {0:.2f}%'.format(dtr.score(X_train, y_train)*100))
print('Testing score of DecisionTreeRegressor is: {0:.2f}%'.format(dtr.score(X_test, y_test)*100))

# %%
dtr_accuracy = metrics.r2_score(y_test, dtr_pred)*100
dtr_accuracy

# %%
print('Mean Squared Error is: {}'.format(metrics.mean_squared_error(y_test, dtr_pred)))

# %%
dtr_df = pd.DataFrame({'Algorithm': ['DecisionTreeRegressor'], 'Accuracy': [dtr_accuracy]}, index={'14'})
table = pd.concat([table, dtr_df])
table = table[['Algorithm', 'Accuracy']]
table

# %%
dc = DecisionTreeRegressor()
dc.fit(X_train, y_train)

# %%
# printing the feature importance
print('Important Features are: \n', pd.DataFrame(dc.feature_importances_, columns=['Importance'], index=X_train.columns))

# %% [markdown]
# - The importance features are helping to predict strength :- cement, age, water, blast_furnace_slag

# %%
# KFold Cross_validation
K = 20

k_fold = KFold(n_splits=K, random_state=100, shuffle=True)
cv = cross_val_score(dc, X, y, cv=k_fold)
dc_accuracy = np.mean(abs(cv))*100

# %%
cv

# %%
dc_accuracy

# %%
dc_df = pd.DataFrame({'Algorithm': ['DecisionTreeRegressor KFold'], 'Accuracy': [dc_accuracy]}, index={'15'})
table = pd.concat([table, dc_df])
table = table[['Algorithm', 'Accuracy']]
table

# %%
# Select the important features for that create a copy of original data
df = data.copy()

# %%
# drop the unimportant features
X = df.drop(['fly_ash', 'superplasticizer', 'coarse_aggregate', 'fine_aggregate ', 'concrete_compressive_strength'], axis=1)
y = df['concrete_compressive_strength']

# %%
# split the X and y for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# %%
# scale the data
X_train = X_train.apply(zscore)

# %%
X_test = X_test.apply(zscore)

# %%
# now create decision tree model with important features
new_dt = DecisionTreeRegressor()
new_dt.fit(X_train, y_train)

# %%
# after check the feature importance again.
print('Important Features are: \n', pd.DataFrame(new_dt.feature_importances_, columns=['Importance'], index=X_train.columns))

# %%
new_dt_pred = new_dt.predict(X_test)

# %%
print('Training score of DecisionTreeRegressor_feature is: {0:.2f}%'.format(new_dt.score(X_train, y_train)*100))
print('Testing score of DecisionTreeRegressor_feature is: {0:.2f}%'.format(new_dt.score(X_test, y_test)*100))

# %% [markdown]
# Here we facing over fitting, The training and testing score between huge gap.

# %%
new_accuracy = metrics.r2_score(y_test, new_dt_pred)*100
new_accuracy

# %%
dc_df = pd.DataFrame({'Algorithm': ['DecisionTreeRegressor_feature'], 'Accuracy': [new_accuracy]}, index={'16'})
table = pd.concat([table, dc_df])
table = table[['Algorithm', 'Accuracy']]
table

# %%
data.columns

# %%
X = data.drop('concrete_compressive_strength', axis=1)
y = data['concrete_compressive_strength']

# %%
X_scaled = X.apply(zscore)
X_scaled_df = pd.DataFrame(X_scaled, columns=data.columns)

# %%
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42 )

# %%
dc_hype = DecisionTreeRegressor(max_depth=4, random_state=42, min_samples_leaf=5)
dc_hype.fit(X_train, y_train)

# %%
print('Important Features are: \n', pd.DataFrame(dc_hype.feature_importances_, columns=['Importance'], index=X_train.columns))

# %%
X_scaled_df = X_scaled_df.drop('concrete_compressive_strength', axis=1)
feature_cols = X_scaled_df.columns

# %%
feature_cols

# %%
from sklearn import tree

# %%
text_representation = tree.export_text(dc_hype)
print(text_representation)

# %%
# to save this on a file
with open("decision_tree.log", "w") as fout:
    fout.write(text_representation)

# %%
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(dc_hype, feature_names=feature_cols,
                   class_names=["0", "1"], filled=True)

# %%
# To save this figure
fig.savefig("decision_tree.png")

# %%
dc_hypepred = dc_hype.predict(X_test)

# %%
# check the score of training and testing
print("Training Score of decision tree hyper is {0:.2f}%".format(dc_hype.score(X_train, y_train)*100))
print("Testing Score of decision tree hyper is {0:.2f}%".format(dc_hype.score(X_test, y_test)*100))

# %%
dchype_accuracy = metrics.r2_score(y_test, dc_hypepred)*100
dchype_accuracy

# %%
print("Mean Squared Error is: {0}".format(metrics.mean_squared_error(y_test, dc_hypepred)))

# %%
dc_df = pd.DataFrame({'Algorithm': ['DecisionTreeRegressor_Hyper'], 'Accuracy': [dchype_accuracy]}, index={'17'})
table = pd.concat([table, dc_df])
table = table[['Algorithm', 'Accuracy']]
table

# %% [markdown]
# - The best performing algorithm is:- XGBRegressor having 93% accuracy.
# - The worst performing algorithm is:- Support Vector Regressor having 64.52% accuracy.
# - Other good performing algorithms are:- AdaBoostRegressor Hyper tuning, Random Forest Regressor, Gradient Boost Regressor, etc...


