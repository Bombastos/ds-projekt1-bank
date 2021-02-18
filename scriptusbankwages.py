## US bank wages EDA
''' 
Our stakeholder wants to implement new regulations for the finance sector.
'''
''' 
Now they need quantitative evidence for a wage gap between female and male and non-minorities and minorities.
'''

# importing required liberies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy import stats
import statsmodels.api as sms
import statsmodels.formula.api as smf
import warnings
from sklearn.metrics import mean_absolute_error
from statsmodels.tools.eval_measures import mse, rmse
import pickle
from sklearn.model_selection import train_test_split

# reading data in our dataframe (df)
df = pd.read_csv('data/us_bank_wages.txt', sep='\t')
# changing column names into small letters
for c in df: df.rename(columns={c:c.lower()},inplace=True)
df.head()

## Getting to know the data
df.info()
df.columns
'''dropping out the unnamed column, since it does not bring us any value'''
df = df.drop(index=0)
df = df.drop(['unnamed: 0'],axis = 1)
df.reset_index(drop = True)

df.head(3)

'''
# Hypothesis: 
1.) We expect positive estimators for a)education, b)gender and c)salbegin. 
* better education leads to higher salary
* higher first salary (salbegin) leads to higher salary
* we expect higher incomes for males than females

2.) We should have a negative estimator for the minority feature
* we expect higher income for non-minorities than minorities

 '''

'''
With the natural logarithm we get rid of outliers and right-taleness
'''
df['lsalary'] = np.log(df['salary'])
df['lsalbegin'] = np.log(df['salbegin'])
df.head(3)

'''
# Addressing the dummy variable trap. 
This is useful for the linear regression later and for the correlation matrix below
'''
jc_dummies = pd.get_dummies(df['jobcat'], prefix='jobcat', drop_first=True)
df_d = df.drop(['jobcat'], axis=1)

df_d = pd.concat([df_d, jc_dummies], axis=1)
df_d.head(3)


'''
## Linear Regression Model
Now we want to derive a good linear regression model via OLS.
'''
X1 = df_d[['educ','lsalbegin', 'gender', 'minority', 'jobcat_2',
       'jobcat_3']]

X1 = sms.add_constant(X1)
y1 = df_d.lsalary

model = sms.OLS(y1, X1)
results = model.fit()
results.summary()

'''
#model with interaction term
'''
model = smf.ols('lsalary ~  educ+ C(jobcat) + gender + lsalbegin+ minority+ +C(jobcat)*minority',data=df).fit()
model.summary()

'''
Linear regression results before train_test_split:
In the first model summary, we can see that all features have a p-value less than 0.05 and hence are statistically significant.
In the second model summary, I have included a cross variable term between jobcat and minority.
However, the interaction of the jobcat dummies and minority had a p-value above 0.05.

After all, the combination of AIC, BIV and R^2 imply the following as the the best model:

$$ log(salary) = 4.123	 + 0.603*x_{lsalbegin} + 0.025*x_{educ} + 0.059*x_{male} - 0.0431*x_{minority} + 0.239*x_{management} + 0.129*x_{custodial}$$ 

Conclusion: We can know reject all our null hypothesis from the hypothesis 1) and 2), i.e. our assumptions are supported by the data after the OLS method
'''

'''
Now  we turn our focus on the train test split method, where we split the data into train and test, in order to train and then test the model
# Model 1:
'''
features = df_d[['educ','lsalbegin', 'gender', 'minority', 'jobcat_2',
       'jobcat_3']]
target = df.lsalary
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.3, random_state = 465)


'''
# Lets train model 1 with 70% of the dataset
'''
X_train = sms.add_constant(X_train)
results = sms.OLS(y_train, X_train).fit()
results.summary()

'''
# useful dict for figure illustration
'''
font = {'family': 'Arial',
        'color':  'darkred',
        'weight': 'normal',
        'size': 14,
        }
'''
# now lets include the test dataset and make a prediction with it on our trained model 1
'''
X_test = sms.add_constant(X_test)
y_preds = results.predict(X_test)


print("Root Mean Squared Error Model 1(RMSE) : {}".format(rmse(y_test, y_preds)))

'''
# Now we take the model where we drop minority feature
# Model 2:
'''
features2 = df_d[['educ','lsalbegin', 'gender', 'jobcat_2',
       'jobcat_3']]
target = df.lsalary
X2_train, X2_test, y2_train, y2_test = train_test_split(features2, target, test_size = 0.3, random_state = 465)

# We train model 2
X2_train = sms.add_constant(X2_train)
results2 = sms.OLS(y2_train, X2_train).fit()
results2.summary()
# We test and make prediction on model 2
X2_test = sms.add_constant(X2_test)
y2_preds = results2.predict(X2_test)


'''
We should compare the root mean squared errors to make a decision.
'''
print("Root Mean Squared Error Model 1(RMSE) : {}".format(rmse(y_test, y_preds)))
print("Root Mean Squared Error Model 2(RMSE) : {}".format(rmse(y2_test, y2_preds)))
'''
Model 1 performs better according to RMSE
'''
# save the model 1 to disk
filename = 'Projekt1_LRmodel.sav'
pickle.dump(results, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.params(X_test, y_test)
print(result)
'''
## Conclusion:
Our train test method suggest that the best model is Model 1, which is the following:
$$ log(salary) = 3.823	 + 0.636*x_{lsalbegin} + 0.023*x_{educ} + 0.061*x_{gender} -0.042*x_{minority} + 0.061*x_{custodial} + 0.19*x_{management} $$ 

The estimators for gender and minority have the expected sign. 

The interpretation goes as follows: a male has, holding other variables constant, a 6,1 % higher salary than a female.
And minorities have about 4,2 % lower salary than non-minorities (holding all other variables constant).

Education and salbegin also have positive impact on salary. One additional year of education leads to 2,3% higher salary.

However for our test statistics we need the assumption of a normal distribution which are not given, although we logarithmized salbegin and salary.
Therefore we should be cautious with our results.
We could also suffer from the omitted variable bias since we don't have variables like experience in our model. 
'''
