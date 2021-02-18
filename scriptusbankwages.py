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

Thus, for educ, salbin and gender we have the following hypothesis:

 $  H_0: \beta_i = 0 $ vs. $ H_1: \beta_i > 0 $ , where i = educ, salbegin or gender 

2.) We should have a negative estimator for the minority feature
* we expect higher income for non-minorities than minorities

For the minority feature we have the following hypothesis:
 $  H_0: \beta_{minority} = 0 $ vs. $ H_1: \beta_{minority} < 0 $ 
 '''

 '''
 # Visualisation
Now lets see whether our hypothesis can be supported or rejected via visualisation.
'''
sns.pairplot(vars=['salary','educ', 'salbegin', 'gender', 'minority',
       'jobcat'],data=df);

'''
What we learn from the pairplot:
* The salary row of our pairplot indicates, that education and salbegin have a positive impact on salary. 
* Moreover we have 3 categorical variables/dummys, where females and minorities earn less, and jobcatagory management positions earn more than the other two jobcats.
* we dont have a normal distribution for salbegin, which could lead to problems in our t-test later.

From the pairplot we saw that the distribution of salbegin is right-taled skewed.
This could imply that salbegin has a non-linear relation to salary. 
Via the natural log, I try to address the problem.
'''
df['lsalary'] = np.log(df['salary'])
df['lsalbegin'] = np.log(df['salbegin'])
df.head(3)

'''
As you can see below, we can decrease the skewness of salbegin and salary via the natural log.
'''
'''
# lsalbegin is the natural log of salbegin and lsalary is the natural log of salary.
'''
warnings.filterwarnings('ignore')
fig = plt.figure(figsize = (8,8))
ax = fig.gca()
df.hist(ax = ax);

'''
Via the natural logarithm, we also address the outlier problem. 
In the following two boxplots you can see how a log can decrease the number of outliers in salbegin.
'''
px.box(df,x='salbegin',template='plotly_dark',points='all'
    ,title="salbegin boxplot and outliers")
px.box(df,x='lsalbegin',template='plotly_dark',points='all'
    ,title="log(salbegin) boxplot and outliers")
'''
The 3 Boxplots below show the differences inside our categories:
* A difference between whites and minorities is hard to see. Hypothesis from 2) is hard to reject.
* the second boxplot confirms a gender gap: females earn less, which supports $$ \beta_{gender} > 0 $$
* the third boxplot shows that those in management positions earn more than those in custodial or administrative positions
'''

'''
# Boxplot 1: non-minority vs. minority
'''
plt.figure(figsize = (10, 6))
ax = sns.boxplot(x='minority', y='salary', data=df)
plt.setp(ax.artists, alpha=.5, linewidth=2, edgecolor="k")
plt.xticks()
plt.title('Boxplot 1: minority')
plt.legend(["non-minority(0)", "minority(1)"])
ax.grid(True);

'''
# Boxplot 2: females(=0) vs. and males(=1)
'''
plt.figure(figsize = (10, 6))
ax = sns.boxplot(x='gender', y='salary',palette="Set2",data=df)
plt.setp(ax.artists, alpha=.5, linewidth=2, edgecolor="k")
plt.xticks(rotation=0)
plt.title('Boxplot 2: females(=0) vs. and males(=1)')
plt.legend(["female", "male"],shadow=True)
ax.grid(True);

'''
# boxplot 3 for jobcatagories
'''
plt.figure(figsize = (10, 6))
ax = sns.boxplot(x='jobcat', y='salary', data=df)
plt.setp(ax.artists, alpha=.5, linewidth=2, edgecolor="k")
plt.xticks(rotation=45)
plt.legend(["administrative(1)", "custodial(2)", "management(3)"],title='Positions',shadow=True)
plt.title('Boxplot 3: job catagories')
ax.grid(True);

'''
# Addressing the dummy variable trap. 
This is useful for the linear regression later and for the correlation matrix below
'''
jc_dummies = pd.get_dummies(df['jobcat'], prefix='jobcat', drop_first=True)
df_d = df.drop(['jobcat'], axis=1)

df_d = pd.concat([df_d, jc_dummies], axis=1)
df_d.head(3)

'''
# heatmap for the correlation matrix
'''
corr=df_d.corr()
plt.figure(figsize=(12, 10))

sns.heatmap(corr, 
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True);

'''
The heatmap above gives us many insightful information. But we concentrate on the lsalary row. 
There we see that our null hypothesis in 1a),1b) and 1c) can be rejected,
i.e. educ, salbegin/lsalbegin and gender have a positive impact on lsalary.
'''

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
print('Training Data Count: {}'.format(X_train.shape[0]))

print('Testing Data Count: {}'.format(X_test.shape[0]))

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

plt.figure(dpi = 75)
plt.scatter(y_test, y_preds)
plt.plot(y_test, y_test, color="red")
plt.xlabel("Actual lsalary", fontdict=font)
plt.ylabel("Estimated lsalary", fontdict=font)
plt.legend('pa')
plt.title("Model: Actual vs Estimated lsalary", fontdict=font)
plt.show()

print("Root Mean Squared Error (RMSE) : {}".format(rmse(y_test, y_preds)))

'''
# Now we take the model where we drop minority feature
# Model 2:
'''
features2 = df_d[['educ','lsalbegin', 'gender', 'jobcat_2',
       'jobcat_3']]
target = df.lsalary
X2_train, X2_test, y2_train, y2_test = train_test_split(features2, target, test_size = 0.3, random_state = 465)
print('Training Data Count: {}'.format(X2_train.shape[0]))
print('Testing Data Count: {}'.format(X2_test.shape[0]))

# We train model 2
X2_train = sms.add_constant(X2_train)
results2 = sms.OLS(y2_train, X2_train).fit()
results2.summary()
# We test and make prediction on model 2
X2_test = sms.add_constant(X2_test)
y2_preds = results2.predict(X2_test)

plt.figure(dpi = 75)
plt.scatter(y2_test, y2_preds)
plt.plot(y2_test, y2_test, color="red")
plt.xlabel("Actual lsalary", fontdict=font)
plt.ylabel("Estimated lsalary", fontdict=font)
plt.title("Model: Actual vs Estimated lsalary", fontdict=font)
plt.show()

'''
Just by looking at the figures of model 1 and model 2, we can't say much which model is better.
Therefore we should compare the RMSE to make a decision.
'''
print("Root Mean Squared Error Model 1(RMSE) : {}".format(rmse(y_test, y_preds)))
print("Root Mean Squared Error Model 2(RMSE) : {}".format(rmse(y2_test, y2_preds)))
'''
Model 1 performs better according to RMSE
'''
'''
## Conclusion:
Our train test method suggest that the best model is Model 1, which is the following:
$$ log(salary) = 3.823	 + 0.636*x_{lsalbegin} + 0.023*x_{educ} + 0.061*x_{gender} -0.042*x_{minority} + 0.061*x_{custodial} + 0.19*x_{management} $$ 

The estimators of interest for our stakeholders are: $$ \hat{\beta}_{gender}\text{ and }\hat{\beta}_{minority}$$
both estimators have the expected sign. 

The interpretation goes as follows: a male has, holding other variables constant, a 6,1 % higher salary than a female.
And minorities have about 4,2 % lower salary than non-minorities (holding all other variables constant).

Education and salbegin also have positive impact on salary. One additional year of education leads to 2,3% higher salary.

However for our test statistics we need the assumption of a normal distribution which are not given, although we logarithmized salbegin and salary.
Therefore we should be cautious with our results.
We could also suffer from the omitted variable bias since we don't have variables like experience in our model. 
'''
