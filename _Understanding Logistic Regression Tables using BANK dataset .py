#!/usr/bin/env python
# coding: utf-8

# # # Understanding Logistic Regression Tables using BANK dataset 
More information about the dataset: 
Note that 
1)interest rate indicates the 3-month interest rate between banks and 
2) duration indicates the time since the last contact was made with a given consumer. 
3)The previous variable shows whether the last marketing campaign was successful with this customer.
4)The March and  May  are Boolean variables that account for when the call was made to the specific customer and 
5)credit shows if the customer has enough credit to avoid defaulting.

 Notes: about the data set 
    1) the first column of the dataset is an index one; 
    2) you don't need the graph for this exercise; 
    3) the dataset used is much bigger 
# ## Import the relevant libraries

# In[1]:


import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)


# ## Load the data

# In[3]:


raw_data = pd.read_csv(r'C:\Users\rawad\OneDrive\Desktop\aws Restart course\Udemy Data Science Course\exercise\Bank_data.csv')
raw_data


# In[4]:


# We make sure to create a copy of the data before we start altering it. Note that we don't change the original data we loaded.
data = raw_data.copy()

# Removes the index column thata comes with the data
data = data.drop(['Unnamed: 0'], axis = 1)

# We use the map function to change any 'yes' values to 1 and 'no'values to 0. 
data['y'] = data['y'].map({'yes':1, 'no':0})
data


# In[5]:


data.describe()


# ### Declare the dependent and independent variables

# In[6]:


y = data['y']
x1 = data['duration']


# ### Simple Logistic Regression

# In[ ]:





# In[7]:


x = sm.add_constant(x1)
reg_log = sm.Logit(y,x)
results_log = reg_log.fit()


# # Interpretation

# In[8]:


results_log.summary()

Based on the logistic regression analysis we just implemented , we can draw the following conclusions:

1) Dependent variable: The dependent variable in this analysis is 'duration'. It appears to be significant, with a coefficient 		of 0.0051.
2) Model and method: The model used is a logistic regression, which is appropriate for predicting binary outcomes. The method 	 used to estimate the model is Maximum Likelihood Estimation (MLE), which is a commonly used technique for estimating 		    logistic regression models.

3) Convergence: The fact that the model has converged is a good sign that the estimation process has been successful and that      the results are reliable.

4) Pseudo R-squared: The Pseudo R-squared value of 0.21 is within an acceptable range, indicating that the model is explaining    a reasonable proportion of the variance in the data.

5) Constant: The constant (intercept) in the model is significant and has a value of -1.70. This suggests that, even when the       value of the 'duration' variable is zero, there is still a significant negative impact on the outcome.
Overall, the results of the logistic regression analysis suggest that there is a significant relationship between the 'duration' variable and the outcome, and that the model is reasonably effective at predicting the outcome. However, it is important to note that the analysis only provides a limited view of the relationship between variables, and other factors may be relevant that are not included in the model. Further analysis and testing may be necessary to fully understand the relationship between the variables and the outcome.
# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




