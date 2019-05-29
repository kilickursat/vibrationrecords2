# -*- coding: utf-8 -*-
"""
Created on Sat May 18 01:06:57 2019

@author: KURSAT
"""



#!/usr/bin/env python
# coding: utf-8

# In[1]:

#get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
df=pd.read_csv('C:/Users/KURSAT/Desktop/Python Program Documents/vibration_records.csv')
#x = np.random.randn(100)
#y = x + np.random.randn(100) + 10
x=df['SD']
y=df['PPV']
# In[2]:

y=np.log10(y)
x=np.log10(x)


import matplotlib.pyplot as plt
# In[3]:


fig, ax = plt.subplots(figsize=(8, 4))

ax.scatter(x, y, alpha=0.5, color='orchid')

fig.suptitle('Example Scatter Plot')

fig.tight_layout(pad=2); 

ax.grid(True)

fig.savefig('filename1.png', dpi=125)

#import matplotlib.pyplot as plt
import statsmodels.api as sm
#%matplotlib notebook


x = sm.add_constant(x) # constant intercept term

# Model: y ~ x + c

model = sm.OLS(y, x)

fitted = model.fit()

x_pred = np.linspace(x.iloc[:,1].min(), x.iloc[:,1].max(), 19)

x_pred2 = sm.add_constant(x_pred)

y_pred = fitted.predict(x_pred2)

ax.plot(x_pred, y_pred, '-', color='darkorchid', linewidth=2)

upperbound_const = fitted.conf_int()[1][0] # Upper bound for 0.95 conf

const_differ = upperbound_const - fitted.params[0]

const_differ_log= np.log(const_differ)

ax.plot(x_pred, y_pred + const_differ_log, '-', color='green', linewidth=2)

plt.xticks([1.4,1.9], (10,100) )
plt.yticks([-0.2,0.2,0.6,1.2], (0.1, 1, 10, 100))

fig.savefig('filename2.png', dpi=125)


# In[4]:


print(const_differ_log)
print(fitted.params)     # the estimated parameters for the regression line
print(fitted.summary())  # summary statistics for the regression

import numpy as np
np.sqrt(fitted.scale)    # Residual Standard Error


# In[ ]:



