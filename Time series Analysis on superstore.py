#!/usr/bin/env python
# coding: utf-8

# In[72]:


#Importing libraris
import numpy as np 
import pandas as pd
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize']= 10,6
from statsmodels.tsa.arima_model import ARIMA
from numpy import log
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


# In[76]:


data = pd.read_csv(r'C:\Users\Pratik G Ratnaparkhi\Desktop\IVY Python\03Case2\superstore.csv')


# In[77]:


#Lets do some discripitive analysis
data.head()


# In[78]:


data.info()


# In[79]:


#Lets do Time series Analysis 
data1 = data.rename(columns={'Order Date': 'Date'})
data1


# In[80]:


#Changing dtype of date
data1['Date'] = pd.to_datetime(data1['Date'],infer_datetime_format=True)


# In[81]:


data1.head()


# In[82]:


data2 = data1.set_index(['Date'])
data2.head()


# In[83]:


#As ARIMA model is best with long term Analysis 
#We will Convert the date into monthly basis
data3 = data2.resample('MS').mean()
data3.head(12)


# In[84]:


print(data3.info())

print(data3.describe())

print(data3.shape)


# In[85]:


#Lets Explore the Dataset
plt.xlabel('Date')
plt.ylabel('Sales')
plt.plot(data3)


# In[86]:


#Lets decompose the Data:-Trend , Seasonal,and Irregular Component
rcParams['figure.figsize']=8,8
decomposition = sm.tsa.seasonal_decompose(data3,model='additive')
fig = decomposition.plot()
plt.show()


# In[87]:


#plotting the data
def test_stationarity(sales_data):
    roll_mean = sales_data.rolling(window=12).mean()
    roll_stdv = sales_data.rolling(window=12).std()

    from statsmodels.tsa.stattools import adfuller
    sales = plt.plot(sales_data,color='orange',label='Sale')
    mean = plt.plot(roll_mean , color = 'blue' , label='mean')
    std = plt.plot(roll_stdv, color = 'black' , label='stdv')
    plt.legend(loc='upper left')
    plt.title('Rolling Mean And Std Deviation')
    plt.show(block=False)

    test = adfuller(sales_data['Sales'],autolag='AIC')
    testop = pd.Series(test[0:4],index=['Test Statistics','p Value','Lags','Observations'])
    for key , value in test[4].items():
        testop['critical value (%s)'%key] = value
    print(testop)


# In[88]:


test_stationarity(data3)


# In[ ]:





# In[89]:


#lets check the sationarity by diffrencing
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf

data4 = data3.reset_index()

fig, axes = plt.subplots(3,2,sharex=True)
axes[0,0].plot(data4.Sales);axes[0,0].set_title('Original series')
plot_acf(data4.Sales,ax=axes[0,1])

axes[1,0].plot(data4.Sales.diff());axes[1,0].set_title('1st order')
plot_acf(data4.Sales.diff().dropna(),ax=axes[1,1])

axes[2,0].plot(data4.Sales.diff().diff());axes[2,0].set_title('2nd order')
plot_acf(data4.Sales.diff().diff().dropna(),ax=axes[2,1])

plt.show()


# In[90]:


#At first order diffrencing we are getting stationarity


# In[91]:


#Calculating the weighted Avg to know the trend which is present
#in time series
roll_mean1 = data3.rolling(window=12).mean()
roll_stdv1 = data3.rolling(window=12).std()
Avg = data3.ewm(halflife=12,min_periods=0,adjust=True).mean()
plt.plot(data3)
plt.plot(Avg,color='red')
plt.plot(roll_mean1,'orange')
plt.plot(roll_stdv1,color='green')


# In[92]:


#getting p and q values by ACF and PACF method
from statsmodels.tsa.stattools import acf,pacf
acf1 = acf(data3,nlags=20)
pacf1 = pacf(data3,nlags=20,method='ols')

plt.subplot(121)
plt.plot(acf1)
plt.axhline(y=0,linestyle='--',color='green')
plt.axhline(y=-1.96/np.sqrt(len(data2)),linestyle='--',color='green')
plt.axhline(y=1.96/np.sqrt(len(data2)),linestyle='--',color='green')
plt.title('Autocorrelation Function')

plt.subplot(122)
plt.plot(pacf1)
plt.axhline(y=0,linestyle='--',color='green')
plt.axhline(y=-1.96/np.sqrt(len(data2)),linestyle='--',color='green')
plt.axhline(y=1.96/np.sqrt(len(data2)),linestyle='--',color='green')
plt.title('Partial Autocorrelation Function')

plt.tight_layout()


# In[135]:


# As our ARIMA model failed we will go ahead with SARIMA model


# In[122]:


model = sm.tsa.statespace.SARIMAX(data3['Sales'],order=(1,0,1),seasonal_order=(1,0,1,12))
result1 = model.fit()


# In[131]:


data3['forecast'] = result1.predict(start=30,end=48,dynamic=True)
data3[['Sales','forecast']].plot(figsize=(12,8))


# In[125]:


from pandas.tseries.offsets import DateOffset
dates = [data3.index[-1]+DateOffset(months=x)for x in range(0,24)]


# In[126]:


datesdf = pd.DataFrame(index=dates[1:],columns=data3.columns)


# In[137]:


datesdff = pd.concat([data3,datesdf])


# In[138]:


datesdf['forecast'] = result1.predict(start=48,end=70,dynamic=True)
datesdf[['Sales','forecast']].plot(figsize=(12,8))


# In[ ]:





# In[ ]:





# In[ ]:




