#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[28]:


data = pd.read_csv(r'C:\Users\Pratik G Ratnaparkhi\Desktop\IVY Python\03Case2\AirPassengers.csv')


# In[29]:


#Lets do some discripitive analysis
data.head()


# In[30]:


data.info()


# In[31]:


#Lets do Time series Analysis 
data1 = data.rename(columns={'#Passengers': 'Passengers'})
data1


# In[32]:


#Changing dtype of date
data1['Month'] = pd.to_datetime(data1['Month'],infer_datetime_format=True)


# In[33]:


data1.head()


# In[34]:


data2 = data1.set_index(['Month'])
data2.head(30)


# In[35]:


#Lets Explore the Dataset
plt.xlabel('Month')
plt.ylabel('Passengers')
plt.plot(data2)


# In[36]:


#Lets decompose the Data:-Trend , Seasonal,and Irregular Component
rcParams['figure.figsize']=8,8
decomposition = sm.tsa.seasonal_decompose(data2,model='additive')
fig = decomposition.plot()
plt.show()


# In[37]:


#plotting the data
def test_stationarity(pass_data):
    roll_mean = pass_data.rolling(window=12).mean()
    roll_stdv = pass_data.rolling(window=12).std()

    from statsmodels.tsa.stattools import adfuller
    sales = plt.plot(pass_data,color='orange',label='Sale ')
    mean = plt.plot(roll_mean , color = 'blue' , label='mean')
    std = plt.plot(roll_stdv, color = 'black' , label='stdv')
    plt.legend(loc='upper left')
    plt.title('Rolling Mean And Std Deviation')
    plt.show(block=False)

    test = adfuller(pass_data['Passengers'],autolag='AIC')
    testop = pd.Series(test[0:4],index=['Test Statistics','p Value','Lags','Observations'])
    for key , value in test[4].items():
        testop['critical value (%s)'%key] = value
    print(testop)


# In[38]:


#Ho = Non-Stationary(p>=0.05)
#H1 = Stationary
test_stationarity(data2)


# In[39]:


#from the above our avg is changing proportionally with data
#As p value is greater than 0.05 thats why we will accept the Ho
#here we can say that data is non-stationary


# In[40]:


#At first order diffrencing we are getting stationarity


# In[41]:


logdata = np.log(data2)
plt.plot(logdata)
plt.show()


# In[42]:


test_stationarity(logdata)


# In[43]:


logroll_mean = logdata.rolling(window=12).mean()
data4 = logdata-logroll_mean
data4.dropna(inplace=True)
data4.head(12)


# In[44]:


test_stationarity(data4)


# In[45]:


diff = logdata-logdata.shift()
plt.plot(diff)
plt.show()


# In[46]:


diff.dropna(inplace=True)
test_stationarity(diff)


# In[47]:


#Calculating the weighted Avg to know the trend which is present
#in time series
roll_mean1 = diff.rolling(window=12).mean()
roll_stdv1 = diff.rolling(window=12).std()
Avg = diff.ewm(halflife=12,min_periods=0,adjust=True).mean()
plt.plot(diff)
plt.plot(Avg,color='red')
plt.plot(roll_mean1,'orange')
plt.plot(roll_stdv1,color='green')


# In[48]:


#getting p and q values by ACF and PACF method
from statsmodels.tsa.stattools import acf,pacf
acf1 = acf(diff,nlags=20)
pacf1 = pacf(diff,nlags=20,method='ols')

plt.subplot(121)
plt.plot(acf1)
plt.axhline(y=0,linestyle='--',color='green')
plt.axhline(y=-1.96/np.sqrt(len(diff)),linestyle='--',color='green')
plt.axhline(y=1.96/np.sqrt(len(diff)),linestyle='--',color='green')
plt.title('Autocorrelation Function')

plt.subplot(122)
plt.plot(pacf1)
plt.axhline(y=0,linestyle='--',color='green')
plt.axhline(y=-1.96/np.sqrt(len(diff)),linestyle='--',color='green')
plt.axhline(y=1.96/np.sqrt(len(diff)),linestyle='--',color='green')
plt.title('Partial Autocorrelation Function')

plt.tight_layout()


# In[49]:


from statsmodels.tsa.stattools import arma_order_select_ic
arma_order_select_ic(diff)


# In[52]:


#AR model
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(logdata,order=(2,1,0))
results = model.fit(disp=-1)
plt.plot(diff)
plt.plot(results.fittedvalues,color='red')
plt.title('Rss: %.4f'% sum((results.fittedvalues-diff['Passengers'])**2))
plt.show()


# In[53]:


#MA model
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(logdata,order=(0,1,2))
results = model.fit(disp=-1)
plt.plot(diff)
plt.plot(results.fittedvalues,color='red')
plt.title('Rss: %.4f'% sum((results.fittedvalues-diff['Passengers'])**2))
plt.show()


# In[54]:


#ARIMA MODEL
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(logdata,order=(2,1,2))
results = model.fit(disp=-1)
plt.plot(diff)
plt.plot(results.fittedvalues,color='red')
plt.title('Rss: %.4f'% sum((results.fittedvalues-diff['Passengers'])**2))
plt.show()


# In[56]:


#Now lets forecast 
pred = pd.Series(results.fittedvalues,copy=True)
print(pred.head())


# In[57]:


#To get our original data we will use cumsum function
predcs = pred.cumsum()
predcs.head()


# In[60]:


#Again the log data need to be convert in the original form
predlog = pd.Series(logdata['Passengers'].iloc[0],index=logdata.index)
predlog = predlog.add(predcs,fill_value=0)
predlog.head()


# In[61]:


predAR = np.exp(predlog)
plt.plot(data2)
plt.plot(predAR)
plt.show()


# In[62]:


results.plot_predict(1,264)
x=results.forecast(steps=120)


# In[63]:


results.forecast(steps=120)


# In[ ]:




