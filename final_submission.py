
# coding: utf-8

# # Submission for Mckinsey Data Hackathon 2017

# ** Author : mbenseddik - Paris/France **
# Website : https://www.mohammed-benseddik.info
# Version : 1.0.1

# ### Problem Statement
# * **Mission**
# *You are working with the government to transform your city into a smart city. The vision is to convert it into a digital and intelligent city to improve the efficiency of services for the citizens. One of the problems faced by the government is traffic. You are a data scientist working to manage the traffic of the city better and to provide input on infrastructure planning for the future.*
#  
# *The government wants to implement a robust traffic system for the city by being prepared for traffic peaks. They want to understand the traffic patterns of the four junctions of the city. Traffic patterns on holidays, as well as on various other occasions during the year, differ from normal working days. This is important to take into account for your forecasting.* 
#  
# * **Your task : ** 
# *To predict traffic patterns in each of these four junctions for the next 4 months.*

# ### Data
# *The sensors on each of these junctions were collecting data at different times, hence you will see traffic data from different time periods. To add to the complexity, some of the junctions have provided limited or sparse data requiring thoughtfulness when creating future projections. Depending upon the historical data of 20 months, the government is looking to you to deliver accurate traffic projections for the coming four months. Your algorithm will become the foundation of a larger transformation to make your city smart and intelligent.*
# 
# *The evaluation metric for the competition is RMSE. Public-Private split for the competition is 25:75.*

# ---

# ### Code :

# The problem is a forecasting of time series data. We will discover the data and do some analysis on the available frequencies in the dataframes.

# #### Python Libraries imports :

# In[1]:

# imports and python libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore")
get_ipython().magic('matplotlib inline')


# #### CSV files imports : 

# In[2]:

# reading csv files
df_train = pd.read_csv('data/train_aWnotuB.csv')
df_test = pd.read_csv('data/test_BdBKkAj.csv')

# converting the dates into pd.datetime types
df_train.DateTime = pd.to_datetime(df_train.DateTime)
df_test.DateTime = pd.to_datetime(df_test.DateTime)

# setting the index of our time series
df_train = df_train.set_index('DateTime')
df_test = df_test.set_index('DateTime')


# In[12]:

# Some basic infos on the train Dataframe
df_train.describe()


# In[15]:

# What are the range of dates for each junction time series?
for junc in df_train.Junction.unique():
    print("Range date for junction {} is : begin date = {}, end date = {}".
          format(junc, df_train[df_train.Junction == junc].index.min(),
                 df_train[df_train.Junction == junc].index.max()))


# **Are there any missing date for the hour frequency?
# We want to see how many observations are available for the day on each junction data. Normally, we would have 24 observations per day.**

# In[22]:

for junction in df_train.Junction.unique():
    fig = plt.figure(junc, figsize=(10,2))
    df_train[df_train.Junction == junction].resample("D")["Vehicles"].count().plot()
    plt.title("Existing values for Junction " + str(junction))
    plt.show()


# ** -> No missing observations for each hour, and for each junction. **

# In[16]:

# We plot vehicles observations per junction
for junc in df_train.Junction.unique():
    fig = plt.figure(junc,figsize=(20,3))
    df_train[df_train.Junction == junc].groupby(pd.TimeGrouper('D')).mean().Vehicles.plot()
    plt.title('Vehicles of Junction {}'.format(junc))
plt.show()


# ** -> The three first junctions are on the same frequencies of dates. The last junction (number 4) has less observations : since Jan. 2017. The frequency per hour remains unchanged**

# ### Stationarity Checks for each TS :

# In[48]:

# Source of code : https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/
from statsmodels.tsa.stattools import adfuller
def test_stationarity(idx, timeseries):
    fig = plt.figure(idx, figsize=(20,5))
    
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation for junc number : {}'.format(junc))
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)


# In[49]:

for junc in df_train.Junction.unique():
    test_stationarity(junc,df_train[df_train.Junction == junc].Vehicles)


# ** -> Our TS are note stationary. We should make our TS non-stationary before fitting complex models. **

# ## Forecasting : First Attempt : Simple Linear Regression :

# We will first try a simple linear regression (without regularization) and without kernel. This will allow us to have a first benchmark for the future models.

# In[23]:

# Machine Learning Libraries imports
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import explained_variance_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# We will first add features to our data matrix. As a simple start, we will add a column for business working day (also weekends), and one column for seasons of the year, as there may be some seasonality over the year for each junction series.

# In[24]:

# adding some date columns
df_train["day"] = df_train.index.day
df_train["hour"] = df_train.index.hour
df_train["weekday"] = df_train.index.weekday
df_train["month"] = df_train.index.month

# adding some date columns
df_test["day"] = df_test.index.day
df_test["hour"] = df_test.index.hour
df_test["weekday"] = df_test.index.weekday
df_test["month"] = df_test.index.month


# In[25]:

# This function returns if a day in the
# dataset is a working business day or a
# weekend
def is_working_day(row):
    if row['weekday'] > 5:
        val = 0
    else:
        val = 1
    return val

# This function returns a number for each section
# of the year, as for seasons.
def which_season(row):
    if row['month'] in [12,1,2]:
        return 1
    if row['month'] in [3,4,5]:
        return 2
    if row['month'] in [6,7,8]:
        return 3
    if row['month'] in [9,10,11]:
        return 0


# In[26]:

# We add the new columns to the train and test datasets
df_train['workingday'] = df_train.apply(is_working_day, axis=1)
df_train['season'] = df_train.apply(which_season, axis=1)

df_test['workingday'] = df_test.apply(is_working_day, axis=1)
df_test['season'] = df_test.apply(which_season, axis=1)


# We will also add a column holding a "score" for each hour of the day, symbolizing the mean of vehicle's number for the 4 junctions.

# In[27]:

hour_score = df_train.groupby("hour")["Vehicles"].mean()
df_train["hour_score"] = df_train.hour.map(hour_score)
df_test["hour_score"] = df_test.hour.map(hour_score)


# We now create our models for each junction, and make predictions :

# In[50]:

# We set our models dictionnary for each Junction
models_dict = {}
features = ['workingday', 'season', 'hour_score']

for junc in df_train.Junction.unique():
    print("*************************************")
    print("Junction Model number {}".format(junc))
    
    # Chunking the dataframe on junction number
    df_train_junc = df_train[df_train.Junction == junc]
    
    # Init Linear regression
    reg_junc = LinearRegression()
    
    # Set split date for train/test
    split_date = "2017-03"
    X_train, X_test = df_train_junc[:split_date][features].values, df_train_junc[split_date:][features].values
    y_train, y_test = df_train_junc[:split_date].Vehicles.values, df_train_junc[split_date:].Vehicles.values
    print("X_train shape : {}".format(X_train.shape))
    print("X_test shape : {}".format(X_test.shape))

    # Scaling data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Fitting the model
    reg_junc.fit(X_train,y_train)
    
    # Save the model in the dictionnary
    models_dict[junc] = (scaler,reg_junc)
    
    # Make predictions using the testing set
    y_pred = reg_junc.predict(X_test)

    # Printing the coefficients
    print('Coefficients: \n', reg_junc.coef_)
    # Printing the mean squared error
    print("Mean squared error: %.2f"
          % mean_squared_error(y_test, y_pred))


# In[51]:

# Predictions :
pred_dfs_arry = []

for junc in df_test.Junction.unique():
    # Take only dates for test set
    df_test_junc = df_test[df_test.Junction == junc]
    # Creating the features matrix for test
    X_valid = df_test_junc[features].values
    # Scaling data
    scaler, regr_model = models_dict[junc]
    X_valid = scaler.transform(X_valid)
    # Predicting data
    y_pred_junc = regr_model.predict(X_valid)
    # Adding the predictions to the dictionnary
    pred_dfs_arry.append(pd.DataFrame({'ID': df_test_junc.ID, 'Vehicles':y_pred_junc}))

df_pred_final = pd.concat(pred_dfs_arry)
# Saving the predictions
df_pred_final.to_csv('reg_lin.csv',index=False)


# ** -> Score on public leaderboard : 22.4. Linear regression is a good beginning, let's try other methods. **

# ## Forecasting : Second Attempt : ARIMA Models :

# #### 1) Eliminating Trend and Seasonality

# We will use **log transform** and **Moving average** :

# In[57]:

ts_log_diff_array = []
# Looping over junctions
for junc in df_train.Junction.unique():
    # Getting our TS
    ts = df_train[df_train.Junction == junc].Vehicles
    # Log transformation
    ts_log = np.log(ts)
    # Moving Average
    moving_avg = pd.rolling_mean(ts_log,12)
    ts_log_moving_avg_diff = ts_log - moving_avg
    ts_log_moving_avg_diff.dropna(inplace=True)
    # Differencing and Decomposition
    ts_log_diff = ts_log - ts_log.shift()
    ts_log_diff.dropna(inplace=True)
    ts_log_diff_array.append(ts_log_diff)
    # Stationarity test
    test_stationarity(junc,ts_log_diff)


# #### 2) Forecasting with ARIMA Models :

# We have to choose the **p** and **q** parameters with ACF and PACF methods : 

# In[62]:

#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf

for junc in df_train.Junction.unique():
    ts_log_diff = ts_log_diff_array[junc - 1]

    lag_acf = acf(ts_log_diff, nlags=20)
    lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')
    
    fig = plt.figure(junc,figsize=(16,5))

    #Plot ACF:
    plt.subplot(121)
    plt.plot(lag_acf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(
        y=-1.96 / np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
    plt.axhline(
        y=1.96 / np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
    plt.title('Autocorrelation Function Junc = {}'.format(junc))

    #Plot PACF:
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(
        y=-1.96 / np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
    plt.axhline(
        y=1.96 / np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
    plt.title('Partial Autocorrelation Function Junc = {}'.format(junc))
    plt.tight_layout()


# ## Forecasting : Third Attempt : Facebook Prophet :

# We will use **FbProphet** for this section. *It is based on an additive model where non-linear trends are fit with yearly and weekly seasonality, plus holidays. It works best with daily periodicity data with at least one year of historical data. Prophet is robust to missing data, shifts in the trend, and large outliers.*. Refers to : https://github.com/facebook/prophet.

# In[31]:

# Python Import 
from fbprophet import Prophet


# We first create a column for the **log** of the vehicles number, to avoid the **non-stationarity** of our time series.

# In[33]:

df_train['Vehicles_log'] = np.log(df_train['Vehicles'])


# We still split our forecasting into 4 models in order to handle junctions. We will prefer logistic predictions to have a more realistic shape of forecasts. 
# 
# We let the seasonality effects on "auto", thus we set the changepoint_prior_scale on 0.01 to be more flexible with hour daily data.
# 
# 
# - *changepoint_prior_scale* : parameter modulating the flexibility of the automatic changepoint selection. Large values will allow many changepoints, small values will allow few changepoints.

# In[41]:

# Setting up our array of prediction :
preds_array = []

# Looping on junctions to have a model per junction
for junc in df_train.Junction.unique():
    # We split treatments : The first 3 junctions are similar on date range. 
    # We will use logarithmique values of vehicle number.
    if junc in (1, 2, 3):
        df = pd.DataFrame({
            'ds': df_train[df_train.Junction == junc].index,
            'y': df_train[df_train.Junction == junc].Vehicles_log
        })
        # Capping the scale of log(Vehicles) for outliers to 5.0
        df['cap'] = 5.0
    else:
        # Here for the junc == 4, we use non logarithmic values
        # to better fit the existing data.
        df = pd.DataFrame({
            'ds': df_train[df_train.Junction == junc].index,
            'y': df_train[df_train.Junction == junc].Vehicles
        })
        # Capping the scale of (Vehicles) for outliers to 15
        df['cap'] = 15
    # Prophet Call
    m = Prophet(growth='logistic', changepoint_prior_scale=0.01)
    m.fit(df)
    # Defining Future Dataframe
    future = pd.DataFrame({'ds': df_test[df_test.Junction == junc].index})
    # Same cap for future
    if junc == 4:
        future['cap'] = 15
    else:
        future['cap'] = 5.0
    # Forecast / prediction
    forecast = m.predict(future)
    # we will plot the predictions
    fig = plt.figure(junc, figsize=(18,4));
    m.plot(forecast);
    plt.title("Forecast Prophet for junction number : {}".format(junc))
    # Constructing the predictions Dataframe and appending it to the array of
    # predictions
    if junc in (1, 2, 3):
        preds_array.append(
            pd.DataFrame({
                'ID': df_test[df_test.Junction == junc].ID.values,
                'Vehicles': np.exp(forecast.yhat)
            }))
    else:
        preds_array.append(
            pd.DataFrame({
                'ID': df_test[df_test.Junction == junc].ID.values,
                'Vehicles': forecast.yhat
            }))

# Saving predictions into CSV file
pd.concat(preds_array).to_csv('prophet.csv')


# ** -> Score on public leaderboard : 7.8701332830, and This is my final submission **.

# # End
