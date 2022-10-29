# Import the Libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

import yfinance as yf
import datetime as dttm

st.set_page_config(page_title='Stock Market Forecasting', page_icon='stock2.jpg')
st.image('stock2.jpg','Stock Market Forecasting')
st.header('Stock Market Forecasting')
st.write('---')
st.sidebar.header('User Input Parameters')

ipt = st.sidebar.selectbox( label ="Take Input DataSet From",
    options=['Yfinance','File'])
if ipt == 'Yfinance':
    st.sidebar.write('**1).Stock Name**')
    stick_name = st.sidebar.text_input('Enter the Name of Stock as in yfinance',
                                       value= 'NESTLEIND.NS'
                                       )
    st.sidebar.write('**2).Starting Date and Ending Date used for Training Data**')
    startDate = st.sidebar.date_input('Starting Date',
                                       value= dttm.date(2011, 1, 1),
                                       min_value = dttm.date(2010,1,1)
                                       )
    endDate = st.sidebar.date_input('Ending Date', 
                                     value= dttm.date(2022, 7, 1), 
                                     min_value = dttm.date(2010,1,1)
                                     )
    n = st.sidebar.number_input("Approx Number of Day's you want to Forecaste",
                               min_value = 1,
                               max_value = 30,
                               value = 10)
elif ipt == 'File':
    st.sidebar.write('**1).Stock Name**')
    stick_name = st.sidebar.text_input('Enter the Name of Stock', 
                                       value= 'NESTLEIND.NS'
                                       )
    data = st.sidebar.file_uploader('Upload DataSet In "csv" formate', 
                                    type = 'csv'
                                    )
    
    endDate = st.sidebar.date_input('Last Date of your Dataset', 
                                    value= dttm.date(2022, 7, 1), 
                                    min_value = dttm.date(2010,1,1)
                                    )
    
    n = st.sidebar.number_input("Approx Number of Day's you want to Forecaste",
                              min_value = 1,
                              max_value = 30,
                              value = 10
                              )
    
else:
    st.error('You Select a wrong option')

if 'start' not in st.session_state:
    st.session_state['start'] = False
    
start = st.sidebar.checkbox('check to start', value = st.session_state['start'])


if start:
    st.session_state['start'] = True
    
else:
    st.session_state['start'] = False

if 'stock_past' not in st.session_state:
    st.session_state['stock_past'] = None
result = st.sidebar.button('Clear the Session State')
if result:
    for key in st.session_state.keys():
        del st.session_state[key]
    
if 'start' not in st.session_state:
    st.session_state['start'] = False

if st.session_state['start']==True:
    # Extract Dataset from yfinance
    if ipt == 'Yfinance':
        
        GetData = yf.Ticker(stick_name)
        
        yf_data = pd.DataFrame(GetData.history(start=startDate, end=endDate))
        
        st.subheader('Input DataFrame')
        st.write(stick_name,'*Stock DataFrame*')
        st.dataframe(yf_data)
        
        if yf_data.empty:
            st.error('No Internet Connection')
            yf_data = None
    
    # Upload .csv file
    elif ipt == 'File':
        
        if data == None:
            st.error('Please Upload the data')
            st.subheader('We have take Default File for Forecating')
            yf_data = pd.read_excel('NESTLEIND_2011_2022.xlsx',index_col='Date',parse_dates=True)
            
        else:
            yf_data = pd.read_csv(data,index_col='Date',parse_dates=True)
        
            st.subheader('Input DataFrame')
            st.write(stick_name,'*Stock DataFrame*')
            st.dataframe(yf_data)
    
    if 'stock_present' not in st.session_state:
        st.session_state['stock_present'] = stick_name
        
    st.session_state['stock_present'] = stick_name
    
    
    st.subheader('Visualisation')
    
    # Different types of plots
    st.sidebar.header('Visualisation')
    #Visualisation
    chart_select = st.sidebar.selectbox(
        label ="Type of chart",
        options=['Lineplots','Scatterplots','Histogram']
    )
    
    
    numeric_columns = list(yf_data.select_dtypes(['float','int']).columns)
    numeric_columns.sort()
    
    if chart_select == 'Scatterplots':
        st.sidebar.subheader('Scatterplot Settings')
        try:
            y_values = st.sidebar.selectbox('Y axis',options=numeric_columns)
            
            plot = px.scatter(data_frame=yf_data,y=y_values,
                              title=str('Scatter Plot for '+y_values+' column'))
            st.write(plot)
        except Exception as e:
            print(e)
    if chart_select == 'Histogram':
        st.sidebar.subheader('Histogram Settings')
        try:
            x_values = st.sidebar.selectbox('X axis',options=numeric_columns)
            plot = px.histogram(data_frame=yf_data,x=x_values,marginal="box",
                                title=str('Histogram Plot for '+y_values+' column'))
            st.write(plot)
        except Exception as e:
            print(e)
    if chart_select == 'Lineplots':
        st.sidebar.subheader('Lineplots Settings')
        try:
            y_values = st.sidebar.selectbox('Y axis',options=numeric_columns)
            plot = px.line(yf_data,y=y_values,
                           title=str('Line Plot for '+y_values+' column'))
            st.write(plot)
        except Exception as e:
            print(e)
    
    
    # Final Dataset for Model Building i.e. selecting only "close" column
    #st.write(numeric_columns)
    if "Close" in numeric_columns:
        final_data = pd.DataFrame(yf_data.Close)
        final_data = final_data.sort_index(ascending=True)
        final_data.rename(columns={'Close': 'Close'},inplace = True)
        
    elif "close" in numeric_columns:
        final_data = pd.DataFrame(yf_data.close)
        final_data = final_data.sort_index(ascending=True)
        final_data.rename(columns={'close': 'Close'},inplace = True)
        
    elif "CLOSE" in numeric_columns:
        final_data = pd.DataFrame(yf_data.CLOSE)
        final_data = final_data.sort_index(ascending=True)
        final_data.rename(columns={'CLOSE': 'Close'},inplace = True)
        
    else:
        final_data = None
        st.subheader('Close Column is not Present in the File, Please Check the file and reupload')
        
    st.subheader('DataSet used for Training')
    st.write(final_data)
    
    try:
        # Setting Frequency of Close column
        training_data = final_data.copy()
        training_data = training_data.asfreq('B')
        training_data.ffill(inplace=True)
        #st.write(data)
        #st.write(data.shape)
        #st.write(data.isnull().sum())
        
    except:
        training_data = None
        st.error('Please Check the settings, You have not choose the appropriate option or You have not upload the File or not in write formate')
    
    # Error function 
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    
    # Data Transformation=========================================================================================================================
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0.001,1))
    full_data_minmax = scaler.fit_transform(np.array(training_data).reshape(-1,1))
    #st.write(train_data_minmax)
    #st.write(train_data.index)
        
    full_data_minmax = pd.DataFrame(full_data_minmax, columns = ['close'])
    full_data_minmax.index = training_data.index
    #st.write(full_data_minmax)
        
        
    # Spliting the DataSet into Train and Test
    train_data = training_data[:int(len(final_data)*0.8)]
    test_data = training_data[int(len(final_data)*0.8):]
    
    train_data_minmax = scaler.fit_transform(np.array(train_data).reshape(-1,1))
    train_data_minmax = pd.DataFrame(train_data_minmax, columns = ['close'])
    train_data_minmax.index = train_data.index
        
        
    #Model Building===============================================================================================================================
        
    from sktime.forecasting.compose import AutoEnsembleForecaster
    from sktime.forecasting.exp_smoothing import ExponentialSmoothing
    from sktime.forecasting.fbprophet import Prophet
    import holidays
    import random
        
    # Prophet Model-------------------------------------------------------------------------------------------------------------------------------
        
    # Holiday
    holiday = pd.DataFrame([])
        
    for date, name in sorted(holidays.India(years=[2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022]).items()):
        holiday = holiday.append(pd.DataFrame({'ds': date, 'holiday': "India_Holidays"}, index=[0]), ignore_index=True)
    holiday['ds'] = pd.to_datetime(holiday['ds'], format='%Y-%m-%d', errors='ignore')
        
    # HyperParameter Tunning
    if st.session_state['stock_present']!= st.session_state['stock_past']:
            
        st.write('**Hypreparameter Tunning is Started for Prophet Model**')
        from sklearn.model_selection import ParameterGrid
        params_grid = {'changepoint_prior_scale':[0.05,1,10,25],
                           'n_changepoints' : [1,10,25,100],
                           'seasonality_prior_scale':[0.05,1,10,25]}
        Pro_model_parameters = pd.DataFrame(columns = ['Parameters','MSE','RMSE'])
            
        grid = ParameterGrid(params_grid)
        
        Pro_bar = st.progress(0)
        
        i = 1
        for p in grid:
            test = pd.DataFrame()
        #    print(i,' ',p)
            Pro_bar.progress(i/len(grid))
            i = i+1
            random.seed(0)
            train_model =Prophet(freq='B', 
                                 changepoint_prior_scale = p['changepoint_prior_scale'],
                                 n_changepoints = p['n_changepoints'],
                                 seasonality_mode = 'multiplicative',
                                 seasonality_prior_scale=p['seasonality_prior_scale'],
                                 weekly_seasonality=False,
                                 daily_seasonality = False,
                                 yearly_seasonality = True,
                                 add_country_holidays={'country_name': 'India'}, 
                                 holidays=holiday)
            train_model.fit(train_data_minmax)
            fh = list(range(1,int(len(test_data))+1))
            test_predictions = train_model.predict(fh=fh)
            test_predictions=scaler.inverse_transform(test_predictions)
            mse = mean_squared_error(test_data, test_predictions)
            rmse = np.sqrt(mse)
            #print('Root Mean Squre Error(RMSE)------------------------------------',rmse)
            Pro_model_parameters = Pro_model_parameters.append({'Parameters':p, 'MSE':mse, 'RMSE':rmse},ignore_index=True)
            
            
        Pro_parameters = Pro_model_parameters.sort_values(by=['RMSE'])
        Pro_parameters = Pro_parameters.reset_index(drop=True)
        #st.write(Pro_parameters)
        st.write('**Hypreparameter Tunning is Done for Prophet Model**')
            
        if 'changepoint_prior_scale' not in st.session_state:
            st.session_state['changepoint_prior_scale'] = Pro_parameters['Parameters'][0]['changepoint_prior_scale']
            
        else:
            pass
        st.session_state['changepoint_prior_scale'] = Pro_parameters['Parameters'][0]['changepoint_prior_scale']
            
        if 'n_changepoints' not in st.session_state:
            st.session_state['n_changepoints'] = Pro_parameters['Parameters'][0]['n_changepoints']
                
        else:
            pass
        st.session_state['n_changepoints'] = Pro_parameters['Parameters'][0]['n_changepoints']
            
        if 'seasonality_prior_scale' not in st.session_state:
            st.session_state['seasonality_prior_scale'] = Pro_parameters['Parameters'][0]['seasonality_prior_scale']
                
        else:
            pass
        st.session_state['seasonality_prior_scale'] = Pro_parameters['Parameters'][0]['seasonality_prior_scale']
        
    else:
        pass
        
    Pro_model = Prophet(freq='B', seasonality_mode='multiplicative', 
                        changepoint_prior_scale=st.session_state['changepoint_prior_scale'], 
                        n_changepoints=st.session_state['n_changepoints'], 
                        seasonality_prior_scale=st.session_state['seasonality_prior_scale'], 
                        add_country_holidays={'country_name': 'India'}, verbose=10,
                        holidays=holiday,
                        yearly_seasonality=True, weekly_seasonality=False , daily_seasonality=False)
    #Pro_model.fit(train_data_minmax)
    
    #fh = list(range(1,int(len(test_data_minmax))+1))
    # fh1 = pd.DatetimeIndex(np.array(test_data.index))
    # fh1
    #test_predictions_minmax = Pro_model.predict(fh=fh)
    #st.write(test_predictions_minmax)
        
    #test_predictions=scaler.inverse_transform(test_predictions_minmax)
    #test_predictions = pd.DataFrame(test_predictions, columns = ['Close'])
    #test_predictions.index = test_data.index
    #st.write(test_predictions)
        
        
        
    # Exponential Smoothing Model-----------------------------------------------------------------------------------------------------------------
        
        
    # HyperParameter Tunning
    if st.session_state['stock_present']!= st.session_state['stock_past']:
            
        st.write('**Hypreparameter Tunning is Started for Exponential Smoothing Model**')
        from sklearn.model_selection import ParameterGrid
        params_grid = {'trend':["add", "mul"],
                       'seasonal' : ["add", "mul"]
                       }
        Expo_model_parameters = pd.DataFrame(columns = ['Parameters','MSE','RMSE'])
            
        grid = ParameterGrid(params_grid)
        
        Expo_bar = st.progress(0)
        i = 1
        for p in grid:
            test = pd.DataFrame()
        #    print(i,' ',p)
            Expo_bar.progress(i/len(grid))
            i = i+1
            random.seed(0)
            train_model = ExponentialSmoothing(trend=p['trend'],
                                               seasonal=p['seasonal'],
                                               sp=262,
                                               damped_trend=False)
            train_model.fit(train_data_minmax)
            fh = list(range(1,int(len(test_data))+1))
            test_predictions = train_model.predict(fh=fh)
            test_predictions=scaler.inverse_transform(test_predictions)
            mse = mean_squared_error(test_data, test_predictions)
            rmse = np.sqrt(mse)
            #    print('Root Mean Squre Error(RMSE)------------------------------------',rmse)
            Expo_model_parameters = Expo_model_parameters.append({'Parameters':p, 'MSE':mse, 'RMSE':rmse},ignore_index=True)
        
        
        Expo_parameters = Expo_model_parameters.sort_values(by=['RMSE'])
        Expo_parameters = Expo_parameters.reset_index(drop=True)
        #st.write(Expo_parameters)
        st.write('**Hypreparameter Tunning is Done for Exponential Smoothing Model**')
        
        if 'trend' not in st.session_state:
            st.session_state['trend'] = Expo_parameters['Parameters'][0]['trend']
            
        else:
            pass
        st.session_state['trend'] = Expo_parameters['Parameters'][0]['trend']
        
        if 'seasonal' not in st.session_state:
            st.session_state['seasonal'] = Expo_parameters['Parameters'][0]['seasonal']
            
        else:
            pass
        st.session_state['seasonal'] = Expo_parameters['Parameters'][0]['seasonal']
    
    else:
        pass
        
    Expo_model = ExponentialSmoothing(trend=st.session_state['trend'],
                                      seasonal=st.session_state['seasonal'],
                                      sp=262,
                                      damped_trend=False)
    #Expo_model.fit(train_data_minmax)
        
    #fh = list(range(1,int(len(test_data_minmax))+1))
    # fh1 = pd.DatetimeIndex(np.array(test_data.index))
    # fh1
    #test_predictions_minmax = Expo_model.predict(fh=fh)
    #st.write(test_predictions_minmax)
    
    #test_predictions=scaler.inverse_transform(test_predictions_minmax)
    #test_predictions = pd.DataFrame(test_predictions, columns = ['Close'])
    #test_predictions.index = test_data.index
    #st.write(test_predictions)
        
        
    # AutoEnsembleForecaster Model----------------------------------------------------------------------------------------------------------------
        
    st.subheader('Model Building')
    st.write('**Validating the final model**')
    forecasters = [
         ("prophet" , Pro_model),
         ("expo" , Expo_model)
        ]
        
    Ensmodel = AutoEnsembleForecaster(forecasters=forecasters, n_jobs=-1, random_state=42)
    Ensmodel.fit(train_data_minmax)
        
        
    fh = list(range(1,int(len(test_data))+1))
    # fh1 = pd.DatetimeIndex(np.array(test_data.index))
    # fh1
    test_predictionsEns = Ensmodel.predict(fh=fh)
    #st.write(test_predictionsEns)
        
        
    test_predictions=scaler.inverse_transform(test_predictionsEns)
    test_predictions = pd.DataFrame(test_predictions, columns = ['Close'])
    test_predictions.index = test_data.index
    #st.write(test_predictions)
    
    mse = mean_squared_error(test_data, test_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_data, test_predictions)
    mape = np.mean(np.abs((test_data-test_predictions)/test_data))*100
    errors = {'MSE':mse, 'RMSE':rmse, 'MAE':mae, 'MAPE':mape}
    errors_df = pd.DataFrame(errors)
        
        
    fig = go.Figure()
       
    fig.add_trace(go.Scatter(x=train_data.index, y=train_data['Close'], mode='lines', name='TRAIN'))
    fig.add_trace(go.Scatter(x=test_data.index, y=test_data['Close'], mode='lines', name='TEST'))
    fig.add_trace(go.Scatter(x=test_predictions.index, y=test_predictions['Close'], mode='lines', name='PREDICTION'))
        
    fig.update_layout(title_text='Forecast vs Actuals', title_x=0.5)
        
    st.plotly_chart(fig)
    
    st.write(errors_df)
        
    st.write('**If you are satisfied with the Validation then Start the Forecast Or Reset the Session State and re-run the app for Hyperparameter Tunning**')
    
    
    numdays = st.number_input("Number of Day's you want to Forecaste",
                                  min_value = 1,
                                  max_value = n*3,
                                  value = n)
        
    if 'fr' not in st.session_state:
        st.session_state['fr'] = 0
            
    if st.session_state['stock_present']!= st.session_state['stock_past']:
        st.session_state['stock_past'] = st.session_state['stock_present']
        st.session_state['fr'] = 0
        
    frct = st.selectbox('Start the Forecast', options = ['No', 'Yes'],
                        index = st.session_state['fr'])
        
    if frct == 'Yes':
        st.subheader('Forecasting')
        st.session_state['fr'] = 1
        forecasters = [
                     ("prophet" , Pro_model),
                     ("expo" , Expo_model)
            ]
            
        st.write('Training the model')
        Ensmodel = AutoEnsembleForecaster(forecasters=forecasters, n_jobs=-1, random_state=42)
        Ensmodel.fit(full_data_minmax)
            
        st.write('Forecasting from trained model')
        prediction_list = [(pd.to_datetime(endDate) + dttm.timedelta(days=x)).date() for x in range(0,numdays)]
        prediction_list = pd.to_datetime(prediction_list)
        forecaste = pd.DataFrame(prediction_list, columns=['Date'])
        #st.write(forecaste)
        for_df = forecaste.set_index('Date')
        #st.write(for_df)
        for_df = for_df.asfreq('B')
        #n = int(len(for_df.index))
        #st.write(n)
            
            
        #fh = list(range(1,n+1))
        fh1 = pd.DatetimeIndex(np.array(for_df.index))
        # fh1
        final_predictions = Ensmodel.predict(fh=fh1)
        #st.write(final_predictions)
            
            
        final_predictions=scaler.inverse_transform(final_predictions)
        #st.write(final_predictions)
        for_df['Close'] = final_predictions
        st.markdown('### Forecast DataSet')
        st.write(for_df)
            
            
            
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=training_data.index, y=training_data['Close'], mode='lines', name='TRAIN'))
        fig.add_trace(go.Scatter(x=for_df.index, y=for_df['Close'], mode='lines', name='Forecast'))
        fig.update_layout(title_text='Final Forecast', title_x=0.5)
        
        st.write(fig)

            
    elif frct == 'No':
        st.session_state['fr'] = 0
        
    else:
        pass
            
        
        
    if st.sidebar.button('Made By'):
        name = ['Ayush Patidar', 'Aditya Rao', 'Farzan Nawaz', 
                'Nikhil Hosamani', 'Lakshmi Supriya', 'Bhavitha Mitte', 'Aadarsh Asthana']
        gmail = ['ayushpatidar1712@gmail.com', 'adityarao0909@gmail.com', 'farzannawaz4787@gmail.com',
                 'nikhilhosamani7777@gmail.com', 'karrilakshmisupriya@gmail.com', 'bhavithamitte292@gmail.com',
                 'aadarshasthana2017@gmail.com']
        dt = {'Name':name, 'Contact Detail': gmail}
        made = pd.DataFrame(dt)
        st.write(made)

else:
    pass