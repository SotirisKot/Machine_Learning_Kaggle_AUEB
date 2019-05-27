
# coding: utf-8

# ## Preparation, Imports and Function Declarations

# In[ ]:


get_ipython().run_line_magic('config', 'IPCompleter.greedy = True')


# In[ ]:


# Install GGPLOT
get_ipython().system('python -m pip install ggplot')


# In[ ]:


from pprint import pprint
import geopy.distance
import datetime
import pandas as pd
from ggplot import *

def get_distance_km(lat1, lon1, lat2, lon2):
    return geopy.distance.distance((lat1, lon1), (lat2, lon2)).km

import datetime

def transform_date(date):
    dates = date.split('-')
    datef = datetime.datetime(int(dates[0]),int(dates[1]),int(dates[2]))
    return datef.year, datef.month, datef.day, datef.weekday()

def Holiday(month, day):
    if month == 7 and day <= 10: return 'IDD'
    if month == 12: return 'CRI'
    if month in [3,4]: return 'SRB'
    if month == 11 and day >=22 and day<=28: return 'THG'
    if month == 1: return 'NYR'
    return 'NOT'

def Season(month,day):
    if(month in [9,10,11]) : return 'AUT'
    if(month in [12,1,2]) : return 'WIN'
    if(month in [3,4,5]) : return 'SPR'
    if(month in [6,7,8]) : return 'SUM'
    return 'NOT'

def train(development):
    df_train = pd.read_csv('dataset/train.csv')
    y_train = df_train[['PAX']]
    y_test = None
    
    if(development==False):
        df_test = pd.read_csv('dataset/test.csv')
    else:
        from sklearn.model_selection import train_test_split
        df_train, df_test, y_train, y_test = train_test_split(df_train, y_train, test_size=0.2, random_state=42)


    ### Extract the date and add new features from date

    # TRAIN SET
    tfdates = df_train.apply(lambda row: transform_date(row['DateOfDeparture']), axis=1)
    years = [t[0] for t in tfdates]
    months = [t[1] for t in tfdates]
    days = [t[2] for t in tfdates]
    weekdays = [t[3] for t in tfdates]
    df_train['Year'], df_train['Month'],df_train['Day'], df_train['WeekDay'] = years, months, days, weekdays

    # TEST SET
    tfdates = df_test.apply(lambda row: transform_date(row['DateOfDeparture']), axis=1)
    years = [t[0] for t in tfdates]
    months = [t[1] for t in tfdates]
    days = [t[2] for t in tfdates]
    weekdays = [t[3] for t in tfdates]
    df_test['Year'], df_test['Month'],df_test['Day'], df_test['WeekDay'] = years, months, days, weekdays

    ### Extract the distance from coordinates, longtitude and latitude are inversed -- !!!Dataset's error!!!

    # TRAIN SET
    distances = df_train.apply(lambda row: round(get_distance_km(row['LongitudeDeparture'],row['LatitudeDeparture'],row['LongitudeArrival'],row['LatitudeArrival']),3), axis=1)
    df_train['Distance'] = distances

    # TEST SET
    distances = df_test.apply(lambda row: round(get_distance_km(row['LongitudeDeparture'],row['LatitudeDeparture'],row['LongitudeArrival'],row['LatitudeArrival']),3), axis=1)
    df_test['Distance'] = distances

    ### Set min and max weeks to departure

    # TRAIN SET
    mins = df_train.apply(lambda row: round(row['WeeksToDeparture']-row['std_wtd'],3), axis=1)
    maxs = df_train.apply(lambda row: round(row['WeeksToDeparture']+row['std_wtd'],3), axis=1)

    df_train['MinWTD'] = mins
    df_train['MaxWTD'] = maxs

    # TEST SET
    mins = df_test.apply(lambda row: round(row['WeeksToDeparture']-row['std_wtd'],3), axis=1)
    maxs = df_test.apply(lambda row: round(row['WeeksToDeparture']+row['std_wtd'],3), axis=1)

    df_test['MinWTD'] = mins
    df_test['MaxWTD'] = maxs

    ### Find holidays, seasons

    # TRAIN SET
    holis = df_train.apply(lambda row: Holiday(row['Month'],row['Day']), axis=1)
    seas = df_train.apply(lambda row: Season(row['Month'],row['Day']), axis=1)

    df_train['Holiday'] = holis
    df_train['Season'] = seas

    # TEST SET
    holis = df_test.apply(lambda row: Holiday(row['Month'],row['Day']), axis=1)
    seas = df_test.apply(lambda row: Season(row['Month'],row['Day']), axis=1)

    df_test['Holiday'] = holis
    df_test['Season'] = seas

    torem = ['DateOfDeparture','CityDeparture','LongitudeDeparture','LatitudeDeparture','CityArrival','LongitudeArrival','LatitudeArrival','WeeksToDeparture','std_wtd','PAX','MinWTD','MaxWTD']
    
    if(development==False):
        
        df_train.drop(torem, axis=1, inplace=True)
                
        torem.remove('PAX')
        
        df_test.drop(torem, axis=1, inplace=True)
        
    else:
        
        df_train.drop(torem, axis=1, inplace=True)
        
        df_test.drop(torem, axis=1, inplace=True)
        
    df_train.reset_index(drop=True,inplace=True)
    df_test.reset_index(drop=True,inplace=True)
        
    print(df_train.head(),'\n'*5)
    print(df_test.head())
    
    return df_train, df_test, y_train, y_test

def runModel(df_train, df_test, y_train, showTsne):
    from sklearn.preprocessing import LabelEncoder
    
    le = LabelEncoder()
    # Departure and Arrival have the same values so we train only on Departure
    le.fit(df_train['Departure'])
    df_train['Departure'] = le.transform(df_train['Departure'])
    df_train['Arrival'] = le.transform(df_train['Arrival'])
    df_test['Departure'] = le.transform(df_test['Departure'])
    df_test['Arrival'] = le.transform(df_test['Arrival'])

    le.fit(df_train['Year'])
    df_train['Year'] = le.transform(df_train['Year'])
    le.fit(df_test['Year'])
    df_test['Year'] = le.transform(df_test['Year'])
    le.fit(df_train['Month'])
    df_train['Month'] = le.transform(df_train['Month'])
    le.fit(df_test['Month'])
    df_test['Month'] = le.transform(df_test['Month'])
    le.fit(df_train['Day'])
    df_train['Day'] = le.transform(df_train['Day'])
    le.fit(df_test['Day'])
    df_test['Day'] = le.transform(df_test['Day'])
    le.fit(df_train['WeekDay'])
    df_train['WeekDay'] = le.transform(df_train['WeekDay'])
    le.fit(df_test['WeekDay'])
    df_test['WeekDay'] = le.transform(df_test['WeekDay'])

    le.fit(df_train['Holiday'])
    df_train['Holiday'] = le.transform(df_train['Holiday'])
    le.fit(df_test['Holiday'])
    df_test['Holiday'] = le.transform(df_test['Holiday'])

    le.fit(df_train['Season'])
    df_train['Season'] = le.transform(df_train['Season'])
    le.fit(df_test['Season'])
    df_test['Season'] = le.transform(df_test['Season'])

    import numpy as np
    import codecs
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.utils import np_utils
    from keras import backend as K
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.under_sampling import TomekLinks
    from imblearn.pipeline import make_pipeline
    from keras import initializers 
    import tensorflow as tf
    
	### Uncomment for GPU
	
    #config = tf.ConfigProto(device_count={'GPU':1, 'CPU':56})
    #sess = tf.Session(config=config)
    #keras.backend.set_session(sess)
    
    X_train = df_train
    X_test = df_test
    y_train = np.ravel(y_train)
    
    ### Scale the data
    
    from sklearn.preprocessing import minmax_scale
    
    X_train['Distance'] = minmax_scale(X_train['Distance'])
    X_test['Distance'] = minmax_scale(X_test['Distance'])
    
    # # One-Hot encoding

    X_train_dep = pd.get_dummies(X_train['Departure'],prefix='dep')
    X_train_arr = pd.get_dummies(X_train['Arrival'],prefix='arr')
    X_train_y = pd.get_dummies(X_train['Year'],prefix='y')
    X_train_m = pd.get_dummies(X_train['Month'],prefix='m')
    X_train_d = pd.get_dummies(X_train['Day'],prefix='d')
    X_train_wd = pd.get_dummies(X_train['WeekDay'],prefix='wd')
    X_train_hol = pd.get_dummies(X_train['Holiday'],prefix='hol')
    X_train_sea = pd.get_dummies(X_train['Season'],prefix='sea')
    
    X_train_extra1 = pd.concat([X_train_dep, X_train_y, X_train_m, X_train_d, X_train_wd, X_train_hol, X_train_sea],axis=1)

    cols = X_train.columns[[0,1,2,3,4,5,7,8]]
    X_train[cols] = minmax_scale(X_train[cols])
        
    X_train_extra = pd.concat([X_train['Departure'], X_train['Arrival'], X_train['Year'], X_train['Month'], X_train['Day'], X_train['WeekDay'], X_train['Holiday'], X_train['Season'], X_train['Distance']],axis=1)
    
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=1)
    pca.fit(X_train_extra)
    X_train_extra = pca.transform(X_train_extra)
    X_train_extra = pd.DataFrame(X_train_extra)
    
    X_train_extra.reset_index(drop=True,inplace=True)    
    
    X_train = pd.concat([X_train,X_train_extra,X_train_dep,X_train_arr,X_train_extra1],axis=1,ignore_index=True)

    X_train.drop([0,1],axis=1, inplace=True)
    
    idx_scale = [9]
    
    for i in idx_scale:
        X_train[i] = minmax_scale(X_train[i])
    
    
    X_test_dep = pd.get_dummies(X_test['Departure'],prefix='dep')
    X_test_arr = pd.get_dummies(X_test['Arrival'],prefix='arr')
    X_test_y = pd.get_dummies(X_test['Year'],prefix='y')
    X_test_m = pd.get_dummies(X_test['Month'],prefix='m')
    X_test_d = pd.get_dummies(X_test['Day'],prefix='d')
    X_test_wd = pd.get_dummies(X_test['WeekDay'],prefix='wd')
    X_test_hol = pd.get_dummies(X_test['Holiday'],prefix='hol')
    X_test_sea = pd.get_dummies(X_test['Season'],prefix='sea')
    
    X_test_extra1 = pd.concat([X_test_dep, X_test_y, X_test_m, X_test_d, X_test_wd, X_test_hol, X_test_sea],axis=1)
    
    cols = X_test.columns[[0,1,2,3,4,5,7,8]]
    X_test[cols] = minmax_scale(X_test[cols])
        
    X_test_extra = pd.concat([X_test['Departure'], X_test['Arrival'], X_test['Year'], X_test['Month'], X_test['Day'], X_test['WeekDay'], X_test['Holiday'], X_test['Season'], X_test['Distance']],axis=1)

    X_test_extra = pca.transform(X_test_extra)
    X_test_extra = pd.DataFrame(X_test_extra)
    
    X_test_extra.reset_index(drop=True,inplace=True)

    X_test = pd.concat([X_test,X_test_extra,X_test_dep,X_test_arr,X_test_extra1],axis=1, ignore_index=True)
    
    X_test.drop([0,1],axis=1, inplace=True)
    
    idx_scale = [9]
    for i in idx_scale:
        X_test[i] = minmax_scale(X_test[i])
        
    ## TSNE
    
    if(showTsne):
    
        from sklearn.manifold import TSNE 
        tsne = TSNE(n_components=2,n_iter=250)
        tsne_res = tsne.fit_transform(X_train)

        df_tnse = pd.DataFrame(tsne_res)

        df_pax = pd.DataFrame(y_train)

        df_tnse = pd.concat([df_tnse,df_pax],axis=1,ignore_index=True)

        df_tnse.columns = ['X','Y','label']

        df_tnse['label'] = df_tnse['label'].astype(str)

        print(df_tnse.head())

        chart = ggplot(df_tnse,aes(x='X',y='Y',color='label'))+ geom_point(alpha=0.8) + ggtitle('tSNE')

        chart.show()
    
    print(X_train.head())
    X_train = X_train.values
    print(X_train.shape)
    
    def baseline_model1():
        # create model
        model = Sequential()
        model.add(Dense(units=200, activation='relu' ,kernel_initializer=keras.initializers.he_uniform(seed=None), bias_initializer='ones',input_shape=(X_train.shape[1],)))
        model.add(Dropout(0.2))
        model.add(Dense(units=100,activation='relu', kernel_initializer=keras.initializers.he_uniform(seed=None), bias_initializer='ones'))
        model.add(Dropout(0.2))
        model.add(Dense(units=50, activation='relu', kernel_initializer=keras.initializers.he_uniform(seed=None), bias_initializer='ones'))
        model.add(Dropout(0.2))
        model.add(Dense(units=25, activation='relu', kernel_initializer=keras.initializers.he_uniform(seed=None),bias_initializer='ones'))
        model.add(Dense(8, activation='softmax', kernel_initializer=keras.initializers.he_uniform(seed=None)))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adagrad')
        return model
    
    def baseline_model2():
        # create model
        model = Sequential()
        model.add(Dense(units=200, activation='relu' ,kernel_initializer=keras.initializers.he_uniform(seed=None), bias_initializer='ones',input_shape=(X_train.shape[1],)))
        model.add(Dropout(0.2))
        model.add(Dense(units=100,activation='relu', kernel_initializer=keras.initializers.he_uniform(seed=None), bias_initializer='ones'))
        model.add(Dropout(0.2))
        model.add(Dense(units=50, activation='relu', kernel_initializer=keras.initializers.he_uniform(seed=None), bias_initializer='ones'))
        model.add(Dropout(0.2))
        model.add(Dense(units=25, activation='relu', kernel_initializer=keras.initializers.he_uniform(seed=None),bias_initializer='ones'))
        model.add(Dense(units=16, activation='relu', kernel_initializer=keras.initializers.he_uniform(seed=None),bias_initializer='ones'))
        model.add(Dense(8, activation='softmax', kernel_initializer=keras.initializers.he_uniform(seed=None)))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adadelta')
        return model
    
    def baseline_model3():
        # create model
        model = Sequential()
        model.add(Dense(units=300, activation='relu' ,kernel_initializer=keras.initializers.he_uniform(seed=None), bias_initializer='ones',input_shape=(X_train.shape[1],)))
        model.add(Dropout(0.2))
        model.add(Dense(units=100,activation='relu', kernel_initializer=keras.initializers.he_uniform(seed=None), bias_initializer='ones'))
        model.add(Dropout(0.4))
        model.add(Dense(units=50, activation='relu', kernel_initializer=keras.initializers.he_uniform(seed=None), bias_initializer='ones'))
        model.add(Dropout(0.2))
        model.add(Dense(units=25, activation='relu', kernel_initializer=keras.initializers.he_uniform(seed=None),bias_initializer='ones'))
        model.add(Dense(8, activation='softmax', kernel_initializer=keras.initializers.he_uniform(seed=None)))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='nadam')
        return model
    
    def baseline_model4():
        # create model
        model = Sequential()
        model.add(Dense(units=200, activation='relu' ,kernel_initializer=keras.initializers.he_uniform(seed=None), bias_initializer='zeros',input_shape=(X_train.shape[1],)))
        model.add(Dropout(0.2))
        model.add(Dense(units=100,activation='relu', kernel_initializer=keras.initializers.he_uniform(seed=None), bias_initializer='zeros'))
        model.add(Dropout(0.2))
        model.add(Dense(units=50, activation='relu', kernel_initializer=keras.initializers.he_uniform(seed=None), bias_initializer='zeros'))
        model.add(Dropout(0.2))
        model.add(Dense(units=25, activation='relu', kernel_initializer=keras.initializers.he_uniform(seed=None),bias_initializer='zeros'))
        model.add(Dense(units=16, activation='relu', kernel_initializer=keras.initializers.he_uniform(seed=None),bias_initializer='zeros'))
        #sto apo panw eixe ones
        model.add(Dense(8, activation='softmax', kernel_initializer=keras.initializers.he_uniform(seed=None)))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        return model
    
    def baseline_model5():
        # create model
        model = Sequential()
        model.add(Dense(units=200, activation='relu' ,kernel_initializer=keras.initializers.he_uniform(seed=None), bias_initializer='zeros',input_shape=(X_train.shape[1],)))
        model.add(Dropout(0.2))
        model.add(Dense(units=100,activation='relu', kernel_initializer=keras.initializers.he_uniform(seed=None), bias_initializer='zeros'))
        model.add(Dropout(0.2))
        model.add(Dense(units=50, activation='relu', kernel_initializer=keras.initializers.he_uniform(seed=None), bias_initializer='zeros'))
        model.add(Dropout(0.2))
        model.add(Dense(units=25, activation='relu', kernel_initializer=keras.initializers.he_uniform(seed=None),bias_initializer='zeros'))
        model.add(Dense(8, activation='softmax', kernel_initializer=keras.initializers.he_uniform(seed=None)))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        return model
    
    def baseline_model6():
        # create model
        model = Sequential()
        model.add(Dense(units=300, activation='relu' ,kernel_initializer=keras.initializers.he_uniform(seed=None), bias_initializer='ones',input_shape=(X_train.shape[1],)))
        model.add(Dropout(0.2))
        model.add(Dense(units=100,activation='relu', kernel_initializer=keras.initializers.he_uniform(seed=None), bias_initializer='ones'))
        model.add(Dropout(0.2))
        model.add(Dense(units=50, activation='relu', kernel_initializer=keras.initializers.he_uniform(seed=None), bias_initializer='ones'))
        model.add(Dropout(0.2))
        model.add(Dense(units=25, activation='relu', kernel_initializer=keras.initializers.he_uniform(seed=None),bias_initializer='ones'))
        model.add(Dense(8, activation='softmax', kernel_initializer=keras.initializers.he_uniform(seed=None)))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='nadam')
        return model
    

    sampling_strategy = 'minority'
    
    tl = RandomOverSampler(sampling_strategy)
    X_train, y_train = tl.fit_resample(X_train, y_train)
    y_train = np_utils.to_categorical(y_train)
   
    model1 = baseline_model1()
    model2 = baseline_model2()
    model3 = baseline_model3()
    model4 = baseline_model4()
    model5 = baseline_model5()
    model6 = baseline_model6()
    model1.fit(X_train, y_train, epochs=150, batch_size=256,shuffle=True)
    model2.fit(X_train, y_train, epochs=150, batch_size=256,shuffle=True)
    model3.fit(X_train, y_train, epochs=100, batch_size=256,shuffle=True)
    model4.fit(X_train, y_train, epochs=150, batch_size=256,shuffle=True)
    model5.fit(X_train, y_train, epochs=100, batch_size=256,shuffle=True)
    model6.fit(X_train, y_train, epochs=100, batch_size=256,shuffle=True)
    prob1 = model1.predict(X_test)
    prob2 = model2.predict(X_test)
    prob3 = model3.predict(X_test)
    prob4 = model4.predict(X_test)
    prob5 = model5.predict(X_test)
    prob6 = model6.predict(X_test)

    final_prob = (prob1 + prob2 + prob3 + prob4 + prob5 + prob6) / 6
    
    return final_prob.argmax(axis=1)
    
def evaluateModel(y_pred, y_test, development):
    if(development==False):
        # Write the predictions to the file
        import csv
        with open('y_pred.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['Id', 'Label'])
            for i in range(y_pred.shape[0]):
                writer.writerow([i, y_pred[i]])
        return None
    else:
        # FOR DEVELOPMENT SET
        print("Your predictions: ", y_pred)
        # print("Test predictions: ", y_test)
        from sklearn.metrics import f1_score
        return f1_score(y_test, y_pred, average='micro')


# ## Main Code

# In[ ]:


development = False

df_train, df_test, y_train, y_test = train(development)


# In[ ]:


showTsne = False
y_pred = runModel(df_train, df_test, y_train, showTsne)
score = evaluateModel(y_pred, y_test, development)
if(score != None): print(score)

