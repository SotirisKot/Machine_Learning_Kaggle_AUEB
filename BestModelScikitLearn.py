
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

from sklearn.feature_extraction import DictVectorizer

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

    le.fit(df_train['Holiday'])
    df_train['Holiday'] = le.transform(df_train['Holiday'])
    le.fit(df_test['Holiday'])
    df_test['Holiday'] = le.transform(df_test['Holiday'])

    le.fit(df_train['Season'])
    df_train['Season'] = le.transform(df_train['Season'])
    le.fit(df_test['Season'])
    df_test['Season'] = le.transform(df_test['Season'])

    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    import numpy as np
    import codecs

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

    cols = X_train.columns[[0,1,2,3,4,5,7,8]]
    X_train[cols] = minmax_scale(X_train[cols])
        
    X_train_extra = pd.concat([X_train['Departure'], X_train['Arrival'], X_train['Year'], X_train['Month'], X_train['Day'], X_train['WeekDay'], X_train['Holiday'], X_train['Season'], X_train['Distance']],axis=1)
    
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=1)
    pca.fit(X_train_extra)
    X_train_extra = pca.transform(X_train_extra)
    X_train_extra = pd.DataFrame(X_train_extra)
    
    X_train_extra.reset_index(drop=True,inplace=True)
       
    X_train = pd.concat([X_train,X_train_extra,X_train_dep,X_train_arr],axis=1,ignore_index=True)
    
    X_train.drop([0,1],axis=1, inplace=True)
    
    idx_scale = [9]
    for i in idx_scale:
        X_train[i] = minmax_scale(X_train[i])
    
    
    X_test_dep = pd.get_dummies(X_test['Departure'],prefix='dep')
    X_test_arr = pd.get_dummies(X_test['Arrival'],prefix='arr')
    
    cols = X_test.columns[[0,1,2,3,4,5,7,8]]
    X_test[cols] = minmax_scale(X_test[cols])
        
    X_test_extra = pd.concat([X_test['Departure'], X_test['Arrival'], X_test['Year'], X_test['Month'], X_test['Day'], X_test['WeekDay'], X_test['Holiday'], X_test['Season'], X_test['Distance']],axis=1)
    
    X_test_extra = pca.transform(X_test_extra)
    X_test_extra = pd.DataFrame(X_test_extra)
    
    X_test_extra.reset_index(drop=True,inplace=True)
    
    X_test = pd.concat([X_test,X_test_extra,X_test_dep,X_test_arr],axis=1, ignore_index=True)
    
    X_test.drop([0,1],axis=1, inplace=True)
    
    idx_scale = [9]
    
    for i in idx_scale:
        X_test[i] = minmax_scale(X_test[i])
    

    ## Print the data
    
    print(X_train.head())
    
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
    
    clf1 = AdaBoostClassifier(
        ExtraTreeClassifier(max_depth=10),
        n_estimators=500,
        learning_rate=0.01,
        algorithm="SAMME")
    clf2 = GradientBoostingClassifier(
         n_estimators=400,
         learning_rate=0.1,
         max_depth=3,
         subsample=1)
    clf3 = RandomForestClassifier(n_estimators=600,max_depth=None,criterion='entropy')

    from sklearn.multiclass import OneVsOneClassifier
    
    clf = OneVsOneClassifier(VotingClassifier(estimators=[('ada',clf1),('gb',clf2),('rf',clf3)],voting='soft',weights=[1,1.5,2]))
    
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(X_train.shape)
    print(X_test.shape)
    
    return y_pred
    
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
        print(y_pred)
        from sklearn.metrics import f1_score
        return f1_score(y_test, y_pred, average='micro')


# # Main code

# In[ ]:


development = False

df_train, df_test, y_train, y_test = train(development)


# In[ ]:


showTsne = False
y_pred = runModel(df_train, df_test, y_train, showTsne)
score = evaluateModel(y_pred, y_test, development)
if(score != None): print(score)

