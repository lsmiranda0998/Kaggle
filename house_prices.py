"""
Algoritmo para prever o preço das casas (Kaggle)

"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
def predict(df,dict_features):
    x_train = np.nan_to_num(df.loc[df['SalePrice'] != -1, dict_features['categorical']+dict_features['numeric']].values)
    y_train = np.nan_to_num(df.loc[df['SalePrice'] != -1, 'SalePrice'])
    
    x_test = np.nan_to_num(df.loc[df['SalePrice'] == -1, dict_features['categorical']+dict_features['numeric']].values)
    
    model = AdaBoostRegressor(RandomForestRegressor(n_estimators = 10, max_depth = 5000), n_estimators = 50, learning_rate = 0.01)
    
    model.fit(x_train,y_train)
    return model.predict(x_test)

def featureEngineering(df):
    #cria dicionario
    dict_features = {}
    
    #elimina do dataset o target
    columns = list(df.columns)
    if ('SalePrice' in columns):
        columns.remove('SalePrice')
    
    #cria mascara para pegar apenas colunas string
    column_mask = np.array(df[columns].applymap(type) == str).all(0)
    str_columns = np.array(df[columns].columns[column_mask])
    
    #cria novas colunas binárias para colunas categóricas
    categorical_features = []
    for col in str_columns:
        labels = df[col].unique()
        
        for l in labels:
            categorical_features.append(col+'_'+l)
            df[col+'_'+l] = (df[col] == l).astype(int)
        df[col+'_'+'NA'] = (df[col].isna()).astype(int)
        
    column_mask = np.array(df[columns].applymap(type) == float).all(0)
    numeric_columns = np.array(df[columns].columns[column_mask])
    dict_features['categorical'] = categorical_features
    dict_features['numeric'] = list(numeric_columns)
    dict_features['target'] = 'SalePrice'
    return df,dict_features

def preProcessing(train,test):
    test['SalePrice'] = -1
    data = train.append(test)
    data = data.drop('Id',1)
    return data
    

    
    
    
    
path = 'C:\\Users\\Infoeste\\Downloads\\houses\\'
train = pd.read_csv (path+'train.csv', sep=',')
test = pd.read_csv(path+'test.csv', sep =',')

df = preProcessing(train,test)

df,dict_features = featureEngineering(df)

pred = predict(df,dict_features)

df_pred = pd.DataFrame()
df_pred['Id'] = test['Id']
df_pred['SalePrice'] = pred

df_pred.to_csv(path+'previsao.csv', index = False)