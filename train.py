import numpy as np
import pandas as pd
import pickle
from xgboost import  XGBClassifier

best_params={
    'objective': 'binary:logistic',
    'learning_rate': 0.07118528947223794,
    'max_depth': 4,
    'n_estimators': 64,
    'subsample': 0.728034992108518,
    'seed': 42
    }

# data
PATH='./data/Zzzs_train.parquet'

# parameters
output_file = './xgb_classifier.bin'

features = ["hour",
            "anglez",
            "enmo"]


def reduce_mem_usage(df):
    
    """ 
    Iterate through all numeric columns of a dataframe and modify the data type
    to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage of dataframe is {start_mem:.2f} MB')
    for col in df.columns:
        col_type = df[col].dtype
        c_min = df[col].min()
        c_max = df[col].max()
        if str(col_type)[:3] == 'int':
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                df[col] = df[col].astype(np.int64)  
            elif  str(col_type)[:5] == 'float':
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        
    df['series_id'] = df['series_id'].astype('category')
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage after optimization is: {end_mem:.2f} MB')
    decrease = 100 * (start_mem - end_mem) / start_mem
    print(f'Decreased by {decrease:.2f}%')
    
    return df


def replace_enmo_by_qunatile(df, q=0.99):
    q_99=df['enmo'].quantile(q)
    mask=df['enmo']>q_99
    df['enmo'] = np.where(mask, q_99, df['enmo'])
    return df


def preprocess(df, train=True):
    print('preprocessing')
    df['timestamp'] = pd.to_datetime(df['timestamp']).apply(lambda t: t.tz_localize(None)) 
    if train:
        df=reduce_mem_usage(df)
    return df

def make_features(df, shuffle=False):
    if shuffle:
        df=df.sample(frac=1., random_state=42)
    df["hour"] = df["timestamp"].dt.hour
    df["anglez"] = abs(df["anglez"])
    df=replace_enmo_by_qunatile(df)
    print('features generated')
    return df


def train_model(X_train, y_train):
    print('trainning model')
    classifier = XGBClassifier(**best_params)
    classifier.fit(X_train, y_train)
    return classifier


def save_model(model):
    with open(output_file, 'wb') as f_out: 
        pickle.dump(model, f_out)


def load_data(path):
    data = pd.read_parquet(path)
    # shrink data because we are ok with current model performance
    data = data[data['series_id'].isin([
    '08db4255286f', '0a96f4993bd7', '0cfc06c129cc', '1087d7b0ff2e',
    '10f8bc1f7b07', '18b61dd5aae8', '29c75c018220', '3452b878e596'])]
    return data


if __name__ == "__main__":
    data=load_data(PATH)
    data = preprocess(data)
    data = make_features(data)
    X_train = data[features]
    y_train = data['awake']
    model=train_model(X_train, y_train)
    save_model(model)
