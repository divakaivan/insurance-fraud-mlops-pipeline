import os
import numpy as np
import pandas as pd
from prefect import task, flow
from prefect_gcp.cloud_storage import GcsBucket
from sqlalchemy import create_engine

db_user = os.getenv('POSTGRES_USER')
db_password = os.getenv('POSTGRES_PASSWORD')
db_table = os.getenv('POSTGRES_TABLE')
db_name = os.getenv('POSTGRES_NAME')

# pre-defined meaningful features based on IV values during feature exploration
meaningful_features = ['NumberOfSuppliments',
                            'AgeOfVehicle',
                            'AgeOfPolicyHolder',
                            'Month',
                            'Deductible',
                            'MonthClaimed',
                            'Make',
                            'AddressChange_Claim',
                            'PastNumberOfClaims',
                            'VehiclePrice',
                            'VehicleCategory',
                            'Fault']

def calculate_woe_iv_categorical(df, feature, target):
    """Calculate WOE and IV for categorical features"""
    df = df[[feature, target]].copy()

    grouped = df.groupby(feature)[target].agg(['count', 'sum'])
    grouped['non_event'] = grouped['count'] - grouped['sum']
    grouped.columns = ['total', 'event', 'non_event']

    grouped['event_rate'] = grouped['event'] / grouped['event'].sum()
    grouped['non_event_rate'] = grouped['non_event'] / grouped['non_event'].sum()

    eps = 0.0001 # to avoid division by zero
    grouped['woe'] = np.log((grouped['event_rate'] + eps) / (grouped['non_event_rate'] + eps))
    grouped['iv'] = (grouped['event_rate'] - grouped['non_event_rate']) * grouped['woe']

    iv = grouped['iv'].sum()

    return grouped, iv

def calculate_woe_iv_numeric(df, feature, target, bins=10):
    """Calculate WOE and IV for numerical features"""
    df = df[[feature, target]].copy()
    df['bin'] = pd.qcut(df[feature], q=bins, duplicates='drop')

    grouped = df.groupby('bin', observed=True)[target].agg(['count', 'sum'])
    grouped['non_event'] = grouped['count'] - grouped['sum']
    grouped.columns = ['total', 'event', 'non_event']

    grouped['event_rate'] = grouped['event'] / grouped['event'].sum()
    grouped['non_event_rate'] = grouped['non_event'] / grouped['non_event'].sum()

    eps = 0.0001 # to avoid division by zero
    grouped['woe'] = np.log((grouped['event_rate'] + eps) / (grouped['non_event_rate'] + eps))
    grouped['iv'] = (grouped['event_rate'] - grouped['non_event_rate']) * grouped['woe']

    iv = grouped['iv'].sum()

    return grouped, iv

@task
def load_from_gcs():
    """Load raw data from GCS bucket"""
    gcs_bucket = GcsBucket.load('fraud_modelling_prefect', validate=False)
    gcs_bucket.download_object_to_path('raw_data/raw_car_insurance_kaggle.csv', 'raw_car_insurance_kaggle.csv')

@task
def load_from_local():
    """Load raw data from local"""
    df = pd.read_csv('raw_data/raw_car_insurance_kaggle.csv')
    return df

@task(log_prints=True)
def monitor_new_data(df):
    """Monitor new data and check if there is deviation from the pre-defined meaningful features"""
    print('New data has been loaded successfully')
    print(f'Meaningful features based on pre-determined IV (>0.02 and < 0.6): {meaningful_features}')

    iv_values = {}

    target = 'FraudFound_P'
    numerical_features = ['DriverRating']
    categorical_features = [
        'Month', 'WeekOfMonth', 'DayOfWeek',
        'Make', 'AccidentArea', 'DayOfWeekClaimed',
        'MonthClaimed', 'WeekOfMonthClaimed', 'Sex',
        'MaritalStatus', 'Fault', 'PolicyType',
        'VehicleCategory', 'VehiclePrice', 'Days_Policy_Accident',
        'Days_Policy_Claim', 'PastNumberOfClaims', 'AgeOfVehicle',
        'AgeOfPolicyHolder', 'PoliceReportFiled', 'WitnessPresent',
        'AgentType', 'NumberOfSuppliments', 'AddressChange_Claim',
        'NumberOfCars', 'Year', 'BasePolicy', 'Deductible'
    ]
    for feature in numerical_features:
        _, iv = calculate_woe_iv_numeric(df, feature, target)
        iv_values[feature] = iv
    for feature in categorical_features:
        _, iv = calculate_woe_iv_categorical(df, feature, target)
        iv_values[feature] = iv
    sorted_iv_values = dict(sorted(iv_values.items(), key=lambda item: item[1]))
    current_meaningful_features = [feature for feature, iv in sorted_iv_values.items() if iv > 0.02 and iv < 0.6]
    print(f'IV of current data\'s features: {current_meaningful_features}')
    print('If the previous and current mismatch, a feature review shall be conducted')

@task
def preprocess_meaningful_features(df):
    """Preprocess meaningful features"""
    meaningful_X = df[meaningful_features]
    y = df['FraudFound_P']
    meaningful_X['NumberOfSuppliments'] = meaningful_X['NumberOfSuppliments'].replace({
                                                'none': 'none or 1 to 2',
                                                '1 to 2': 'none or 1 to 2',
                                                '3 to 5': 'more than 3',
                                                'more than 5': 'more than 3'
                                                })
    meaningful_X['AgeOfVehicle'] = meaningful_X['AgeOfVehicle'].replace({
                                                '3 years': '3-4 years',
                                                '4 years': '3-4 years',
                                                '5 years': 'more than 5 years',
                                                '6 years': 'more than 5 years',
                                                '7 years': 'more than 5 years',
                                                'more than 7': 'more than 5 years'
                                                })
    meaningful_X['AgeOfPolicyHolder'] = meaningful_X['AgeOfPolicyHolder'].replace({
                                                '41 to 50': '41 to 65',
                                                '51 to 65': '41 to 65',
                                                })
    meaningful_X['Month'] = meaningful_X['Month'].replace({
                                    'Jan': 'Jan-Feb',
                                    'Feb': 'Jan-Feb'
                                    })
    meaningful_X['Deductible'] = meaningful_X['Deductible'].astype(str)
    meaningful_X['MonthClaimed'] = meaningful_X['MonthClaimed'].replace({
                                            'Jan': 'Jan-Feb',
                                            'Feb': 'Jan-Feb'
                                            })
    meaningful_X['Make'] = meaningful_X['Make'].replace({
                                'Lexus': 'Lexus/Ferrari/Porche/Jaguar',
                                'Ferrari': 'Lexus/Ferrari/Porche/Jaguar',
                                'Porche': 'Lexus/Ferrari/Porche/Jaguar',
                                'Jaguar': 'Lexus/Ferrari/Porche/Jaguar'
                                })
    meaningful_X['VehiclePrice'] = meaningful_X['VehiclePrice'].replace({
                                                '20000 to 29000': '20000 to 39000',
                                                '30000 to 39000': '20000 to 39000'
                                                })
    meaningful_df = pd.concat([meaningful_X, y], axis=1)
    return meaningful_df

@task
def upload_meaningful_df_to_postgres(meaningful_df):
    """Upload meaningful data to PostgreSQL"""
    engine = create_engine(f'postgresql://{db_user}:{db_password}@localhost:5432/{db_name}')
    meaningful_df.to_sql(db_table, engine, if_exists='replace', index=False)

@task
def create_model_data_w_dummy(meaningful_df):
    """Create model data with dummy variables"""
    meaningful_X = meaningful_df.drop('FraudFound_P', axis=1)
    model_data_w_dummy = pd.get_dummies(meaningful_X, drop_first=True)
    model_data_w_dummy['FraudFound_P'] = meaningful_df['FraudFound_P']
    return model_data_w_dummy

@task
def upload_model_data_to_postgres(model_data_w_dummy):
    """Upload model data with dummy variables to PostgreSQL"""
    engine = create_engine(f'postgresql://{db_user}:{db_password}@localhost:5432/{db_name}')
    model_data_w_dummy.to_sql(db_table, engine, if_exists='replace', index=False)

@flow
def preprocess_data():
    """Orchestrate the data preprocessing tasks"""
    df = load_from_local()
    monitor_new_data(df)
    meaningful_df = preprocess_meaningful_features(df)
    upload_meaningful_df_to_postgres(meaningful_df)
    model_data_w_dummy = create_model_data_w_dummy(meaningful_df)
    upload_model_data_to_postgres(model_data_w_dummy)

if __name__ == '__main__':
    preprocess_data()
