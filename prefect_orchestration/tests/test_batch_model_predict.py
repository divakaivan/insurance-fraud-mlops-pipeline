import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from sqlalchemy.engine.base import Engine
from batch_model_predict import read_new_data, predict, save_predictions, batch_model_predict

@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    monkeypatch.setenv('db_host', 'localhost')
    monkeypatch.setenv('db_port', '5432')
    monkeypatch.setenv('db_name', 'test_db')
    monkeypatch.setenv('db_username', 'test_user')
    monkeypatch.setenv('db_password', 'test_pass')

@pytest.fixture
def mock_engine():
    with patch('batch_model_predict.create_engine') as mock_create_engine:
        mock_engine = MagicMock(spec=Engine)
        mock_create_engine.return_value = mock_engine
        yield mock_engine

@pytest.fixture
def mock_read_sql():
    with patch('batch_model_predict.pd.read_sql_query') as mock_read_sql_query:
        mock_df = pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4]})
        mock_read_sql_query.return_value = mock_df
        yield mock_read_sql_query

def test_read_new_data(mock_read_sql, mock_engine):
    table_name = 'test_table'
    result = read_new_data.fn(table_name)
    assert not result.empty
    mock_read_sql.assert_called_once()
    mock_engine.dispose.assert_called_once()

@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict.return_value = [0, 1]
    model.predict_proba.return_value = [[0.1, 0.9], [0.8, 0.2]]
    return model

def test_predict(mock_model):
    data = pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4]})
    result = predict.fn(mock_model, data, get_probs=True)
    assert 'FraudFound_P' in result.columns
    assert 'FraudFound_P_Prob' in result.columns

@pytest.fixture
def mock_to_sql():
    with patch('batch_model_predict.pd.DataFrame.to_sql') as mock_to_sql:
        yield mock_to_sql

def test_save_predictions(mock_to_sql, mock_engine):
    data = pd.DataFrame({'feature1': [1, 2], 'FraudFound_P': [0, 1]})
    table_name = 'test_predictions'
    save_predictions.fn(data, table_name)
    mock_to_sql.assert_called_once_with(table_name, mock_engine, if_exists='append', index=False)
    mock_engine.dispose.assert_called_once()

def test_batch_model_predict(mock_model):
    with patch('batch_model_predict.read_new_data') as mock_read_new_data, \
         patch('batch_model_predict.load_model_from_mlflow') as mock_load_model, \
         patch('batch_model_predict.predict') as mock_predict, \
         patch('batch_model_predict.save_predictions') as mock_save_predictions, \
         patch('batch_model_predict.pause_flow_run') as mock_pause_flow_run:
        
        mock_pause_flow_run.return_value = 'test_table'
        mock_read_new_data.return_value = pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4]})
        mock_load_model.return_value = mock_model
        mock_predict.return_value = pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4], 'FraudFound_P': [0, 1]})
        
        batch_model_predict()

        mock_pause_flow_run.assert_called_once()
        mock_read_new_data.assert_called_once_with('test_table')
        mock_load_model.assert_called_once()
        mock_predict.assert_called_once()
        mock_save_predictions.assert_called_once()
