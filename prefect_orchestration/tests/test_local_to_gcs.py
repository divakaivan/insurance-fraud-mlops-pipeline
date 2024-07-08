from unittest.mock import patch, MagicMock
import pandas as pd
from raw_kaggle_to_gcs import local_to_gcs
from prefect_gcp import GcpCredentials

def test_local_to_gcs():
    """Test the local_to_gcs function"""
    # Mock the pandas read_csv function to return a sample DataFrame
    with patch('pandas.read_csv') as mock_read_csv:
        mock_df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        mock_read_csv.return_value = mock_df

        # Mock the GcpCredentials.load method and the GcpCredentials instance
        with patch('prefect_gcp.GcpCredentials.load', autospec=True) as mock_load_creds:
            mock_gcp_credentials = MagicMock(spec=GcpCredentials)
            mock_gcp_credentials.service_account_info = {}
            mock_load_creds.return_value = mock_gcp_credentials

            # Mock the GcsBucket class and its method
            with patch('prefect_gcp.GcsBucket', autospec=True):

                # Call the function to be tested
                local_to_gcs('/fake/path')

                # Assert that read_csv was called with the correct file path
                mock_read_csv.assert_called_once_with('/fake/path/fraud_oracle.csv')

                # Assert that GcpCredentials.load was called with the correct arguments
                mock_load_creds.assert_called_once_with("my-gcp-creds-block", validate=False)
