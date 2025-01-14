# Module to fetch economic data

import requests
import os
import json

import tomllib  # Available in Python 3.11 and above

def get_api_key(service_name):
    """
    Retrieve the API key for a given service from a TOML configuration file.

    :param service_name: Name of the service (e.g., 'openAI', 'FRED', 'BLS', 'BEA')
    :return: API key value as a string
    :raises FileNotFoundError: If the configuration file is not found
    :raises KeyError: If the service name is not found in the configuration
    :raises ValueError: If there is an error decoding the TOML file
    """
    config_file_path = "API_Key/api_key.toml"

    try:
        with open(config_file_path, 'rb') as file:
            config = tomllib.load(file)
            try:
                return config['api_keys'][service_name]
            except KeyError:
                raise KeyError(f"API key for service '{service_name}' not found.")
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file '{config_file_path}' not found.")
    except tomllib.TOMLDecodeError:
        raise ValueError(f"Error decoding TOML from the configuration file '{config_file_path}'.")


# from dotenv import load_dotenv
#
# # Load environment variables from .env file
# load_dotenv()


class DataFetcher:
    """Class to fetch economic data from BLS, BEA, and FRED APIs."""

    def __init__(self):
        # Get API keys from environment variables
        self.bls_api_key = get_api_key("BLS")
        self.bea_api_key = get_api_key("BEA")
        self.fred_api_key = get_api_key("FRED")

        # Base URLs for APIs
        self.bls_base_url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
        self.bea_base_url = "https://apps.bea.gov/api/data/"
        self.fred_base_url = "https://api.stlouisfed.org/fred/series/observations"

        # Check if API keys are set
        if not self.bls_api_key:
            raise ValueError("BLS API key not found. Please set BLS_API_KEY in your .env file.")
        if not self.fred_api_key:
            raise ValueError("FRED API key not found. Please set FRED_API_KEY in your .env file.")
        if not self.bea_api_key:
            raise ValueError("BEA API key not found. Please set BEA_API_KEY in your .env file.")

    def fetch_unemployment_rate(self, start_year, end_year):
        """Fetch Unemployment Rate from BLS."""
        series_id = "LNS14000000"  # Unemployment rate (seasonally adjusted)
        return self._fetch_bls_data(series_id, start_year, end_year)

    def fetch_nonfarm_payroll(self, start_year, end_year):
        """Fetch Unemployment Rate from BLS."""
        series_id = "CES0000000001"  # Unemployment rate (seasonally adjusted)
        return self._fetch_bls_data(series_id, start_year, end_year)
    def fetch_cpi(self, start_year, end_year):
        """Fetch Consumer Price Index (CPI) from BLS."""
        series_id = "CUUR0000SA0"  # All Urban Consumers, All Items
        return self._fetch_bls_data(series_id, start_year, end_year)

    def fetch_fed_funds_rate(self, start_year, end_year):
        """Fetch historical Fed Funds Rate from FRED API."""
        series_id = "FEDFUNDS"  # Federal Funds Rate
        params = {
            "api_key": self.fred_api_key,
            "file_type": "json",
            "series_id": series_id,
            "observation_start": f"{start_year}-01-01",
            "observation_end": f"{end_year}-12-31",
        }

        response = requests.get(self.fred_base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            return data["observations"]
        else:
            raise Exception(f"Failed to fetch Fed Funds Rate data: {response.status_code}")

    def fetch_pce(self, start_year, end_year):
        """Fetch Personal Consumption Expenditures (PCE) data from BEA."""
        params = {
            "UserID": self.bea_api_key,
            "method": "GetData",
            "datasetname": "NIPA",
            "TableName": "T20804",  # PCE table
            "Frequency": "M",  # Monthly data
            "Year": f"{start_year}-{end_year}",
            "ResultFormat": "json",
        }

        response = requests.get(self.bea_base_url, params=params)
    
        if response.status_code == 200:
            data = response.json()
            try:
                return data["BEAAPI"]["Results"]["Data"]
            except KeyError as e:
                raise Exception(f"Missing key in response: {e}")
        else:
            raise Exception(f"Failed to fetch PCE data: {response.status_code}")


    def _fetch_bls_data(self, series_id, start_year, end_year):
        """Internal method to fetch data from BLS."""
        response = requests.post(
            self.bls_base_url,
            json={
                "seriesid": [series_id],
                "startyear": str(start_year),
                "endyear": str(end_year),
                "registrationkey": self.bls_api_key,
            },
        )

        if response.status_code == 200:
            data = response.json()
            return data["Results"]["series"][0]["data"]
        else:
            raise Exception(f"Failed to fetch BLS data: {response.status_code}")
