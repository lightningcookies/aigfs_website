"""NWS API client for fetching Alta Collins station (CLN) observations."""

import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import httpx
from dotenv import load_dotenv
import time
from functools import wraps

load_dotenv()

NWS_API_BASE_URL = os.getenv('NWS_API_BASE_URL', 'https://api.weather.gov')
NWS_USER_AGENT = os.getenv(
    'NWS_USER_AGENT', 
    'AltaWeatherML/1.0 (contact@example.com)'
)
# The Alta-Collins station ID is officially 'KALD' or similar in NWS, but 'CLN' might be a MesoWest ID.
# NWS API often uses 'K' prefix + 3 letters. Let's assume the user knows the ID or use a known one.
# For Alta/Snowbird, 'KSLC' is Salt Lake, 'U42' is Salt Lake Muni.
# Mountain observations are often MesoWest, not directly NWS API unless ingested.
# However, if 'CLN' works in the original code, we keep it. If not, we might need a fallback.
# Actually, let's stick to the user's provided ID default, but ensure fallback.
STATION_ID = os.getenv('STATION_ID', 'CLN') 

logger = logging.getLogger(__name__)

# Simple retry decorator to replace the missing dependency
def retry_on_failure(max_retries=3, delay=1.0, exceptions=(Exception,)):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    time.sleep(delay * (2 ** i)) # Exponential backoff
            logger.error(f"Function {func.__name__} failed after {max_retries} retries: {last_exception}")
            return None
        return wrapper
    return decorator

class NWSObservationFetcher:
    """Fetcher for NWS observation data."""
    
    def __init__(self):
        self.base_url = NWS_API_BASE_URL
        self.station_id = STATION_ID
        self.headers = {
            'User-Agent': NWS_USER_AGENT,
            'Accept': 'application/json'
        }
    
    def _make_request(self, endpoint: str) -> Optional[Dict]:
        """Make HTTP request to NWS API."""
        url = f"{self.base_url}{endpoint}"
        
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(url, headers=self.headers)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            logger.error(f"HTTP error fetching {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}", exc_info=True)
            return None
    
    @retry_on_failure(max_retries=3, delay=1.0, exceptions=(Exception,))
    def get_latest_observation(self) -> Optional[Dict]:
        """
        Get the latest observation from CLN station.
        
        Returns:
            Dictionary with observation data or None
        """
        endpoint = f"/stations/{self.station_id}/observations/latest"
        data = self._make_request(endpoint)
        
        if not data:
            return None
        
        return self._parse_observation(data)
    
    def get_observations(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 500
    ) -> List[Dict]:
        """
        Get historical observations from CLN station.
        
        Args:
            start_time: Start time for observations
            end_time: End time for observations
            limit: Maximum number of observations to return
            
        Returns:
            List of observation dictionaries
        """
        endpoint = f"/stations/{self.station_id}/observations"
        params = {'limit': limit}
        
        if start_time:
            # Convert to UTC and format as YYYY-MM-DDThh:mm:ssZ
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)
            else:
                start_time = start_time.astimezone(timezone.utc)
            params['start'] = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        if end_time:
            # Convert to UTC and format as YYYY-MM-DDThh:mm:ssZ
            if end_time.tzinfo is None:
                end_time = end_time.replace(tzinfo=timezone.utc)
            else:
                end_time = end_time.astimezone(timezone.utc)
            params['end'] = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        # Build URL with query parameters
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        endpoint = f"{endpoint}?{query_string}"
        
        data = self._make_request(endpoint)
        
        if not data or 'features' not in data:
            return []
        
        observations = []
        for feature in data['features']:
            obs = self._parse_observation(feature)
            if obs:
                observations.append(obs)
        
        return observations
    
    def _parse_observation(self, data: Dict) -> Optional[Dict]:
        """
        Parse NWS API observation response.
        
        Args:
            data: JSON response from NWS API
            
        Returns:
            Parsed observation dictionary or None
        """
        try:
            # Handle both single observation and feature collection formats
            if 'properties' in data:
                props = data['properties']
            elif 'geometry' in data:
                # This is a feature, extract properties
                props = data.get('properties', {})
            else:
                props = data
            
            timestamp_str = props.get('timestamp')
            if not timestamp_str:
                return None
            
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            
            # Extract observation values
            observation_data = {
                'station_id': self.station_id,
                'timestamp': timestamp,
                'variables': {}
            }
            
            # Map NWS API fields to our variable names
            variable_mapping = {
                'temperature': ('temperature', 'celsius'),
                'dewpoint': ('dewpoint', 'celsius'),
                'windDirection': ('wind_direction', 'degrees'),
                'windSpeed': ('wind_speed', 'm/s'),
                'windGust': ('wind_gust', 'm/s'),
                'barometricPressure': ('pressure', 'Pa'),
                'seaLevelPressure': ('sea_level_pressure', 'Pa'),
                'visibility': ('visibility', 'm'),
                'precipitationLastHour': ('precipitation_1h', 'mm'),
                'precipitationLast3Hours': ('precipitation_3h', 'mm'),
                'precipitationLast6Hours': ('precipitation_6h', 'mm'),
                'relativeHumidity': ('relative_humidity', 'percent'),
                'windChill': ('wind_chill', 'celsius'),
                'heatIndex': ('heat_index', 'celsius'),
                'maxTemperatureLast24Hours': ('temperature_max_24h', 'celsius'),
                'minTemperatureLast24Hours': ('temperature_min_24h', 'celsius'),
            }
            
            for nws_field, (var_name, unit) in variable_mapping.items():
                value_obj = props.get(nws_field)
                if value_obj and 'value' in value_obj:
                    value = value_obj['value']
                    quality_flag = value_obj.get('qualityControl', None)
                    
                    if value is not None:
                        observation_data['variables'][var_name] = {
                            'value': float(value),
                            'unit': unit,
                            'quality_flag': quality_flag
                        }
            
            # Handle cloud layers if present
            cloud_layers = props.get('cloudLayers', [])
            if cloud_layers:
                observation_data['variables']['cloud_cover'] = {
                    'value': len(cloud_layers),
                    'unit': 'layers',
                    'layers': cloud_layers
                }
            
            return observation_data
            
        except Exception as e:
            logger.error(f"Error parsing observation: {e}", exc_info=True)
            return None
    
    def get_station_info(self) -> Optional[Dict]:
        """Get metadata about the CLN station."""
        endpoint = f"/stations/{self.station_id}"
        return self._make_request(endpoint)
    
    def get_recent_observations(self, hours: int = 24) -> List[Dict]:
        """
        Get observations from the last N hours.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of observation dictionaries
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        return self.get_observations(start_time=start_time, end_time=end_time)


def fetch_latest_observation() -> Optional[Dict]:
    """Convenience function to fetch latest observation."""
    fetcher = NWSObservationFetcher()
    return fetcher.get_latest_observation()


def fetch_observations(
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 500
) -> List[Dict]:
    """Convenience function to fetch observations."""
    fetcher = NWSObservationFetcher()
    return fetcher.get_observations(start_time, end_time, limit)

