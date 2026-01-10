import os
import sys
import sqlite3
import time
import logging
from datetime import datetime, timedelta
import numpy as np
import xarray as xr
import pytz

# Add parent directory to path to import observation_fetcher
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from observation_fetcher import NWSObservationFetcher

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = os.path.join("backend", "ml_data.db")

# Alta/Collins Coordinates (Approximate for GFS extraction)
ALTA_LAT = 40.57  # Collins is around here
ALTA_LON = -111.63

# NWS to GFS Variable Mapping
# GFS keys are based on our cfgrib/xarray loading
VAR_MAPPING = {
    'temperature': 't2m',       # degC vs K
    'wind_speed': 'wind_speed', # m/s vs m/s (derived)
    'precipitation_1h': 'tp'    # mm vs mm (derived from acum)
}

def init_db():
    """Initialize the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Table for training data pairs
    c.execute('''
        CREATE TABLE IF NOT EXISTS training_data (
            timestamp TEXT PRIMARY KEY,
            
            obs_temp REAL,
            obs_wind REAL,
            obs_precip REAL,
            
            gfs_temp REAL,
            gfs_wind REAL,
            gfs_precip REAL,
            
            gfs_run_date TEXT,
            gfs_fhr INTEGER
        )
    ''')
    
    # Table for trained model coefficients
    c.execute('''
        CREATE TABLE IF NOT EXISTS model_coefficients (
            variable TEXT PRIMARY KEY,
            slope REAL,
            intercept REAL,
            rmse REAL,
            last_updated TEXT,
            sample_count INTEGER
        )
    ''')
    
    conn.commit()
    conn.close()

def get_gfs_forecast_for_time(target_time_utc):
    """
    Finds the most recent GFS run and forecast hour that covers the target time.
    Returns dictionary of GFS values or None.
    """
    data_dir = "data"
    if not os.path.exists(data_dir):
        return None

    # We look for the most recent run that is BEFORE target_time
    # GFS runs: 00, 06, 12, 18 UTC
    # We want the run that is closest but prior to target_time
    
    best_run = None
    best_fhr = None
    best_file = None
    
    # Scan all available runs
    runs = []
    for d in os.listdir(data_dir):
        if '_' in d:
            try:
                dt_str, hr_str = d.split('_')
                run_dt = datetime.strptime(f"{dt_str}{hr_str}", "%Y%m%d%H")
                runs.append((run_dt, d))
            except: pass
            
    runs.sort(reverse=True) # Newest first
    
    for run_dt, run_dir_name in runs:
        # Check if this run is before target time (it should be, otherwise it's predicting the future relative to run?)
        # Actually, GFS predicts future. So run_time < target_time is correct.
        if run_dt > target_time_utc:
            continue
            
        # Calculate needed FHR
        diff_hours = (target_time_utc - run_dt).total_seconds() / 3600
        fhr = int(round(diff_hours))
        
        # GFS files are usually every 3 or 6 hours.
        # We need to find the file that matches this FHR.
        # Our downloader gets every 6h? 
        # Filename: aigfs.t{run}z.sfc.f{fhr}.grib2
        
        # We need exact match or close match? 
        # Let's look for exact match in our downloaded files
        # Check f{fhr:03d}
        
        fname = f"aigfs.t{run_dt.strftime('%H')}z.sfc.f{fhr:03d}.grib2"
        fpath = os.path.join(data_dir, run_dir_name, fname)
        
        if os.path.exists(fpath):
            best_run = run_dir_name
            best_fhr = fhr
            best_file = fpath
            break
            
    if not best_file:
        return None
        
    # Extract values from GRIB
    try:
        values = {}
        
        # T2m
        try:
            ds = xr.open_dataset(best_file, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 2}})
            val = ds['t2m'].sel(latitude=ALTA_LAT, longitude=ALTA_LON, method='nearest').values
            values['temp'] = float(val) - 273.15 # K to C
            ds.close()
        except: values['temp'] = None
        
        # Wind
        try:
            ds = xr.open_dataset(best_file, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 10}})
            u = ds['u10'].sel(latitude=ALTA_LAT, longitude=ALTA_LON, method='nearest').values
            v = ds['v10'].sel(latitude=ALTA_LAT, longitude=ALTA_LON, method='nearest').values
            values['wind'] = float(np.sqrt(u**2 + v**2)) # m/s
            ds.close()
        except: values['wind'] = None

        # Precip (Total Precipitation)
        # Note: GFS 'tp' is often accumulated. 
        # If we want 1h precip, we'd need to diff with previous hour.
        # But we download every 6h. So we can only get 6h precip accurately?
        # NWS gives 'precipitationLastHour'.
        # For this simple model, let's skip Precip for now or just store raw TP for future diffing.
        # Let's store raw TP.
        try:
            ds = xr.open_dataset(best_file, engine='cfgrib', backend_kwargs={'filter_by_keys': {'shortName': 'tp'}})
            val = ds['tp'].sel(latitude=ALTA_LAT, longitude=ALTA_LON, method='nearest').values
            values['precip'] = float(val) # mm
            ds.close()
        except: values['precip'] = 0.0

        return {'values': values, 'run': best_run, 'fhr': best_fhr}

    except Exception as e:
        logger.error(f"Error extracting GFS: {e}")
        return None

def collect_and_store():
    """Main loop logic."""
    fetcher = NWSObservationFetcher()
    
    # Get recent observations (last 6 hours to catch up)
    obs_list = fetcher.get_recent_observations(hours=6)
    
    if not obs_list:
        logger.info("No observations found.")
        return

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    new_count = 0
    
    for obs in obs_list:
        ts = obs['timestamp']
        # Ensure UTC
        if ts.tzinfo is None: ts = ts.replace(tzinfo=timezone.utc)
        ts_utc = ts.astimezone(pytz.utc).replace(tzinfo=None) # naive UTC for simple comparison
        
        ts_str = ts_utc.isoformat()
        
        # Check if exists
        c.execute("SELECT 1 FROM training_data WHERE timestamp = ?", (ts_str,))
        if c.fetchone():
            continue
            
        # Get Obs Values
        # NWS uses 'temperature' (C), 'windSpeed' (m/s), 'precipitationLastHour' (mm)
        # If any major variable is missing, skip
        vars = obs['variables']
        if 'temperature' not in vars or 'wind_speed' not in vars:
            continue
            
        obs_temp = vars['temperature']['value']
        obs_wind = vars['wind_speed']['value']
        obs_precip = vars.get('precipitation_1h', {}).get('value', 0.0)
        
        # Get GFS Forecast
        gfs_data = get_gfs_forecast_for_time(ts_utc)
        
        if gfs_data:
            gfs_vals = gfs_data['values']
            c.execute('''
                INSERT INTO training_data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                ts_str,
                obs_temp, obs_wind, obs_precip,
                gfs_vals['temp'], gfs_vals['wind'], gfs_vals['precip'],
                gfs_data['run'], gfs_data['fhr']
            ))
            new_count += 1
            
    conn.commit()
    conn.close()
    
    if new_count > 0:
        logger.info(f"Added {new_count} new training pairs to database.")
    else:
        logger.info("No new matching data pairs found.")

def run_service():
    init_db()
    logger.info("ML Data Collector Service Started")
    
    while True:
        try:
            collect_and_store()
        except Exception as e:
            logger.error(f"Collector error: {e}")
            
        # Run every hour
        time.sleep(3600)

if __name__ == "__main__":
    run_service()
