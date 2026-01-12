import os
import sys
import sqlite3
import time
import logging
from datetime import datetime, timedelta, timezone
import numpy as np
import xarray as xr
import pytz

# Add parent directory to path to import observation_fetcher
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from observation_fetcher import NWSObservationFetcher

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_data.db")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
EXAMPLE_FILE = os.path.join(BASE_DIR, "aigfs_example.grib2")

# Alta/Collins Coordinates (Approximate for AIGFS extraction)
ALTA_LAT = 40.57  # Collins is around here
ALTA_LON = -111.63

# NWS to AIGFS Variable Mapping
# AIGFS keys are based on our cfgrib/xarray loading
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
            obs_u10 REAL,
            obs_v10 REAL,
            obs_pressure REAL,
            obs_precip_1h REAL,
            obs_precip_6h REAL,
            
            gfs_temp REAL,
            gfs_u10 REAL,
            gfs_v10 REAL,
            gfs_pressure REAL,
            gfs_tp_accum REAL,
            
            gfs_run_date TEXT,
            gfs_fhr INTEGER
        )
    ''')
    
    # Check if we need to add new columns to existing table (simple migration)
    c.execute("PRAGMA table_info(training_data)")
    columns = [col[1] for col in c.fetchall()]
    
    new_cols = {
        'obs_u10': 'REAL',
        'obs_v10': 'REAL',
        'obs_pressure': 'REAL',
        'obs_precip_1h': 'REAL',
        'obs_precip_6h': 'REAL',
        'gfs_temp': 'REAL',
        'gfs_u10': 'REAL',
        'gfs_v10': 'REAL',
        'gfs_pressure': 'REAL',
        'gfs_tp_accum': 'REAL'
    }
    
    for col_name, col_type in new_cols.items():
        if col_name not in columns:
            logger.info(f"Adding column {col_name} to training_data")
            c.execute(f"ALTER TABLE training_data ADD COLUMN {col_name} {col_type}")
    
    conn.commit()
    conn.close()

def extract_from_grib(fpath, run_name, fhr):
    """Logic to read values from an AIGFS GRIB2 file at Alta's coordinates."""
    if not os.path.exists(fpath):
        return None
        
    try:
        values = {}
        # AIGFS longitudes are 0-360. Convert if needed.
        grib_lon = ALTA_LON if ALTA_LON >= 0 else ALTA_LON + 360

        # T2m (2m Temperature)
        try:
            ds = xr.open_dataset(fpath, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 2}})
            val = ds['t2m'].sel(latitude=ALTA_LAT, longitude=grib_lon, method='nearest').values
            values['temp'] = float(val) - 273.15 # K to C
            ds.close()
        except Exception as e:
            logger.debug(f"Could not extract t2m from {fpath}: {e}")
            values['temp'] = None
        
        # Wind (10m U/V)
        try:
            ds = xr.open_dataset(fpath, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 10}})
            u = ds['u10'].sel(latitude=ALTA_LAT, longitude=grib_lon, method='nearest').values
            v = ds['v10'].sel(latitude=ALTA_LAT, longitude=grib_lon, method='nearest').values
            values['u10'] = float(u)
            values['v10'] = float(v)
            ds.close()
        except Exception as e:
            logger.debug(f"Could not extract wind from {fpath}: {e}")
            values['u10'] = None
            values['v10'] = None

        # Pressure (Mean Sea Level)
        try:
            ds = xr.open_dataset(fpath, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'meanSea'}})
            val = ds['prmsl'].sel(latitude=ALTA_LAT, longitude=grib_lon, method='nearest').values
            values['pressure'] = float(val) # Pa
            ds.close()
        except Exception as e:
            logger.debug(f"Could not extract pressure from {fpath}: {e}")
            values['pressure'] = None

        # Precip (Total Precipitation - Accumulated)
        try:
            ds = xr.open_dataset(fpath, engine='cfgrib', backend_kwargs={'filter_by_keys': {'shortName': 'tp'}})
            val = ds['tp'].sel(latitude=ALTA_LAT, longitude=grib_lon, method='nearest').values
            values['tp_accum'] = float(val) # mm
            ds.close()
        except Exception as e:
            logger.debug(f"Could not extract tp from {fpath}: {e}")
            values['tp_accum'] = 0.0

        return {'values': values, 'run': run_name, 'fhr': fhr}

    except Exception as e:
        logger.error(f"Error extracting from GRIB {fpath}: {e}")
        return None

def get_aigfs_forecast_for_time(target_time_utc):
    """
    Finds the most recent AIGFS run and forecast hour that covers the target time.
    Returns dictionary of AIGFS values or None.
    """
    # 1. Check for the example file in the root directory first
    if os.path.exists(EXAMPLE_FILE):
        try:
            ds = xr.open_dataset(EXAMPLE_FILE, engine='cfgrib')
            valid_time = ds.valid_time.values
            ds.close()

            if isinstance(valid_time, np.datetime64):
                vt_dt = datetime.utcfromtimestamp(valid_time.astype('O') / 1e9)
            else:
                vt_dt = valid_time
            
            if abs((target_time_utc - vt_dt).total_seconds()) < 2700:
                logger.info(f"MATCH: Using example file {EXAMPLE_FILE} for valid time {vt_dt}")
                return extract_from_grib(EXAMPLE_FILE, "example_run", 324) 
        except Exception as e:
            logger.debug(f"Example file check skipped: {e}")

    if not os.path.exists(DATA_DIR):
        logger.debug(f"Data directory {DATA_DIR} not found.")
        return None

    # Scan all available runs
    runs = []
    for d in os.listdir(DATA_DIR):
        if '_' in d:
            try:
                dt_str, hr_str = d.split('_')
                run_dt = datetime.strptime(f"{dt_str}{hr_str}", "%Y%m%d%H")
                runs.append((run_dt, d))
            except: pass
            
    runs.sort(reverse=True) # Newest first
    
    for run_dt, run_dir_name in runs:
        if run_dt > target_time_utc:
            continue
            
        diff_hours = (target_time_utc - run_dt).total_seconds() / 3600
        fhr = int(6 * round(diff_hours / 6))
        
        if abs(diff_hours - fhr) > 3.0:
            continue
            
        fname = f"aigfs.t{run_dt.strftime('%H')}z.sfc.f{fhr:03d}.grib2"
        fpath = os.path.join(DATA_DIR, run_dir_name, fname)
        
        if os.path.exists(fpath):
            logger.info(f"MATCH: Found {fpath} for observation at {target_time_utc}")
            return extract_from_grib(fpath, run_dir_name, fhr)
            
    return None

def collect_and_store(start_date=None):
    """
    Main loop logic.
    If start_date is provided (datetime object), fetches history from that date.
    Otherwise fetches recent (last 6 hours).
    """
    fetcher = NWSObservationFetcher()
    
    obs_list = []
    
    if start_date:
        logger.info(f"Backfilling data from {start_date}...")
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
            
        current_start = start_date
        now = datetime.now(timezone.utc)
        
        while current_start < now:
            next_end = min(current_start + timedelta(days=3), now)
            logger.info(f"Fetching chunk: {current_start.date()} to {next_end.date()}")
            chunk = fetcher.get_observations(start_time=current_start, end_time=next_end, limit=500) 
            logger.info(f"Fetched {len(chunk)} observations in this chunk.")
            obs_list.extend(chunk)
            current_start = next_end
            time.sleep(1) 
            
    else:
        logger.info("Fetching recent observations...")
        obs_list = fetcher.get_recent_observations(hours=6)
        logger.info(f"Fetched {len(obs_list)} recent observations.")
    
    if not obs_list:
        logger.info("No observations found from NWS API. Check station ID or connection.")
        return

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    new_count = 0
    match_attempts = 0
    
    for obs in obs_list:
        ts = obs['timestamp']
        if ts.tzinfo is None: ts = ts.replace(tzinfo=timezone.utc)
        ts_utc = ts.astimezone(pytz.utc).replace(tzinfo=None) 
        
        ts_str = ts_utc.isoformat()
        
        # Check if exists
        c.execute("SELECT 1 FROM training_data WHERE timestamp = ?", (ts_str,))
        if c.fetchone():
            continue
            
        match_attempts += 1
        # Get Obs Values
        vars = obs['variables']
        if 'temperature' not in vars or 'wind_speed' not in vars:
            logger.debug(f"Skipping obs at {ts_str}: Missing temp or wind")
            continue
            
        obs_temp = vars['temperature']['value']
        obs_wind_speed = vars['wind_speed']['value']
        obs_wind_dir = vars.get('wind_direction', {}).get('value')
        
        # Calculate U/V components
        obs_u10 = None
        obs_v10 = None
        if obs_wind_speed is not None and obs_wind_dir is not None:
            rad = np.radians(obs_wind_dir)
            obs_u10 = -obs_wind_speed * np.sin(rad)
            obs_v10 = -obs_wind_speed * np.cos(rad)
            
        obs_pressure = vars.get('sea_level_pressure', {}).get('value')
        obs_precip_1h = vars.get('precipitation_1h', {}).get('value', 0.0)
        obs_precip_6h = vars.get('precipitation_6h', {}).get('value', 0.0)
        
        # Get AIGFS Forecast
        aigfs_data = get_aigfs_forecast_for_time(ts_utc)
        
        if aigfs_data:
            gfs_vals = aigfs_data['values']
            try:
                c.execute('''
                    INSERT INTO training_data (
                        timestamp, 
                        obs_temp, obs_u10, obs_v10, obs_pressure, obs_precip_1h, obs_precip_6h,
                        gfs_temp, gfs_u10, gfs_v10, gfs_pressure, gfs_tp_accum,
                        gfs_run_date, gfs_fhr
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    ts_str,
                    obs_temp, obs_u10, obs_v10, obs_pressure, obs_precip_1h, obs_precip_6h,
                    gfs_vals['temp'], gfs_vals['u10'], gfs_vals['v10'], gfs_vals['pressure'], gfs_vals['tp_accum'],
                    aigfs_data['run'], aigfs_data['fhr']
                ))
                new_count += 1
            except Exception as e:
                logger.error(f"Database insertion error at {ts_str}: {e}")
        else:
            logger.debug(f"No AIGFS coverage for observation at {ts_str}")
            
    conn.commit()
    conn.close()
    
    if new_count > 0:
        logger.info(f"SUCCESS: Added {new_count} new training pairs to database.")
    else:
        logger.info(f"No new pairs added. (Checked {match_attempts} new observations, but no AIGFS files matched).")

def run_service():
    init_db()
    logger.info("ML Data Collector Service Started")
    
    # Initial Backfill Check
    # If DB is empty, or user requested, we backfill from 2026-01-01
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM training_data")
    count = c.fetchone()[0]
    conn.close()
    
    if count == 0:
        logger.info("Database empty. Starting backfill from Jan 1, 2026...")
        backfill_start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        try:
            collect_and_store(start_date=backfill_start)
        except Exception as e:
            logger.error(f"Backfill failed: {e}")
    
    while True:
        try:
            collect_and_store()
        except Exception as e:
            logger.error(f"Collector error: {e}")
            
        # Run every hour
        time.sleep(3600)

if __name__ == "__main__":
    run_service()
