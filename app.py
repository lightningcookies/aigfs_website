from flask import Flask, render_template, send_from_directory, request, jsonify
import os
import xarray as xr
import numpy as np
from datetime import datetime, timedelta
import pytz
import gc
import json
import concurrent.futures
import sqlite3

app = Flask(__name__)

VAR_DISPLAY = {
    't2m': 'Temperature (2m)',
    'tp': 'Precipitation (6h)',
    'prmsl': 'MSL Pressure',
    'u10': 'U Wind (10m)',
    'v10': 'V Wind (10m)'
}

REGION_DISPLAY = {
    'global': 'Global',
    'conus': 'USA (CONUS)',
    'west': 'Western US'
}

VAR_FILTERS = {
    't2m': {'typeOfLevel': 'heightAboveGround', 'level': 2},
    'u10': {'typeOfLevel': 'heightAboveGround', 'level': 10, 'shortName': 'u10'},
    'v10': {'typeOfLevel': 'heightAboveGround', 'level': 10, 'shortName': 'v10'},
    'prmsl': {'shortName': 'prmsl'},
    'tp': {'shortName': 'tp'}
}

UNIT_CONV = {
    't2m': lambda x: (float(x) - 273.15) * 9/5 + 32,
    'prmsl': lambda x: float(x) / 100.0,
    'u10': lambda x: float(x) * 2.23694,
    'v10': lambda x: float(x) * 2.23694,
    'tp': lambda x: float(x) / 25.4
}

def utc_to_tz(date_str, hour_str, timezone='US/Mountain'):
    try:
        utc_dt = datetime.strptime(f"{date_str}{hour_str}", "%Y%m%d%H")
        utc_dt = pytz.utc.localize(utc_dt)
        tz_dt = utc_dt.astimezone(pytz.timezone(timezone))
        return tz_dt
    except:
        return datetime.now()

# Wrapper for backward compatibility if needed, though we will update usages
def utc_to_mst(date_str, hour_str):
    return utc_to_tz(date_str, hour_str, 'US/Mountain')

@app.route('/')
def index():
    maps_dir = os.path.join('static', 'maps')
    if not os.path.exists(maps_dir):
        return "No maps generated yet. Please run the backend services."
    
    files = os.listdir(maps_dir)
    # Structure: { date: { run: { region: { var: [fhrs] } } } }
    catalog = {}

    for f in files:
        if f.endswith('.png') and not f.startswith('legend_'):
            parts = f.replace('.png', '').split('_')
            if len(parts) == 6:
                region, date, run, fhr, var = parts[1], parts[2], parts[3], parts[4], parts[5]
                
                if date not in catalog:
                    # Use 12Z (05 AM MST) as the reference for the date label
                    # This ensures the label matches the calendar day for the morning runs
                    mst_dt = utc_to_mst(date, "12") 
                    catalog[date] = {'label': mst_dt.strftime("%b %d, %Y"), 'runs': {}}
                
                if run not in catalog[date]['runs']:
                    mst_run_dt = utc_to_mst(date, run)
                    # Calculate UTC epoch for frontend calc
                    try:
                        utc_dt_obj = datetime.strptime(f"{date}{run}", "%Y%m%d%H")
                        utc_dt_obj = pytz.utc.localize(utc_dt_obj)
                        epoch = utc_dt_obj.timestamp()
                    except:
                        epoch = 0
                    
                    catalog[date]['runs'][run] = {
                        'label': mst_run_dt.strftime("%a %I %p MST"), 
                        'epoch': epoch,
                        'fhrs': set(), 'regions': set(), 'vars': set()
                    }
                
                catalog[date]['runs'][run]['fhrs'].add(fhr)
                catalog[date]['runs'][run]['regions'].add(region)
                catalog[date]['runs'][run]['vars'].add(var)

    if not catalog:
        return "No map images found."

    # Sort dates descending
    sorted_dates = sorted(catalog.keys(), reverse=True)
    
    # Get initial values for first load
    latest_date = sorted_dates[0]
    # Sort runs chronologically by actual MST time
    def run_sort_key(r):
        dt = utc_to_mst(latest_date, r)
        return dt.timestamp()
        
    available_runs = sorted(catalog[latest_date]['runs'].keys(), key=run_sort_key)
    latest_run = available_runs[0]
    
    # Convert sets to sorted lists for JSON
    for d in catalog:
        for r in catalog[d]['runs']:
            catalog[d]['runs'][r]['fhrs'] = sorted(list(catalog[d]['runs'][r]['fhrs']))
            catalog[d]['runs'][r]['regions'] = sorted(list(catalog[d]['runs'][r]['regions']))
            catalog[d]['runs'][r]['vars'] = sorted(list(catalog[d]['runs'][r]['vars']))

    return render_template('index.html', 
                          catalog=catalog,
                          sorted_dates=sorted_dates,
                          var_display=VAR_DISPLAY,
                          region_display=REGION_DISPLAY)

@app.route('/api/value')
def get_value():
    # ... (same as before)
    date = request.args.get('date')
    run = request.args.get('run')
    fhr = request.args.get('fhr')
    var = request.args.get('var')
    try:
        lat = float(request.args.get('lat'))
        lon = float(request.args.get('lon'))
        if lon < 0: lon += 360
    except:
        return jsonify({'error': 'Invalid coordinates'}), 400

    file_path = os.path.join('data', f"{date}_{run}", f"aigfs.t{run}z.sfc.f{fhr}.grib2")
    if not os.path.exists(file_path):
        return jsonify({'error': 'Data file not found'}), 404

    ds = None
    try:
        ds = xr.open_dataset(file_path, engine='cfgrib', 
                            cache=False,
                            backend_kwargs={'filter_by_keys': VAR_FILTERS[var]})
        actual_var = list(ds.data_vars)[0]
        value = ds[actual_var].sel(latitude=lat, longitude=lon, method='nearest').values
        final_value = UNIT_CONV[var](value)
        return jsonify({'value': round(final_value, 2), 'lat': lat, 'lon': lon})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if ds:
            ds.close()
            del ds
        gc.collect()

@app.route('/api/runs')
def get_available_runs():
    data_dir = 'data'
    if not os.path.exists(data_dir):
        return jsonify([])
    
    runs = []
    for d in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, d)) and '_' in d:
            runs.append(d)
    
    # Sort descending (newest first)
    runs.sort(reverse=True)
    return jsonify(runs)

@app.route('/point-analysis')
def point_analysis():
    return render_template('point_analysis.html')

def extract_grib_point(args):
    fpath, lat, lon, fhr, date_str, run_hour, timezone = args
    try:
        # T2m
        ds_t2m = xr.open_dataset(fpath, engine='cfgrib', cache=False,
                                backend_kwargs={'filter_by_keys': VAR_FILTERS['t2m']})
        t2m = float(ds_t2m['t2m'].sel(latitude=lat, longitude=lon, method='nearest').values)
        ds_t2m.close()
        
        # Wind
        ds_wind = xr.open_dataset(fpath, engine='cfgrib', cache=False,
            backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 10}})
        u10 = float(ds_wind['u10'].sel(latitude=lat, longitude=lon, method='nearest').values)
        v10 = float(ds_wind['v10'].sel(latitude=lat, longitude=lon, method='nearest').values)
        ds_wind.close()
        
        # TP
        ds_tp = xr.open_dataset(fpath, engine='cfgrib', cache=False,
                                backend_kwargs={'filter_by_keys': VAR_FILTERS['tp']})
        tp = float(ds_tp['tp'].sel(latitude=lat, longitude=lon, method='nearest').values)
        ds_tp.close()
        
        # Conversions
        t2m_f = UNIT_CONV['t2m'](t2m)
        wind_mph = UNIT_CONV['u10'](np.sqrt(u10**2 + v10**2))
        tp_in = UNIT_CONV['tp'](tp)
        
        valid_time = utc_to_tz(date_str, run_hour, timezone) + timedelta(hours=fhr)
        
        res = {
            'fhr': fhr,
            'time': valid_time.isoformat(),
            't2m': round(t2m_f, 1),
            'wind': round(wind_mph, 1),
            'tp_val': tp_in # Raw value, accumulated later
        }
        
        # Cleanup local references
        del ds_t2m, ds_wind, ds_tp
        
        return res
    except Exception as e:
        return None

@app.route('/api/point-data')
def get_point_data():
    try:
        lat = float(request.args.get('lat'))
        lon = float(request.args.get('lon'))
        if lon < 0: lon += 360
        timezone = request.args.get('timezone', 'US/Mountain')
    except:
        return jsonify({'error': 'Invalid coordinates'}), 400

    # Determine runs to process
    data_dir = 'data'
    if not os.path.exists(data_dir):
        return jsonify({'error': 'No data available'}), 404

    requested_runs = request.args.get('runs')
    
    if requested_runs:
        selected_runs = requested_runs.split(',')
    else:
        # Default to last 3
        all_runs = []
        for d in os.listdir(data_dir):
            if os.path.isdir(os.path.join(data_dir, d)) and '_' in d:
                all_runs.append(d)
        all_runs.sort(reverse=True)
        selected_runs = all_runs[:3]

    result = {'runs': []}
    
    # Process each run
    for run_dir in selected_runs:
        if not os.path.exists(os.path.join(data_dir, run_dir)):
            continue
            
        date_str, run_hour = run_dir.split('_')
        run_label = f"{date_str} {run_hour}Z"
        run_path = os.path.join(data_dir, run_dir)
        
        # Identify files
        tasks = []
        for f in os.listdir(run_path):
            if f.endswith('.grib2'):
                try:
                    fhr = int(f.split('.f')[-1].replace('.grib2', ''))
                    # Optimization: Skip intermediate hours for long range
                    if fhr > 120 and fhr % 12 != 0:
                        continue
                        
                    tasks.append((os.path.join(run_path, f), lat, lon, fhr, date_str, run_hour, timezone))
                except: pass
        
        # Execute parallel reads - Reduced workers to save RAM and prevent swap spikes
        run_points = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            results = executor.map(extract_grib_point, tasks)
            
            for res in results:
                if res:
                    run_points.append(res)
        
        # Sort by forecast hour
        run_points.sort(key=lambda x: x['fhr'])
        
        # Calculate Accumulation
        final_data = []
        current_tp_accum = 0.0
        
        for p in run_points:
            current_tp_accum += p['tp_val']
            final_data.append({
                'time': p['time'],
                't2m': p['t2m'],
                'wind': p['wind'],
                'tp_acum': round(current_tp_accum, 2)
            })
            
        if final_data:
            result['runs'].append({'name': run_label, 'data': final_data})
        
        # AGGRESSIVE CLEANUP: Clear references and force collection after each run
        del run_points
        del tasks
        gc.collect()

    return jsonify(result)

import sqlite3

# ... existing code ...

@app.route('/api/alta-ml')
def get_alta_ml_forecast():
    # 1. Get coefficients
    db_path = os.path.join("backend", "ml_data.db")
    coeffs = {}
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            c.execute("SELECT variable, slope, intercept, rmse FROM model_coefficients")
            for row in c.fetchall():
                coeffs[row[0]] = {'slope': row[1], 'intercept': row[2], 'rmse': row[3]}
            conn.close()
        except: pass

    # 2. Get Raw Forecast (using existing point logic, but hardcoded for Alta)
    # Alta Coords
    lat, lon = 40.57, -111.63
    
    # Re-use logic from get_point_data but simplified for latest run only
    # ... logic similar to get_point_data ...
    # For brevity, we will call the internal logic if we refactored, but here we duplicate slightly for speed
    
    data_dir = 'data'
    runs = []
    if os.path.exists(data_dir):
        for d in os.listdir(data_dir):
            if '_' in d: runs.append(d)
    runs.sort(reverse=True)
    
    if not runs:
        return jsonify({'error': 'No GFS data'})
        
    latest_run = runs[0]
    date_str, run_hour = latest_run.split('_')
    run_path = os.path.join(data_dir, latest_run)
    
    raw_data = []
    
    # Get files
    files = []
    for f in os.listdir(run_path):
        if f.endswith('.grib2'):
            try:
                fhr = int(f.split('.f')[-1].replace('.grib2', ''))
                files.append((fhr, os.path.join(run_path, f)))
            except: pass
    files.sort()
    
    # Process
    for fhr, fpath in files:
        if fhr > 120 and fhr % 12 != 0: continue
        
        try:
            # T2m
            ds = xr.open_dataset(fpath, engine='cfgrib', cache=False,
                                backend_kwargs={'filter_by_keys': VAR_FILTERS['t2m']})
            t2m_k = float(ds['t2m'].sel(latitude=lat, longitude=lon, method='nearest').values)
            ds.close()
            
            # Wind
            ds = xr.open_dataset(fpath, engine='cfgrib', cache=False,
                                backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 10}})
            u = float(ds['u10'].sel(latitude=lat, longitude=lon, method='nearest').values)
            v = float(ds['v10'].sel(latitude=lat, longitude=lon, method='nearest').values)
            ds.close()
            
            t2m_c = t2m_k - 273.15
            wind_ms = np.sqrt(u**2 + v**2)
            
            # Apply Correction
            corrected_t_c = t2m_c
            corrected_w_ms = wind_ms
            
            if 'temperature' in coeffs:
                # Corrected = Slope * Raw + Intercept
                corrected_t_c = (coeffs['temperature']['slope'] * t2m_c) + coeffs['temperature']['intercept']
                
            if 'wind_speed' in coeffs:
                corrected_w_ms = (coeffs['wind_speed']['slope'] * wind_ms) + coeffs['wind_speed']['intercept']
            
            # Convert to Display Units (F, MPH)
            t_raw_f = (t2m_c * 9/5) + 32
            t_corr_f = (corrected_t_c * 9/5) + 32
            
            w_raw_mph = wind_ms * 2.237
            w_corr_mph = corrected_w_ms * 2.237
            
            valid_time = utc_to_mst(date_str, run_hour) + timedelta(hours=fhr)
            
            raw_data.append({
                'time': valid_time.isoformat(),
                'temp_raw': round(t_raw_f, 1),
                'temp_corrected': round(t_corr_f, 1),
                'wind_raw': round(w_raw_mph, 1),
                'wind_corrected': round(w_corr_mph, 1)
            })
            
        except: pass
        
    return jsonify({
        'data': raw_data,
        'model_info': coeffs
    })

@app.route('/alta-ml')
def alta_ml_page():
    return render_template('alta_ml.html')

@app.route('/static/maps/<path:filename>')
def serve_map(filename):
    return send_from_directory('static/maps', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
