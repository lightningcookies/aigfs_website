from flask import Flask, render_template, send_from_directory, request, jsonify
import os
import xarray as xr
import numpy as np
from datetime import datetime, timedelta
import pytz
import gc
import json

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

def utc_to_mst(date_str, hour_str):
    try:
        utc_dt = datetime.strptime(f"{date_str}{hour_str}", "%Y%m%d%H")
        utc_dt = pytz.utc.localize(utc_dt)
        mst_dt = utc_dt.astimezone(pytz.timezone('US/Mountain'))
        return mst_dt
    except:
        return datetime.now()

@app.route('/')
def index():
    maps_dir = os.path.join('static', 'maps')
    if not os.path.exists(maps_dir):
        return "No maps generated yet. Please run the backend services."
    
    files = os.listdir(maps_dir)
    # Structure: { date: { run: { region: { var: [fhrs] } } } }
    catalog = {}
    fhr_labels = {}

    for f in files:
        if f.endswith('.png') and not f.startswith('legend_'):
            parts = f.replace('.png', '').split('_')
            if len(parts) == 6:
                region, date, run, fhr, var = parts[1], parts[2], parts[3], parts[4], parts[5]
                
                if date not in catalog:
                    mst_dt = utc_to_mst(date, "00") # Use 00 as base for date label
                    catalog[date] = {'label': mst_dt.strftime("%b %d, %Y"), 'runs': {}}
                
                if run not in catalog[date]['runs']:
                    mst_run_dt = utc_to_mst(date, run)
                    catalog[date]['runs'][run] = {'label': mst_run_dt.strftime("%I %p MST"), 'fhrs': set(), 'regions': set(), 'vars': set()}
                
                catalog[date]['runs'][run]['fhrs'].add(fhr)
                catalog[date]['runs'][run]['regions'].add(region)
                catalog[date]['runs'][run]['vars'].add(var)

                # Store FHR display labels
                mst_run_dt = utc_to_mst(date, run)
                fhr_dt = mst_run_dt + timedelta(hours=int(fhr))
                fhr_labels[fhr] = fhr_dt.strftime("%a %I %p MST")

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
                          fhr_labels=fhr_labels,
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
        ds = xr.open_dataset(file_path, engine='cfgrib', backend_kwargs={'filter_by_keys': VAR_FILTERS[var]})
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

@app.route('/static/maps/<path:filename>')
def serve_map(filename):
    return send_from_directory('static/maps', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
