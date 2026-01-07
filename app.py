from flask import Flask, render_template, send_from_directory, request, jsonify
import os
import xarray as xr
import numpy as np
from datetime import datetime, timedelta
import pytz
import gc

app = Flask(__name__)

# Variable mapping for better user display
VAR_DISPLAY = {
    't2m': 'Temperature (2m)',
    'tp': 'Precipitation (6h)',
    'prmsl': 'MSL Pressure',
    'u10': 'U Wind (10m)',
    'v10': 'V Wind (10m)'
}

# Display names for regions
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
    't2m': lambda x: float(x) - 273.15,
    'prmsl': lambda x: float(x) / 100.0,
    'u10': lambda x: float(x),
    'v10': lambda x: float(x),
    'tp': lambda x: float(x)
}

def utc_to_mst(date_str, hour_str):
    """Converts UTC date/run to MST."""
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
    map_data = []
    for f in files:
        if f.endswith('.png'):
            # New format: aigfs_region_date_run_fhr_var.png
            parts = f.replace('.png', '').split('_')
            if len(parts) == 6:
                region_str, date_str, run_str, fhr_str, var_str = parts[1], parts[2], parts[3], parts[4], parts[5]
                mst_dt = utc_to_mst(date_str, run_str)
                fhr_dt = mst_dt + timedelta(hours=int(fhr_str))
                
                map_data.append({
                    'filename': f,
                    'region': region_str,
                    'region_display': REGION_DISPLAY.get(region_str, region_str.capitalize()),
                    'date': date_str,
                    'date_display': mst_dt.strftime("%b %d, %Y"),
                    'run': run_str,
                    'run_display': mst_dt.strftime("%I %p MST"),
                    'fhr': fhr_str,
                    'fhr_display': fhr_dt.strftime("%a %I %p MST"),
                    'var': var_str,
                    'var_display': VAR_DISPLAY.get(var_str, var_str.upper())
                })
    
    if not map_data:
        return "No map images found. Processing in progress..."

    # Sorting and grouping logic
    def get_unique(key, display_key=None):
        items = []
        seen = set()
        # Sort based on the key value
        for m in sorted(map_data, key=lambda x: x[key], reverse=(key == 'date')):
            if m[key] not in seen:
                items.append({'id': m[key], 'label': m[display_key] if display_key else m[key]})
                seen.add(m[key])
        return items

    regions = get_unique('region', 'region_display')
    dates = get_unique('date', 'date_display')
    runs = get_unique('run', 'run_display')
    fhrs = sorted(list(set(m['fhr'] for m in map_data)))
    vars_list = get_unique('var', 'var_display')
    
    return render_template('index.html', 
                          regions=regions,
                          dates=dates, 
                          runs=runs, 
                          fhrs=fhrs, 
                          vars=vars_list, 
                          map_data=map_data)

@app.route('/api/value')
def get_value():
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

@app.route('/static/maps/<path:filename>')
def serve_map(filename):
    return send_from_directory('static/maps', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
