from flask import Flask, render_template, send_from_directory, request, jsonify
import os
import xarray as xr
import numpy as np
from datetime import datetime, timedelta
import pytz

app = Flask(__name__)

# Variable mapping for better user display
VAR_DISPLAY = {
    't2m': 'Temperature (2m)',
    'tp': 'Precipitation (6h)',
    'prmsl': 'MSL Pressure',
    'u10': 'U Wind (10m)',
    'v10': 'V Wind (10m)'
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
    utc_dt = datetime.strptime(f"{date_str}{hour_str}", "%Y%m%d%H")
    utc_dt = pytz.utc.localize(utc_dt)
    mst_dt = utc_dt.astimezone(pytz.timezone('US/Mountain'))
    return mst_dt

@app.route('/')
def index():
    maps_dir = os.path.join('static', 'maps')
    if not os.path.exists(maps_dir):
        return "No maps generated yet. Please run the backend services."
    
    files = os.listdir(maps_dir)
    map_data = []
    for f in files:
        if f.endswith('.png'):
            parts = f.replace('.png', '').split('_')
            if len(parts) == 5:
                date_str = parts[1]
                run_str = parts[2]
                fhr_str = parts[3]
                var_str = parts[4]
                
                # Create MST display labels
                mst_dt = utc_to_mst(date_str, run_str)
                # Forecast time in MST
                fhr_dt = mst_dt + timedelta(hours=int(fhr_str))
                
                map_data.append({
                    'filename': f,
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
        return "No valid map images found."

    # Sorting logic for the UI
    # We use a dict to get unique values but preserve the original ID for the selector
    dates = []
    seen_dates = set()
    for m in sorted(map_data, key=lambda x: x['date'], reverse=True):
        if m['date'] not in seen_dates:
            dates.append({'id': m['date'], 'label': m['date_display']})
            seen_dates.add(m['date'])

    runs = []
    seen_runs = set()
    for m in sorted(map_data, key=lambda x: x['run']):
        if m['run'] not in seen_runs:
            runs.append({'id': m['run'], 'label': m['run_display']})
            seen_runs.add(m['run'])

    fhrs = sorted(list(set(m['fhr'] for m in map_data)))
    
    vars_list = []
    seen_vars = set()
    for m in sorted(map_data, key=lambda x: x['var']):
        if m['var'] not in seen_vars:
            vars_list.append({'id': m['var'], 'label': m['var_display']})
            seen_vars.add(m['var'])
    
    return render_template('index.html', dates=dates, runs=runs, fhrs=fhrs, vars=vars_list, map_data=map_data)

@app.route('/api/value')
def get_value():
    """
    Returns the exact value from a GRIB file for a given lat/lon.
    """
    date = request.args.get('date')
    run = request.args.get('run')
    fhr = request.args.get('fhr')
    var = request.args.get('var')
    lat = float(request.args.get('lat'))
    lon = float(request.args.get('lon'))

    # Normalize longitude to 0-360 if needed (GRIB standard)
    if lon < 0:
        lon += 360

    file_path = os.path.join('data', f"{date}_{run}", f"aigfs.t{run}z.sfc.f{fhr}.grib2")
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'Data file not found'}), 404

    try:
        ds = xr.open_dataset(file_path, engine='cfgrib', 
                            backend_kwargs={'filter_by_keys': VAR_FILTERS[var]})
        
        actual_var = list(ds.data_vars)[0]
        # Find nearest point
        value = ds[actual_var].sel(latitude=lat, longitude=lon, method='nearest').values
        ds.close()

        # Convert units
        final_value = UNIT_CONV[var](value)
        
        return jsonify({
            'value': round(final_value, 2),
            'lat': lat,
            'lon': lon
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/static/maps/<path:filename>')
def serve_map(filename):
    return send_from_directory('static/maps', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
