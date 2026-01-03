from flask import Flask, render_template, send_from_directory
import os

app = Flask(__name__)

@app.route('/')
def index():
    maps_dir = os.path.join('static', 'maps')
    if not os.path.exists(maps_dir):
        return "No maps generated yet. Please run: python run_all.py"
    
    files = os.listdir(maps_dir)
    map_data = []
    for f in files:
        if f.endswith('.png'):
            parts = f.replace('.png', '').split('_')
            # Handle both old (4 parts) and new (5 parts) formats
            if len(parts) == 5:
                map_data.append({
                    'filename': f,
                    'date': parts[1],
                    'run': parts[2],
                    'fhr': parts[3],
                    'var': parts[4]
                })
            elif len(parts) == 4:
                map_data.append({
                    'filename': f,
                    'date': '20260103', # Default for old files
                    'run': parts[1],
                    'fhr': parts[2],
                    'var': parts[3]
                })
    
    if not map_data:
        return "No valid map images found. Please run: python run_all.py"

    # Sort and get unique values
    dates = sorted(list(set(m['date'] for m in map_data)), reverse=True)
    runs = sorted(list(set(m['run'] for m in map_data)))
    fhrs = sorted(list(set(m['fhr'] for m in map_data)))
    vars_list = sorted(list(set(m['var'] for m in map_data)))
    
    return render_template('index.html', dates=dates, runs=runs, fhrs=fhrs, vars=vars_list, map_data=map_data)

@app.route('/static/maps/<path:filename>')
def serve_map(filename):
    return send_from_directory('static/maps', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
