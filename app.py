from flask import Flask, render_template, send_from_directory
import os

app = Flask(__name__)

# Route for the main page
@app.route('/')
def index():
    # Get list of available maps to send to frontend
    maps_dir = os.path.join('static', 'maps')
    if not os.path.exists(maps_dir):
        return "No maps generated yet. Please run the backend scripts first."
    
    files = os.listdir(maps_dir)
    # Filter and sort files (example: aigfs_00_000_t2m.png)
    # We can extract run, fhr, and variable from filenames
    map_data = []
    for f in files:
        if f.endswith('.png'):
            parts = f.replace('.png', '').split('_')
            if len(parts) == 4:
                map_data.append({
                    'filename': f,
                    'run': parts[1],
                    'fhr': parts[2],
                    'var': parts[3]
                })
    
    # Get unique runs, fhrs, and vars for selectors
    runs = sorted(list(set(m['run'] for m in map_data)))
    fhrs = sorted(list(set(m['fhr'] for m in map_data)))
    vars_list = sorted(list(set(m['var'] for m in map_data)))
    
    return render_template('index.html', runs=runs, fhrs=fhrs, vars=vars_list, map_data=map_data)

# Route to serve static files (maps)
@app.route('/static/maps/<path:filename>')
def serve_map(filename):
    return send_from_directory('static/maps', filename)

if __name__ == '__main__':
    # Listen on all interfaces so it's accessible on the local network
    app.run(host='0.0.0.0', port=5000, debug=True)

