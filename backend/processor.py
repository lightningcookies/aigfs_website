import os
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import time
import gc
import psutil
import json
from multiprocessing import Pool, cpu_count
from datetime import datetime
from PIL import Image

# Processing settings
CLEANUP_GRIB = False
REPROCESS = False     
MAX_WORKERS = min(4, cpu_count()) 
MIN_FREE_RAM_GB = 2.0

# Region Definitions (Strict Lat/Lon Boxes)
# Updated for Web Mercator (Global max lat ~85) and Extended Pacific
REGIONS = {
    'global': {'extent': [-180, 180, -85, 85], 'max_fhr': 384},
    'conus': {'extent': [-135, -60, 20, 55], 'max_fhr': 168}, # Extended West into Pacific
    'west': {'extent': [-145, -100, 25, 50], 'max_fhr': 168}  # Extended West into Pacific
}

# NWS Style Color Configs
NWS_PRECIP_COLORS = ['#E1E1E1', '#A6F28F', '#3DBA3D', '#166B16', '#1EB5EE', '#093BB3', '#D32BE2', '#FF00F5', '#FFBA00', '#FF0000', '#B50000', '#7F0000', '#4B0000']
NWS_PRECIP_LEVELS = [0.01, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]

NWS_TEMP_COLORS = ['#4B0082', '#8A2BE2', '#0000FF', '#4169E1', '#00BFFF', '#00FFFF', '#00FF00', '#32CD32', '#FFFF00', '#FFD700', '#FFA500', '#FF4500', '#FF0000', '#B22222', '#8B0000']

# High Contrast Pressure Colors
NWS_PRESSURE_COLORS = ['#0000FF', '#4169E1', '#00BFFF', '#E0FFFF', '#FFFFE0', '#FFD700', '#FF8C00', '#FF0000', '#8B0000']
PRESSURE_LEVELS = np.arange(960, 1060, 4)

# Wind Speed Colors (NWS-ish)
# Added padding colors for <0 (impossible) and >100 (extreme) to match BoundaryNorm bins
WIND_COLORS = ['#FFFFFF', '#E0E0E0', '#B0C4DE', '#87CEFA', '#00BFFF', '#1E90FF', '#0000FF', '#8A2BE2', '#DA70D6', '#FF00FF', '#FF1493', '#8B0000', '#4B0000']
WIND_LEVELS = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100]

VAR_CONFIG = {
    't2m': {
        'cmap': mcolors.LinearSegmentedColormap.from_list('nws_temp', NWS_TEMP_COLORS),
        'levels': np.arange(-40, 121, 2), 
        'unit_conv': lambda x: (x - 273.15) * 9/5 + 32, 'unit_label': '°F',
        'key': 't2m', # Internal key to look for in loaded dataset
        'filter': {'typeOfLevel': 'heightAboveGround', 'level': 2}
    },
    'tp': {
        'cmap': mcolors.ListedColormap(NWS_PRECIP_COLORS),
        'levels': NWS_PRECIP_LEVELS,
        'unit_conv': lambda x: x / 25.4, 'unit_label': 'in',
        'key': 'tp',
        'filter': {'shortName': 'tp'}
    },
    'prmsl': {
        'cmap': mcolors.LinearSegmentedColormap.from_list('nws_pres', NWS_PRESSURE_COLORS),
        'levels': PRESSURE_LEVELS,
        'unit_conv': lambda x: x / 100.0, 'unit_label': 'hPa',
        'key': 'prmsl',
        'filter': {'shortName': 'prmsl'}
    },
    'wind_speed': {
        'cmap': mcolors.ListedColormap(WIND_COLORS), 
        'levels': WIND_LEVELS, 
        'unit_conv': lambda x: x * 2.23694, 'unit_label': 'mph',
        'key': 'wind_speed', # Calculated
        'filter': {'typeOfLevel': 'heightAboveGround', 'level': 10} # Needs u10 and v10
    }
}

def lat_to_mercator_y(lat):
    """Convert latitude to Web Mercator Y."""
    lat_rad = np.deg2rad(lat)
    return np.log(np.tan(np.pi / 4 + lat_rad / 2))

def mercator_y_to_lat(y):
    """Convert Web Mercator Y to latitude."""
    return np.rad2deg(2 * np.arctan(np.exp(y)) - np.pi / 2)

def process_file(file_path):
    """
    Process a single GRIB file:
    1. Load all variables.
    2. Compute derived variables (Wind).
    3. Generate maps for all regions and variables.
    4. Save stats.
    """
    try:
        # Check memory
        mem = psutil.virtual_memory()
        if mem.available < (MIN_FREE_RAM_GB * 1024**3): 
            print("Skipping due to low RAM")
            return False

        # Parse filename info
        # aigfs.t12z.sfc.f006.grib2
        basename = os.path.basename(file_path)
        parts = basename.split('.')
        run = parts[1][1:3]
        fhr_str = parts[3][1:]
        fhr_int = int(fhr_str)
        
        # Get date from directory name: data/20231027_12/...
        date_str = os.path.basename(os.path.dirname(file_path)).split('_')[0]
        
        output_dir = os.path.join("static", "maps")
        
        # Determine needed tasks to skip loading if all done
        tasks_needed = False
        for reg_name, reg_cfg in REGIONS.items():
            if fhr_int > reg_cfg['max_fhr']: continue
            for var_key in VAR_CONFIG.keys():
                out_filename = f"aigfs_{reg_name}_{date_str}_{run}_{fhr_str}_{var_key}.png"
                out_path = os.path.join(output_dir, out_filename)
                if REPROCESS or not os.path.exists(out_path):
                    tasks_needed = True
                    break
        
        if not tasks_needed:
            return True

        print(f"Processing {basename}...")
        
        # Open Dataset with cfgrib
        # We load specific variables to save memory, but we need multiple messages.
        # Efficient way: open once, select variables.
        # Note: cfgrib can be slow with filters on large files. 
        # We'll try opening with specific backend_kwargs for each if needed, 
        # or just open generic and select. 
        # Strategy: Open per variable group to ensure we get the right messages.
        
        # 1. Load Data
        data_cache = {}
        
        # Helper to load a var
        def load_var(filter_keys, internal_name):
            try:
                index_path = f"{file_path}.{internal_name}.{os.getpid()}.idx"
                ds = xr.open_dataset(file_path, engine='cfgrib', 
                                    backend_kwargs={'filter_by_keys': filter_keys, 'indexpath': index_path})
                if not ds.data_vars:
                    ds.close()
                    if os.path.exists(index_path): os.remove(index_path)
                    # excessive logging
                    # print(f"Variable {internal_name} not found in {basename}")
                    return

                var = list(ds.data_vars)[0]
                val = ds[var]
                # Fix Longitude (0-360 -> -180-180)
                val = val.assign_coords(longitude=(((val.longitude + 180) % 360) - 180)).sortby(['latitude', 'longitude'])
                data_cache[internal_name] = val
                ds.close()
                if os.path.exists(index_path): os.remove(index_path)
            except Exception as e:
                # print(f"Failed to load {internal_name}: {e}")
                pass

        # Load T2M, TP, PRMSL
        load_var(VAR_CONFIG['t2m']['filter'], 't2m')
        load_var(VAR_CONFIG['tp']['filter'], 'tp')
        load_var(VAR_CONFIG['prmsl']['filter'], 'prmsl')
        
        # Load U/V for Wind
        load_var({'typeOfLevel': 'heightAboveGround', 'level': 10, 'shortName': 'u10'}, 'u10')
        load_var({'typeOfLevel': 'heightAboveGround', 'level': 10, 'shortName': 'v10'}, 'v10')

        # Compute Wind Speed
        if 'u10' in data_cache and 'v10' in data_cache:
            data_cache['wind_speed'] = np.sqrt(data_cache['u10']**2 + data_cache['v10']**2)
            # Copy coords from u10
            data_cache['wind_speed'] = data_cache['wind_speed'].assign_coords(latitude=data_cache['u10'].latitude, longitude=data_cache['u10'].longitude)

        # 2. Generate Maps
        for reg_name, reg_cfg in REGIONS.items():
            if fhr_int > reg_cfg['max_fhr']: continue
            
            lon_min, lon_max, lat_min, lat_max = reg_cfg['extent']

            # Create Target Web Mercator Grid
            # We want ~1000px height for quality
            H = 1000
            # Aspect ratio based on mercator bounds
            y_min_merc = lat_to_mercator_y(lat_min)
            y_max_merc = lat_to_mercator_y(lat_max)
            x_min = lon_min
            x_max = lon_max
            
            # Aspect ratio of the box in Mercator space
            aspect = (x_max - x_min) / np.rad2deg(y_max_merc - y_min_merc) # Approx since x is deg, y is rad-like... wait. 
            # Mercator X is just Lon (in radians normally, but here we treat degree scaling).
            # Let's stick to simple:
            # Mercator Projection: x = lon, y = ln(tan(pi/4 + lat/2))
            # If lon is in degrees, we should scale y similarly or just map interpolation range.
            # We just need the TARGET LATITUDES to be spaced such that they correspond to linear Mercator Y.
            
            # Target Y grid (linear in Mercator space)
            target_merc_y = np.linspace(y_max_merc, y_min_merc, H) # Top to Bottom
            target_lat = mercator_y_to_lat(target_merc_y)
            
            # Target X grid (linear in Longitude, which is linear in Mercator)
            W = int(H * abs((lon_max - lon_min) / (np.rad2deg(y_max_merc - y_min_merc))))
            if W > 2500: W = 2500 # Cap width
            target_lon = np.linspace(lon_min, lon_max, W)

            for var_key, config in VAR_CONFIG.items():
                out_filename = f"aigfs_{reg_name}_{date_str}_{run}_{fhr_str}_{var_key}.png"
                out_path = os.path.join(output_dir, out_filename)
                json_path = out_path.replace('.png', '.json')

                if not REPROCESS and os.path.exists(out_path):
                    continue

                if config['key'] not in data_cache:
                    continue

                raw_data = data_cache[config['key']]

                # Unit Conversion
                data = config['unit_conv'](raw_data)
                
                # Crop first to reduce interpolation cost (loose crop)
                # Add buffer for interpolation
                data_crop = data.sel(latitude=slice(lat_max + 1, lat_min - 1), longitude=slice(lon_min - 1, lon_max + 1))
                if data_crop.size == 0: continue

                # Interpolate to Web Mercator Grid
                # xarray interp: latitude must be monotonic. 
                # target_lat is descending (Top to Bottom), data_crop.latitude is usually descending too.
                data_interp = data_crop.interp(latitude=target_lat, longitude=target_lon, method='linear')

                # Calculate Stats (Min/Max)
                valid_values = data_interp.values[~np.isnan(data_interp.values)]
                if len(valid_values) == 0:
                    min_val, max_val = 0, 0
                else:
                    min_val = float(np.min(valid_values))
                    max_val = float(np.max(valid_values))

                # Apply Colormap
                cmap = config['cmap']
                if isinstance(cmap, str): cmap = plt.get_cmap(cmap)
                norm = mcolors.BoundaryNorm(config['levels'], ncolors=cmap.N, extend='both')
                
                rgba_data = cmap(norm(data_interp.values))
                
                # Masking/Alpha
                if var_key == 'tp':
                    rgba_data[data_interp.values < 0.01, 3] = 0
                else:
                    # Make 'no data' or masked areas transparent? 
                    # Usually just valid data. 
                    # If wind is 0, maybe transparent? No, user wants to see low wind.
                    # Just setting alpha for overlay
                    rgba_data[..., 3] = 0.7 
                    # Optional: Mask very low wind? No.

                # Save Image
                img = Image.fromarray((rgba_data * 255).astype(np.uint8))
                img.save(out_path, "PNG")

                # Save Stats
                with open(json_path, 'w') as jf:
                    json.dump({'min': min_val, 'max': max_val, 'unit': config['unit_label']}, jf)
        
        # Cleanup
        del data_cache
        gc.collect()
        return True

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def generate_legends(output_dir):
    print("--- Generating Color Legends ---")
    for var_key, config in VAR_CONFIG.items():
        out_path = os.path.join(output_dir, f"legend_{var_key}.png")
        fig, ax = plt.subplots(figsize=(4, 0.8))
        fig.subplots_adjust(bottom=0.5)
        cmap = config['cmap']
        if isinstance(cmap, str): cmap = plt.get_cmap(cmap)
        norm = mcolors.BoundaryNorm(config['levels'], ncolors=cmap.N, extend='both')
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='horizontal',
                         ticks=config['levels'][::2] if len(config['levels']) > 15 else config['levels'], label=f"{config['unit_label']}")
        cb.ax.tick_params(labelsize=8, colors='white')
        cb.set_label(config['unit_label'], color='white', size=9)
        plt.savefig(out_path, transparent=True, bbox_inches='tight', dpi=150)
        plt.close(fig)

def run_processor_service():
    print(f"--- AIGFS Raster Processor Started ({MAX_WORKERS} Workers) ---")
    data_dir, output_dir = "data", os.path.join("static", "maps")
    os.makedirs(output_dir, exist_ok=True)
    generate_legends(output_dir)

    while True:
        # Scan for files
        files_to_process = []
        for root, dirs, files in os.walk(data_dir):
            for f in sorted(files):
                if f.endswith('.grib2'):
                    files_to_process.append(os.path.join(root, f))
        
        if files_to_process:
            print(f"\n[Parallel Cycle] Scanning {len(files_to_process)} files...")
            # We use map_async or map to process files in parallel.
            # Each worker handles one file completely (Opening once).
            with Pool(MAX_WORKERS) as pool:
                pool.map(process_file, files_to_process)
        
        # Cleanup orphan index files
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                if f.endswith('.idx') and not os.path.exists(os.path.join(root, f.replace('.idx', ''))):
                    try: os.remove(os.path.join(root, f))
                    except: pass
        
        print("Cycle complete. Sleeping...")
        time.sleep(60)

if __name__ == "__main__":
    run_processor_service()
