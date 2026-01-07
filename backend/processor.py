import os
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import time
import gc
import psutil
from multiprocessing import Pool, cpu_count
from datetime import datetime
from PIL import Image

# Processing settings
CLEANUP_GRIB = False
REPROCESS = False     
MAX_FILES_PER_CYCLE = 20 
MAX_WORKERS = min(4, cpu_count()) 
MIN_FREE_RAM_GB = 2.0

# Region Definitions (Strict Lat/Lon Boxes)
REGIONS = {
    'global': {'extent': [-180, 180, -90, 90], 'max_fhr': 384},
    'conus': {'extent': [-125, -66, 23, 50], 'max_fhr': 168},
    'west': {'extent': [-126, -103, 31, 50], 'max_fhr': 168}
}

# NWS Style Color Config
NWS_PRECIP_COLORS = ['#E1E1E1', '#A6F28F', '#3DBA3D', '#166B16', '#1EB5EE', '#093BB3', '#D32BE2', '#FF00F5', '#FFBA00', '#FF0000', '#B50000', '#7F0000', '#4B0000']
NWS_PRECIP_LEVELS = [0.01, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]
NWS_TEMP_COLORS = ['#4B0082', '#8A2BE2', '#0000FF', '#4169E1', '#00BFFF', '#00FFFF', '#00FF00', '#32CD32', '#FFFF00', '#FFD700', '#FFA500', '#FF4500', '#FF0000', '#B22222', '#8B0000']

VAR_CONFIG = {
    't2m': {
        'cmap': mcolors.LinearSegmentedColormap.from_list('nws_temp', NWS_TEMP_COLORS),
        'levels': np.arange(-40, 121, 2), 
        'unit_conv': lambda x: (x - 273.15) * 9/5 + 32, 'unit_label': '°F',
        'filter': {'typeOfLevel': 'heightAboveGround', 'level': 2}
    },
    'tp': {
        'cmap': mcolors.ListedColormap(NWS_PRECIP_COLORS),
        'levels': NWS_PRECIP_LEVELS,
        'unit_conv': lambda x: x / 25.4, 'unit_label': 'in',
        'filter': {'shortName': 'tp'}
    },
    'prmsl': {
        'cmap': 'coolwarm', 'levels': np.arange(960, 1051, 2),
        'unit_conv': lambda x: x / 100.0, 'unit_label': 'hPa',
        'filter': {'shortName': 'prmsl'}
    },
    'u10': { 'cmap': 'viridis', 'levels': np.arange(-60, 61, 10), 'unit_conv': lambda x: x * 2.23694, 'unit_label': 'mph', 'filter': {'typeOfLevel': 'heightAboveGround', 'level': 10} },
    'v10': { 'cmap': 'viridis', 'levels': np.arange(-60, 61, 10), 'unit_conv': lambda x: x * 2.23694, 'unit_label': 'mph', 'filter': {'typeOfLevel': 'heightAboveGround', 'level': 10} }
}

def generate_map_task(args):
    file_path, output_dir, region_name, var_key, date_str, run, fhr_str = args
    config = VAR_CONFIG[var_key]
    reg_config = REGIONS[region_name]
    out_filename = f"aigfs_{region_name}_{date_str}_{run}_{fhr_str}_{var_key}.png"
    out_path = os.path.join(output_dir, out_filename)

    if os.path.exists(out_path) and not REPROCESS:
        return False

    ds = None
    try:
        mem = psutil.virtual_memory()
        if mem.available < (MIN_FREE_RAM_GB * 1024**3): return "LOW_RAM"

        # Unique index per file AND variable to prevent cfgrib conflicts
        index_path = f"{file_path}.{var_key}.{os.getpid()}.idx"
        
        ds = xr.open_dataset(
            file_path, 
            engine='cfgrib', 
            backend_kwargs={
                'filter_by_keys': config['filter'], 
                'indexpath': index_path
            }
        )
        
        # 1. Coordinate Wrapping & Conversion
        if not ds.data_vars:
            raise ValueError(f"No variables found matching filter {config['filter']}")
            
        # Find the actual variable name (e.g., 't2m', 'u', 'v10', etc.)
        actual_var = None
        for v in ds.data_vars:
            if var_key in v or v in ['u', 'v', 'gh', 'prmsl', 'tp']:
                actual_var = v
                break
        if not actual_var: actual_var = list(ds.data_vars)[0]

        data = config['unit_conv'](ds[actual_var])
        data = data.assign_coords(longitude=(((data.longitude + 180) % 360) - 180)).sortby(['latitude', 'longitude'])

        # 2. Strict Regional Cropping
        lon_min, lon_max, lat_min, lat_max = reg_config['extent']
        data = data.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))

        # 3. Interpolate to a fixed high-res grid (removes alignment errors)
        # This ensures the array exactly fits the output image size
        target_lat = np.linspace(lat_max, lat_min, 1000)
        target_lon = np.linspace(lon_min, lon_max, 2000)
        data_interp = data.interp(latitude=target_lat, longitude=target_lon, method='linear')

        # 4. Apply Colormap manually
        cmap = config['cmap']
        if isinstance(cmap, str): cmap = plt.get_cmap(cmap)
        
        # Normalize data to 0-1 based on fixed levels
        norm = mcolors.BoundaryNorm(config['levels'], ncolors=cmap.N, extend='both')
        
        # Mask Precipitation
        if var_key == 'tp':
            rgba_data = cmap(norm(data_interp.values))
            rgba_data[data_interp.values < 0.01, 3] = 0 # Set alpha to 0 for no rain
        else:
            rgba_data = cmap(norm(data_interp.values))
            rgba_data[..., 3] = 0.7 # Set global transparency for other layers

        # 5. Save directly as Image (No Matplotlib Axes!)
        img = Image.fromarray((rgba_data * 255).astype(np.uint8))
        img.save(out_path, "PNG")

        ds.close()
        if os.path.exists(index_path): os.remove(index_path)
        gc.collect()
        return True
    except Exception as e:
        print(f"  [ERROR] Failed {out_filename}: {e}")
        if ds: ds.close()
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
                         ticks=config['levels'][::2] if len(config['levels']) > 10 else config['levels'], label=f"{config['unit_label']}")
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
        tasks = []
        for root, dirs, files in os.walk(data_dir):
            for f in sorted(files):
                if f.endswith('.grib2'):
                    file_path = os.path.join(root, f)
                    parts = f.split('.')
                    run, fhr_int, fhr_str = parts[1][1:3], int(parts[3][1:]), parts[3][1:]
                    date_str = os.path.basename(os.path.dirname(file_path)).split('_')[0]
                    for reg_name, reg_cfg in REGIONS.items():
                        if fhr_int <= reg_cfg['max_fhr']:
                            for v_key in VAR_CONFIG.keys():
                                if not os.path.exists(os.path.join(output_dir, f"aigfs_{reg_name}_{date_str}_{run}_{fhr_str}_{v_key}.png")) or REPROCESS:
                                    tasks.append((file_path, output_dir, reg_name, v_key, date_str, run, fhr_str))
                if len(tasks) >= 100: break
            if len(tasks) >= 100: break
        
        if tasks:
            print(f"\n[Parallel Cycle] Generating {len(tasks)} raster maps...")
            with Pool(MAX_WORKERS) as pool: pool.map(generate_map_task, tasks)
        
        # Cleanup orphan index files
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                if f.endswith('.idx') and not os.path.exists(os.path.join(root, f.replace('.idx', ''))):
                    try: os.remove(os.path.join(root, f))
                    except: pass
        time.sleep(60 if tasks else 300)

if __name__ == "__main__":
    run_processor_service()
