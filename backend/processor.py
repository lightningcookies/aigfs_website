import os
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import time
import gc
import psutil
from multiprocessing import Pool, cpu_count
from datetime import datetime

# Processing settings
CLEANUP_GRIB = False
REPROCESS = False     
MAX_FILES_PER_CYCLE = 20 
MAX_WORKERS = min(4, cpu_count()) 
MIN_FREE_RAM_GB = 2.0

# Region Definitions
REGIONS = {
    'global': {'extent': [-180, 180, -90, 90], 'max_fhr': 384, 'dpi': 120},
    'conus': {'extent': [-125, -66, 23, 50], 'max_fhr': 168, 'dpi': 180},
    'west': {'extent': [-126, -103, 31, 50], 'max_fhr': 168, 'dpi': 220}
}

# --- FIXED NWS STYLE COLORMAPS ---
NWS_PRECIP_COLORS = [
    '#E1E1E1', '#A6F28F', '#3DBA3D', '#166B16', '#1EB5EE', '#093BB3', 
    '#D32BE2', '#FF00F5', '#FFBA00', '#FF0000', '#B50000', '#7F0000', '#4B0000'
]
NWS_PRECIP_LEVELS = [0.01, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]

NWS_TEMP_COLORS = [
    '#4B0082', '#8A2BE2', '#0000FF', '#4169E1', '#00BFFF', '#00FFFF', 
    '#00FF00', '#32CD32', '#FFFF00', '#FFD700', '#FFA500', '#FF4500', 
    '#FF0000', '#B22222', '#8B0000'
]

VAR_CONFIG = {
    't2m': {
        'label': 'Temperature (2m)', 
        'cmap': mcolors.LinearSegmentedColormap.from_list('nws_temp', NWS_TEMP_COLORS),
        'levels': np.arange(-40, 121, 2), 
        'unit_conv': lambda x: (x - 273.15) * 9/5 + 32, 'unit_label': '°F',
        'filter': {'typeOfLevel': 'heightAboveGround', 'level': 2}
    },
    'u10': {
        'label': 'U Wind (10m)', 'cmap': 'viridis', 'levels': np.arange(-60, 61, 10),
        'unit_conv': lambda x: x * 2.23694, 'unit_label': 'mph',
        'filter': {'typeOfLevel': 'heightAboveGround', 'level': 10, 'shortName': 'u10'}
    },
    'v10': {
        'label': 'V Wind (10m)', 'cmap': 'viridis', 'levels': np.arange(-60, 61, 10),
        'unit_conv': lambda x: x * 2.23694, 'unit_label': 'mph',
        'filter': {'typeOfLevel': 'heightAboveGround', 'level': 10, 'shortName': 'v10'}
    },
    'prmsl': {
        'label': 'MSL Pressure', 'cmap': 'coolwarm', 'levels': np.arange(960, 1051, 2),
        'unit_conv': lambda x: x / 100.0, 'unit_label': 'hPa',
        'filter': {'shortName': 'prmsl'}
    },
    'tp': {
        'label': 'Total Precipitation (6h)', 
        'cmap': mcolors.ListedColormap(NWS_PRECIP_COLORS),
        'levels': NWS_PRECIP_LEVELS,
        'unit_conv': lambda x: x / 25.4, 'unit_label': 'in',
        'filter': {'shortName': 'tp'}
    }
}

def generate_map_task(args):
    file_path, output_dir, region_name, var_key, date_str, run, fhr_str = args
    config = VAR_CONFIG[var_key]
    reg_config = REGIONS[region_name]
    out_filename = f"aigfs_{region_name}_{date_str}_{run}_{fhr_str}_{var_key}.png"
    out_path = os.path.join(output_dir, out_filename)

    if os.path.exists(out_path) and not REPROCESS:
        return False

    try:
        mem = psutil.virtual_memory()
        if mem.available < (MIN_FREE_RAM_GB * 1024**3): return "LOW_RAM"

        # Unique index path for each process to avoid "fighting" over index files
        index_path = f"{file_path}.{os.getpid()}.idx"
        
        ds = xr.open_dataset(
            file_path, 
            engine='cfgrib', 
            backend_kwargs={
                'filter_by_keys': config['filter'],
                'indexpath': index_path
            }
        )
        actual_var = list(ds.data_vars)[0] if ds.data_vars else None
        if not actual_var:
            ds.close()
            return False

        # Robust coordinate processing
        data = config['unit_conv'](ds[actual_var])
        data = data.assign_coords(longitude=(((data.longitude + 180) % 360) - 180)).sortby('longitude')

        # Create figure with EXACT aspect ratio and NO borders
        fig = plt.figure(figsize=(20, 10))
        ax = plt.axes([0, 0, 1, 1], projection=ccrs.PlateCarree())
        ax.set_axis_off()
        
        # CRITICAL: Force the map to stretch to the edges of the image
        # This ensures the PNG bounds match Leaflet's bounds perfectly
        ax.set_aspect('auto', adjustable='datalim')

        levels = config['levels']
        cmap = config['cmap']
        if isinstance(cmap, str): cmap = plt.get_cmap(cmap)
        norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N, extend='both')

        if var_key == 'tp':
            data = data.where(data >= 0.01)
        
        # Use pcolormesh for all to ensure grid-perfect alignment
        data.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), 
                            cmap=cmap, norm=norm, 
                            add_colorbar=False, add_labels=False)
        
        ax.set_extent(reg_config['extent'], crs=ccrs.PlateCarree())
        
        # Save with zero padding and no background
        plt.savefig(out_path, bbox_inches=0, pad_inches=0, transparent=True, dpi=reg_config['dpi'])
        
        plt.close(fig)
        ds.close()
        
        # Cleanup the unique index file immediately
        try:
            if os.path.exists(index_path): os.remove(index_path)
        except: pass

        del data
        gc.collect()
        return True
    except Exception as e:
        return False

def generate_legends(output_dir):
    print("--- Generating Color Legends ---")
    for var_key, config in VAR_CONFIG.items():
        out_path = os.path.join(output_dir, f"legend_{var_key}.png")
        fig, ax = plt.subplots(figsize=(4, 0.8))
        fig.subplots_adjust(bottom=0.5)
        levels = config['levels']
        cmap = config['cmap']
        if isinstance(cmap, str): cmap = plt.get_cmap(cmap)
        norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N, extend='both')
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='horizontal',
                         ticks=levels[::2] if len(levels) > 10 else levels, label=f"{config['unit_label']}")
        cb.ax.tick_params(labelsize=8, colors='white')
        cb.set_label(config['unit_label'], color='white', size=9)
        plt.savefig(out_path, transparent=True, bbox_inches='tight', dpi=150)
        plt.close(fig)

def run_processor_service():
    print(f"--- AIGFS Parallel Processor Service Started ({MAX_WORKERS} Workers) ---")
    data_dir = "data"
    output_dir = os.path.join("static", "maps")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        generate_legends(output_dir)
    except Exception as e:
        print(f"Legend generation error: {e}")

    while True:
        tasks = []
        for root, dirs, files in os.walk(data_dir):
            for f in sorted(files):
                if f.endswith('.grib2'):
                    file_path = os.path.join(root, f)
                    parts = f.split('.')
                    run = parts[1][1:3]
                    fhr_int = int(parts[3][1:])
                    fhr_str = parts[3][1:]
                    parent_dir = os.path.basename(os.path.dirname(file_path))
                    date_str = parent_dir.split('_')[0] if '_' in parent_dir else "unknown"

                    for region_name, reg_config in REGIONS.items():
                        if fhr_int <= reg_config['max_fhr']:
                            for var_key in VAR_CONFIG.keys():
                                out_filename = f"aigfs_{region_name}_{date_str}_{run}_{fhr_str}_{var_key}.png"
                                if not os.path.exists(os.path.join(output_dir, out_filename)) or REPROCESS:
                                    tasks.append((file_path, output_dir, region_name, var_key, date_str, run, fhr_str))
                if len(tasks) >= 100: break
            if len(tasks) >= 100: break
        
        if tasks:
            print(f"\n[Parallel Cycle] Processing {len(tasks)} tasks...")
            with Pool(MAX_WORKERS) as pool:
                results = pool.map(generate_map_task, tasks)
            print(f"  > Cycle complete. New maps: {sum(1 for r in results if r is True)}")
        
        # Cleanup orphan .idx files
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                if f.endswith('.idx'):
                    grib_f = f.replace('.idx', '')
                    if not os.path.exists(os.path.join(root, grib_f)):
                        try: os.remove(os.path.join(root, f))
                        except: pass
        
        time.sleep(60 if tasks else 300)

if __name__ == "__main__":
    run_processor_service()
