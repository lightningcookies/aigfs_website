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
MAX_FILES_PER_CYCLE = 10 
# SET THIS based on your core count. 4 is safe for 16GB.
MAX_WORKERS = min(4, cpu_count()) 
MIN_FREE_RAM_GB = 2.0  # Safety buffer

# Region Definitions
REGIONS = {
    'global': {'extent': None, 'max_fhr': 384, 'dpi': 120},
    'conus': {'extent': [-125, -66, 23, 50], 'max_fhr': 168, 'dpi': 180},
    'west': {'extent': [-126, -103, 31, 50], 'max_fhr': 168, 'dpi': 220}
}

# Centralized configuration for standardized scales (Imperial Units)
VAR_CONFIG = {
    't2m': {
        'label': 'Temperature (2m)',
        'cmap': 'RdYlBu_r',
        'levels': np.arange(-20, 111, 5),
        'unit_conv': lambda x: (x - 273.15) * 9/5 + 32,
        'unit_label': '°F',
        'filter': {'typeOfLevel': 'heightAboveGround', 'level': 2}
    },
    'u10': {
        'label': 'U Wind (10m)',
        'cmap': 'viridis',
        'levels': np.arange(-60, 61, 10),
        'unit_conv': lambda x: x * 2.23694,
        'unit_label': 'mph',
        'filter': {'typeOfLevel': 'heightAboveGround', 'level': 10, 'shortName': 'u10'}
    },
    'v10': {
        'label': 'V Wind (10m)',
        'cmap': 'viridis',
        'levels': np.arange(-60, 61, 10),
        'unit_conv': lambda x: x * 2.23694,
        'unit_label': 'mph',
        'filter': {'typeOfLevel': 'heightAboveGround', 'level': 10, 'shortName': 'v10'}
    },
    'prmsl': {
        'label': 'MSL Pressure',
        'cmap': 'coolwarm',
        'levels': np.arange(960, 1051, 2),
        'unit_conv': lambda x: x / 100.0,
        'unit_label': 'hPa',
        'filter': {'shortName': 'prmsl'}
    },
    'tp': {
        'label': 'Total Precipitation (6h)',
        'cmap': 'YlGnBu',
        'levels': [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5],
        'unit_conv': lambda x: x / 25.4,
        'unit_label': 'in',
        'filter': {'shortName': 'tp'}
    }
}

def generate_map_task(args):
    """Worker function to generate a single map."""
    file_path, output_dir, region_name, var_key, date_str, run, fhr_str = args
    
    config = VAR_CONFIG[var_key]
    reg_config = REGIONS[region_name]
    out_filename = f"aigfs_{region_name}_{date_str}_{run}_{fhr_str}_{var_key}.png"
    out_path = os.path.join(output_dir, out_filename)

    if os.path.exists(out_path) and not REPROCESS:
        return False

    try:
        # Check RAM safety before starting
        mem = psutil.virtual_memory()
        if mem.available < (MIN_FREE_RAM_GB * 1024**3):
            return "LOW_RAM"

        ds = xr.open_dataset(file_path, engine='cfgrib', 
                            backend_kwargs={'filter_by_keys': config['filter']})
        
        actual_var = list(ds.data_vars)[0] if ds.data_vars else None
        if not actual_var:
            ds.close()
            return False

        data = config['unit_conv'](ds[actual_var])

        fig = plt.figure(figsize=(20, 10), frameon=False)
        ax = plt.axes([0, 0, 1, 1], projection=ccrs.PlateCarree())
        ax.set_axis_off()

        levels = config['levels']
        norm = mcolors.BoundaryNorm(levels, ncolors=plt.get_cmap(config['cmap']).N, extend='both')

        if var_key == 'tp':
            data = data.where(data >= 0.01)
        
        data.plot.contourf(ax=ax, transform=ccrs.PlateCarree(), 
                         levels=levels, cmap=config['cmap'], norm=norm, 
                         add_colorbar=False, add_labels=False)
        
        if reg_config['extent']:
            ax.set_extent(reg_config['extent'], crs=ccrs.PlateCarree())
        else:
            ax.set_global()
        
        plt.savefig(out_path, bbox_inches=0, pad_inches=0, transparent=True, dpi=reg_config['dpi'])
        
        plt.close(fig)
        ds.close()
        del data
        gc.collect()
        return True
    except Exception as e:
        if "filter_by_keys" not in str(e):
            print(f"  ! Error on {region_name}/{var_key}: {e}")
        return False

def generate_legends(output_dir):
    """Generates standalone colorbar legends for each variable."""
    print("--- Generating Color Legends ---")
    for var_key, config in VAR_CONFIG.items():
        out_path = os.path.join(output_dir, f"legend_{var_key}.png")
        if os.path.exists(out_path): continue

        fig, ax = plt.subplots(figsize=(4, 0.8))
        fig.subplots_adjust(bottom=0.5)
        levels = config['levels']
        cmap = plt.get_cmap(config['cmap'])
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
    generate_legends(output_dir)

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
                                tasks.append((file_path, output_dir, region_name, var_key, date_str, run, fhr_str))
                
                if len(tasks) >= MAX_FILES_PER_CYCLE * 15: # Cap task list size
                    break
        
        if tasks:
            print(f"\n[Parallel Cycle] Processing {len(tasks)} map tasks...")
            with Pool(MAX_WORKERS) as pool:
                results = pool.map(generate_map_task, tasks)
            
            new_maps = sum(1 for r in results if r is True)
            low_ram_hits = sum(1 for r in results if r == "LOW_RAM")
            
            print(f"  > Cycle complete. New maps: {new_maps}")
            if low_ram_hits > 0:
                print(f"  ! WARNING: {low_ram_hits} tasks skipped due to low RAM.")

            # Optional Cleanup (if enabled)
            if CLEANUP_GRIB:
                # Cleanup logic here...
                pass

        time.sleep(120 if tasks else 300)

if __name__ == "__main__":
    run_processor_service()
