import os
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import time
import gc
from datetime import datetime

# Processing settings
CLEANUP_GRIB = False
REPROCESS = False     # Set to True only when testing new visual changes
MAX_FILES_PER_CYCLE = 5 # Small batches to keep RAM low

# Region Definitions
REGIONS = {
    'global': {'extent': None, 'max_fhr': 384, 'dpi': 120},
    'conus': {'extent': [-125, -66, 23, 50], 'max_fhr': 168, 'dpi': 180},
    'west': {'extent': [-126, -103, 31, 50], 'max_fhr': 168, 'dpi': 220}
}

# Centralized configuration for standardized scales
VAR_CONFIG = {
    't2m': {
        'label': 'Temperature (2m)',
        'cmap': 'RdYlBu_r',
        'levels': np.arange(-40, 51, 2), # 2-degree increments for better detail
        'unit_conv': lambda x: x - 273.15,
        'unit_label': '°C',
        'filter': {'typeOfLevel': 'heightAboveGround', 'level': 2}
    },
    'u10': {
        'label': 'U Wind (10m)',
        'cmap': 'viridis',
        'levels': np.arange(-50, 51, 5),
        'unit_conv': lambda x: x,
        'unit_label': 'm/s',
        'filter': {'typeOfLevel': 'heightAboveGround', 'level': 10, 'shortName': 'u10'}
    },
    'v10': {
        'label': 'V Wind (10m)',
        'cmap': 'viridis',
        'levels': np.arange(-50, 51, 5),
        'unit_conv': lambda x: x,
        'unit_label': 'm/s',
        'filter': {'typeOfLevel': 'heightAboveGround', 'level': 10, 'shortName': 'v10'}
    },
    'prmsl': {
        'label': 'MSL Pressure',
        'cmap': 'coolwarm',
        'levels': np.arange(960, 1051, 2), # 2hPa intervals for tighter detail
        'unit_conv': lambda x: x / 100.0,
        'unit_label': 'hPa',
        'filter': {'shortName': 'prmsl'}
    },
    'tp': {
        'label': 'Total Precipitation (6h)',
        'cmap': 'YlGnBu',
        'levels': [0.1, 0.5, 1, 2, 5, 10, 15, 20, 25, 35, 50, 75, 100, 150, 200],
        'unit_conv': lambda x: x,
        'unit_label': 'mm',
        'filter': {'shortName': 'tp'}
    }
}

def process_grib_file(file_path, output_dir, date_str=None):
    if not os.path.exists(file_path):
        return

    filename = os.path.basename(file_path)
    parts = filename.split('.')
    if len(parts) < 4: return
        
    run = parts[1][1:3]
    fhr_int = int(parts[3][1:])
    fhr_str = parts[3][1:]
    
    if not date_str:
        parent_dir = os.path.basename(os.path.dirname(file_path))
        date_str = parent_dir.split('_')[0] if '_' in parent_dir else "unknown"

    print(f"--- Processing {filename} ---")

    for region_name, reg_config in REGIONS.items():
        # Check if this forecast hour should be processed for this region
        if fhr_int > reg_config['max_fhr']:
            continue

        for var_key, var_config in VAR_CONFIG.items():
            out_filename = f"aigfs_{region_name}_{date_str}_{run}_{fhr_str}_{var_key}.png"
            out_path = os.path.join(output_dir, out_filename)
            
            if os.path.exists(out_path) and not REPROCESS:
                continue

            try:
                # Open dataset
                ds = xr.open_dataset(file_path, engine='cfgrib', 
                                    backend_kwargs={'filter_by_keys': var_config['filter']})
                
                actual_var = list(ds.data_vars)[0] if ds.data_vars else None
                if not actual_var:
                    ds.close()
                    continue

                print(f"  > [{region_name.upper()}] {var_key}: {out_filename}")
                data = var_config['unit_conv'](ds[actual_var])

                # High-res transparency map for Leaflet
                fig = plt.figure(figsize=(20, 10), frameon=False)
                ax = plt.axes([0, 0, 1, 1], projection=ccrs.PlateCarree())
                ax.set_axis_off()

                levels = var_config['levels']
                norm = mcolors.BoundaryNorm(levels, ncolors=plt.get_cmap(var_config['cmap']).N, extend='both')

                if var_key == 'tp':
                    data = data.where(data >= 0.1)
                
                # Plot
                data.plot.contourf(ax=ax, transform=ccrs.PlateCarree(), 
                                 levels=levels, cmap=var_config['cmap'], norm=norm, 
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

            except Exception as e:
                if "filter_by_keys" not in str(e):
                    print(f"  ! Error on {region_name}/{var_key}: {e}")

def run_processor_service():
    print("--- AIGFS Regional Processor Service Started ---")
    data_dir = "data"
    output_dir = os.path.join("static", "maps")
    os.makedirs(output_dir, exist_ok=True)

    while True:
        processed_in_cycle = 0
        for root, dirs, files in os.walk(data_dir):
            for f in sorted(files):
                if f.endswith('.grib2'):
                    grib_path = os.path.join(root, f)
                    process_grib_file(grib_path, output_dir)
                    processed_in_cycle += 1
                    
                    if CLEANUP_GRIB:
                        try:
                            os.remove(grib_path)
                            idx_path = grib_path + ".idx"
                            if os.path.exists(idx_path): os.remove(idx_path)
                        except: pass
                
                if processed_in_cycle >= MAX_FILES_PER_CYCLE:
                    break
            if processed_in_cycle >= MAX_FILES_PER_CYCLE:
                break
        
        time.sleep(120 if processed_in_cycle > 0 else 300)

if __name__ == "__main__":
    run_processor_service()
