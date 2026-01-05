import os
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import time
from datetime import datetime

# Processing settings
CLEANUP_GRIB = False  # Set to True to delete GRIB2 files after processing (saves space)
REPROCESS = False     # Set to True to overwrite existing maps

# Centralized configuration for standardized scales
VAR_CONFIG = {
    't2m': {
        'label': 'Temperature (2m)',
        'cmap': 'RdYlBu_r',
        'vmin': -30,
        'vmax': 45,
        'unit_conv': lambda x: x - 273.15,
        'unit_label': '°C',
        'filter': {'typeOfLevel': 'heightAboveGround', 'level': 2}
    },
    'u10': {
        'label': 'U Wind (10m)',
        'cmap': 'viridis',
        'vmin': -50,
        'vmax': 50,
        'unit_conv': lambda x: x,
        'unit_label': 'm/s',
        'filter': {'typeOfLevel': 'heightAboveGround', 'level': 10, 'shortName': 'u10'}
    },
    'v10': {
        'label': 'V Wind (10m)',
        'cmap': 'viridis',
        'vmin': -50,
        'vmax': 50,
        'unit_conv': lambda x: x,
        'unit_label': 'm/s',
        'filter': {'typeOfLevel': 'heightAboveGround', 'level': 10, 'shortName': 'v10'}
    },
    'prmsl': {
        'label': 'MSL Pressure',
        'cmap': 'coolwarm',
        'vmin': 970,
        'vmax': 1040,
        'unit_conv': lambda x: x / 100.0,
        'unit_label': 'hPa',
        'filter': {'shortName': 'prmsl'}
    },
    'tp': {
        'label': 'Total Precipitation (6h)',
        'cmap': 'YlGnBu',
        'vmin': 0,
        'vmax': 50,
        'unit_conv': lambda x: x,
        'unit_label': 'mm',
        'filter': {'shortName': 'tp'}
    }
}

def process_grib_file(file_path, output_dir, date_str=None):
    """
    Processes a single AIGFS GRIB2 file and generates maps for specified variables.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    try:
        import cfgrib
    except ImportError:
        print("CRITICAL ERROR: 'cfgrib' is not installed.")
        return

    filename = os.path.basename(file_path)
    parts = filename.split('.')
    if len(parts) < 4:
        return
        
    run = parts[1][1:3]
    fhr = parts[3][1:]
    
    if not date_str:
        parent_dir = os.path.basename(os.path.dirname(file_path))
        date_str = parent_dir.split('_')[0] if '_' in parent_dir else "unknown"

    print(f"--- Processing {filename} ---")

    for var_key, config in VAR_CONFIG.items():
        out_filename = f"aigfs_{date_str}_{run}_{fhr}_{var_key}.png"
        out_path = os.path.join(output_dir, out_filename)
        
        if os.path.exists(out_path) and not REPROCESS:
            continue

        try:
            ds = xr.open_dataset(file_path, engine='cfgrib', 
                                backend_kwargs={'filter_by_keys': config['filter']})
            
            actual_var = list(ds.data_vars)[0] if ds.data_vars else None
            if not actual_var:
                ds.close()
                continue

            print(f"  > Creating map: {out_filename}")
            data = ds[actual_var]
            
            # Apply unit conversion
            data = config['unit_conv'](data)

            fig = plt.figure(figsize=(15, 8))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=1)
            ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
            ax.add_feature(cfeature.STATES, linestyle='--', linewidth=0.2, edgecolor='gray')

            # Mask out zeros for precipitation
            if var_key == 'tp':
                data = data.where(data > 0.1)
            
            # Use fixed vmin and vmax for standardized scales
            im = data.plot(ax=ax, transform=ccrs.PlateCarree(), 
                           cmap=config['cmap'], 
                           vmin=config['vmin'], 
                           vmax=config['vmax'], 
                           add_colorbar=False)
            
            plt.colorbar(im, ax=ax, label=f"{config['label']} ({config['unit_label']})", 
                        orientation='horizontal', pad=0.05, shrink=0.8)
            plt.title(f"AIGFS {date_str} {run}z - Forecast Hour {fhr} - {config['label']}")
            
            plt.savefig(out_path, bbox_inches='tight', dpi=120)
            plt.close(fig)
            ds.close()

        except Exception as e:
            if "filter_by_keys" not in str(e):
                print(f"  ! Error on {var_key}: {e}")

def run_processor_service():
    """
    Infinite loop that watches for new GRIB2 files and processes them.
    """
    print("--- AIGFS Processor Service Started ---")
    data_dir = "data"
    output_dir = os.path.join("static", "maps")
    os.makedirs(output_dir, exist_ok=True)

    while True:
        processed_count = 0
        for root, dirs, files in os.walk(data_dir):
            for f in sorted(files):
                if f.endswith('.grib2'):
                    grib_path = os.path.join(root, f)
                    # Simple check: if all PNGs for this file exist, we might skip
                    # but process_grib_file already does that check per variable.
                    process_grib_file(grib_path, output_dir)
                    processed_count += 1
                    
                    # Optional Cleanup: Delete GRIB2 file after processing to save space
                    if CLEANUP_GRIB:
                        try:
                            os.remove(grib_path)
                            print(f"  > Cleaned up {f}")
                            # Also remove the .idx file created by cfgrib if it exists
                            idx_path = grib_path + ".idx"
                            if os.path.exists(idx_path):
                                os.remove(idx_path)
                        except Exception as e:
                            print(f"  ! Cleanup failed for {f}: {e}")
        
        if processed_count == 0:
            # Nothing to do, sleep for 5 minutes
            time.sleep(300)
        else:
            # Just processed some files, wait a bit before next scan
            time.sleep(60)

if __name__ == "__main__":
    run_processor_service()
