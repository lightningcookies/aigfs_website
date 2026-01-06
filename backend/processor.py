import os
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import time
from datetime import datetime

# Processing settings
CLEANUP_GRIB = False  # Set to True to delete GRIB2 files after processing (saves space)
REPROCESS = True      # Set to True to overwrite existing maps

# Centralized configuration for standardized scales
VAR_CONFIG = {
    't2m': {
        'label': 'Temperature (2m)',
        'cmap': 'RdYlBu_r',
        'levels': np.arange(-40, 51, 5), # Discrete steps every 5 degrees
        'unit_conv': lambda x: x - 273.15,
        'unit_label': '°C',
        'filter': {'typeOfLevel': 'heightAboveGround', 'level': 2}
    },
    'u10': {
        'label': 'U Wind (10m)',
        'cmap': 'viridis',
        'levels': np.arange(-50, 51, 10),
        'unit_conv': lambda x: x,
        'unit_label': 'm/s',
        'filter': {'typeOfLevel': 'heightAboveGround', 'level': 10, 'shortName': 'u10'}
    },
    'v10': {
        'label': 'V Wind (10m)',
        'cmap': 'viridis',
        'levels': np.arange(-50, 51, 10),
        'unit_conv': lambda x: x,
        'unit_label': 'm/s',
        'filter': {'typeOfLevel': 'heightAboveGround', 'level': 10, 'shortName': 'v10'}
    },
    'prmsl': {
        'label': 'MSL Pressure',
        'cmap': 'coolwarm',
        'levels': np.arange(960, 1051, 4), # Standard 4hPa intervals
        'unit_conv': lambda x: x / 100.0,
        'unit_label': 'hPa',
        'filter': {'shortName': 'prmsl'}
    },
    'tp': {
        'label': 'Total Precipitation (6h)',
        'cmap': 'YlGnBu',
        # NOAA-style discrete levels for rainfall (mm)
        'levels': [0.1, 0.5, 1, 2, 5, 10, 15, 20, 25, 35, 50, 75, 100],
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
            # Check if file is still being written to
            initial_size = os.path.getsize(file_path)
            time.sleep(0.5)
            if os.path.getsize(file_path) != initial_size:
                return

            ds = xr.open_dataset(file_path, engine='cfgrib', 
                                backend_kwargs={'filter_by_keys': config['filter']})
            
            actual_var = list(ds.data_vars)[0] if ds.data_vars else None
            if not actual_var:
                ds.close()
                continue

            print(f"  > Creating map: {out_filename}")
            data = ds[actual_var]
            data = config['unit_conv'](data)

            # Create a clean map without borders for Leaflet overlay
            # We use PlateCarree which maps directly to Lat/Lon
            fig = plt.figure(figsize=(20, 10), frameon=False)
            ax = plt.axes([0, 0, 1, 1], projection=ccrs.PlateCarree())
            ax.set_axis_off()

            # Define discrete norm for the colormap
            levels = config['levels']
            norm = mcolors.BoundaryNorm(levels, ncolors=plt.get_cmap(config['cmap']).N, extend='both')

            # Plot data
            if var_key == 'tp':
                data = data.where(data >= 0.1) # Hide 0 precip
            
            # Use 'pcolormesh' or 'contourf' for clean edges in Leaflet
            im = data.plot.contourf(ax=ax, transform=ccrs.PlateCarree(), 
                                  levels=levels, cmap=config['cmap'], norm=norm, 
                                  add_colorbar=False, add_labels=False)
            
            # Set extent to full global range to ensure PNG matches Leaflet expectations
            ax.set_global()
            
            # Save the PNG with NO padding/borders/background
            # This is crucial for Leaflet overlays
            plt.savefig(out_path, bbox_inches=0, pad_inches=0, transparent=True, dpi=150)
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
                    process_grib_file(grib_path, output_dir)
                    processed_count += 1
                    
                    if CLEANUP_GRIB:
                        try:
                            os.remove(grib_path)
                            idx_path = grib_path + ".idx"
                            if os.path.exists(idx_path):
                                os.remove(idx_path)
                        except: pass
        
        time.sleep(60 if processed_count > 0 else 300)

if __name__ == "__main__":
    run_processor_service()
