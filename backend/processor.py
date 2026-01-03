import os
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from datetime import datetime

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

    # Define variables to extract
    # We add 'tp' for Total Precipitation (APCP)
    variables = {
        't2m': {'typeOfLevel': 'heightAboveGround', 'level': 2},
        'u10': {'typeOfLevel': 'heightAboveGround', 'level': 10, 'shortName': 'u10'},
        'v10': {'typeOfLevel': 'heightAboveGround', 'level': 10, 'shortName': 'v10'},
        'prmsl': {'shortName': 'prmsl'},
        'tp': {'shortName': 'tp'} # APCP
    }

    var_labels = {
        't2m': 'Temperature (2m)',
        'u10': 'U Wind (10m)',
        'v10': 'V Wind (10m)',
        'prmsl': 'MSL Pressure',
        'tp': 'Total Precipitation (6h)'
    }

    # Different colormaps for different variables
    var_cmaps = {
        't2m': 'RdYlBu_r',
        'u10': 'viridis',
        'v10': 'viridis',
        'prmsl': 'coolwarm',
        'tp': 'YlGnBu'
    }

    print(f"--- Processing {filename} ---")

    for var_key, filter_keys in variables.items():
        out_filename = f"aigfs_{date_str}_{run}_{fhr}_{var_key}.png"
        out_path = os.path.join(output_dir, out_filename)
        
        if os.path.exists(out_path):
            continue

        try:
            # Attempt to open dataset
            ds = xr.open_dataset(file_path, engine='cfgrib', 
                                backend_kwargs={'filter_by_keys': filter_keys})
            
            actual_var = list(ds.data_vars)[0] if ds.data_vars else None
            if not actual_var:
                ds.close()
                continue

            print(f"  > Creating map: {out_filename}")
            data = ds[actual_var]
            label = var_labels[var_key]
            cmap = var_cmaps.get(var_key, 'viridis')
            
            # Unit conversions
            if var_key == 't2m':
                data = data - 273.15
                unit = "°C"
            elif var_key == 'prmsl':
                data = data / 100.0
                unit = "hPa"
            elif var_key == 'tp':
                # Precip is usually in kg/m^2 which is equivalent to mm
                unit = "mm"
            else:
                unit = data.attrs.get('units', '')

            fig = plt.figure(figsize=(15, 8))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=1)
            ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
            ax.add_feature(cfeature.STATES, linestyle='--', linewidth=0.2, edgecolor='gray')

            # Plot data
            # For precip, we often want to mask out zeros
            if var_key == 'tp':
                data = data.where(data > 0.1)
            
            im = data.plot(ax=ax, transform=ccrs.PlateCarree(), 
                           cmap=cmap, add_colorbar=False)
            
            plt.colorbar(im, ax=ax, label=f"{label} ({unit})", orientation='horizontal', pad=0.05, shrink=0.8)
            plt.title(f"AIGFS {date_str} {run}z - Forecast Hour {fhr} - {label}")
            
            plt.savefig(out_path, bbox_inches='tight', dpi=120)
            plt.close(fig)
            ds.close()

        except Exception as e:
            if "filter_by_keys" not in str(e):
                print(f"  ! Error on {var_key}: {e}")

if __name__ == "__main__":
    data_dir = os.path.join("data")
    output_dir = os.path.join("static", "maps")
    os.makedirs(output_dir, exist_ok=True)
    for root, dirs, files in os.walk(data_dir):
        for f in sorted(files):
            if f.endswith('.grib2'):
                process_grib_file(os.path.join(root, f), output_dir)
