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
        return

    filename = os.path.basename(file_path)
    # aigfs.t00z.sfc.f000.grib2
    parts = filename.split('.')
    if len(parts) < 4:
        return
        
    run = parts[1][1:3]
    fhr = parts[3][1:]
    
    if not date_str:
        parent_dir = os.path.basename(os.path.dirname(file_path))
        if '_' in parent_dir:
            date_str = parent_dir.split('_')[0]
        else:
            date_str = datetime.now().strftime("%Y%m%d")

    variables = {
        't2m': {'typeOfLevel': 'heightAboveGround', 'level': 2},
        'u10': {'typeOfLevel': 'heightAboveGround', 'level': 10, 'shortName': 'u10'},
        'v10': {'typeOfLevel': 'heightAboveGround', 'level': 10, 'shortName': 'v10'},
        'prmsl': {'shortName': 'prmsl'}
    }

    var_labels = {
        't2m': 'Temperature (2m)',
        'u10': 'U Wind (10m)',
        'v10': 'V Wind (10m)',
        'prmsl': 'MSL Pressure'
    }

    for var_key, filter_keys in variables.items():
        out_filename = f"aigfs_{date_str}_{run}_{fhr}_{var_key}.png"
        out_path = os.path.join(output_dir, out_filename)
        
        if os.path.exists(out_path):
            continue

        try:
            # cfgrib can be noisy, we want to know if it actually fails
            ds = xr.open_dataset(file_path, engine='cfgrib', 
                                backend_kwargs={'filter_by_keys': filter_keys})
            
            actual_var = list(ds.data_vars)[0] if ds.data_vars else None
            if not actual_var:
                ds.close()
                continue

            print(f"Generating map: {out_filename}")
            data = ds[actual_var]
            label = var_labels[var_key]
            
            if var_key == 't2m':
                data = data - 273.15
                unit = "°C"
            elif var_key == 'prmsl':
                data = data / 100.0
                unit = "hPa"
            else:
                unit = data.attrs.get('units', '')

            fig = plt.figure(figsize=(15, 8))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linestyle=':')

            im = data.plot(ax=ax, transform=ccrs.PlateCarree(), 
                           cmap='viridis', add_colorbar=False)
            
            plt.colorbar(im, ax=ax, label=f"{label} ({unit})", orientation='horizontal', pad=0.05, shrink=0.8)
            plt.title(f"AIGFS {date_str} {run}z - Forecast Hour {fhr} - {label}")
            
            plt.savefig(out_path, bbox_inches='tight', dpi=100)
            plt.close(fig)
            ds.close()

        except Exception as e:
            # Only print if it's not a common filtering issue
            if "filter_by_keys" not in str(e):
                print(f"Error processing {var_key} in {filename}: {e}")

if __name__ == "__main__":
    # This is just for manual testing
    data_dir = os.path.join("data")
    output_dir = os.path.join("static", "maps")
    os.makedirs(output_dir, exist_ok=True)

    for root, dirs, files in os.walk(data_dir):
        for f in sorted(files):
            if f.endswith('.grib2'):
                process_grib_file(os.path.join(root, f), output_dir)
