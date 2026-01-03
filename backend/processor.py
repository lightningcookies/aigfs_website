import os
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from datetime import datetime

def process_grib_file(file_path, output_dir):
    """
    Processes a single AIGFS GRIB2 file and generates maps for specified variables.
    """
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return

    # Extract filename info
    # aigfs.t00z.sfc.f000.grib2
    filename = os.path.basename(file_path)
    parts = filename.split('.')
    run = parts[1][1:3]
    fhr = parts[3][1:]

    print(f"Processing {filename}...")

    # Define variables to extract with their GRIB filter keys
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
        try:
            # Open with filter
            # cfgrib creates index files (.idx), we ignore them if they exist
            ds = xr.open_dataset(file_path, engine='cfgrib', 
                                backend_kwargs={'filter_by_keys': filter_keys})
            
            # The variable name in xarray might still be 't', 'u', 'v', etc.
            # but cfgrib often renames them if it recognizes them.
            actual_var = list(ds.data_vars)[0] if ds.data_vars else None
            if not actual_var:
                print(f"Variable {var_key} not found in {filename}")
                continue

            data = ds[actual_var]
            label = var_labels[var_key]
            
            # Unit conversions
            if var_key == 't2m':
                data = data - 273.15 # K to C
                unit = "°C"
            elif var_key == 'prmsl':
                data = data / 100.0 # Pa to hPa/mb
                unit = "hPa"
            else:
                unit = data.attrs.get('units', '')

            # Create plot
            fig = plt.figure(figsize=(15, 8))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linestyle=':')

            # Plot data
            im = data.plot(ax=ax, transform=ccrs.PlateCarree(), 
                           cmap='viridis', add_colorbar=False)
            
            plt.colorbar(im, ax=ax, label=f"{label} ({unit})", orientation='horizontal', pad=0.05, shrink=0.8)
            plt.title(f"AIGFS Run {run}z - Forecast Hour {fhr} - {label}")
            
            # Save plot
            out_filename = f"aigfs_{run}_{fhr}_{var_key}.png"
            out_path = os.path.join(output_dir, out_filename)
            plt.savefig(out_path, bbox_inches='tight', dpi=100)
            plt.close(fig)
            print(f"Generated {out_path}")
            
            ds.close()

        except Exception as e:
            print(f"Error processing variable {var_key} in {file_path}: {e}")

if __name__ == "__main__":
    # Example: process all files in a directory
    data_dir = os.path.join("data", "20260103_00")
    output_dir = os.path.join("static", "maps")
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(data_dir):
        files = sorted([f for f in os.listdir(data_dir) if f.endswith('.grib2')])
        for f in files:
            process_grib_file(os.path.join(data_dir, f), output_dir)
    else:
        print(f"Data directory {data_dir} not found.")
