import os
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import time
import gc
import psutil
import json
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from multiprocessing import Pool, cpu_count
from datetime import datetime

# Processing settings
CLEANUP_GRIB = False
REPROCESS = False     
MAX_WORKERS = 1  # SERIAL PROCESSING ONLY to save RAM
MIN_FREE_RAM_GB = 0.5 

# Region Definitions (Strict Lat/Lon Boxes)
REGIONS = {
    'global': {'extent': [-180, 180, -85, 85], 'max_fhr': 384},
    'conus': {'extent': [-135, -60, 20, 55], 'max_fhr': 168},
    'west': {'extent': [-145, -100, 25, 50], 'max_fhr': 168}
}

# NWS Style Color Configs
NWS_PRECIP_COLORS = ['#E1E1E1', '#A6F28F', '#3DBA3D', '#166B16', '#1EB5EE', '#093BB3', '#D32BE2', '#FF00F5', '#FFBA00', '#FF0000', '#B50000', '#7F0000', '#4B0000']
NWS_PRECIP_LEVELS = [0.01, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]

NWS_TEMP_COLORS = ['#4B0082', '#8A2BE2', '#0000FF', '#4169E1', '#00BFFF', '#00FFFF', '#00FF00', '#32CD32', '#FFFF00', '#FFD700', '#FFA500', '#FF4500', '#FF0000', '#B22222', '#8B0000']

# High Contrast Pressure Colors
NWS_PRESSURE_COLORS = ['#0000FF', '#4169E1', '#00BFFF', '#E0FFFF', '#FFFFE0', '#FFD700', '#FF8C00', '#FF0000', '#8B0000']
PRESSURE_LEVELS = np.arange(960, 1060, 4)

# Wind Speed Colors
WIND_COLORS = ['#FFFFFF', '#E0E0E0', '#B0C4DE', '#87CEFA', '#00BFFF', '#1E90FF', '#0000FF', '#8A2BE2', '#DA70D6', '#FF00FF', '#FF1493', '#8B0000', '#4B0000']
WIND_LEVELS = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100]

VAR_CONFIG = {
    't2m': {
        'cmap': mcolors.LinearSegmentedColormap.from_list('nws_temp', NWS_TEMP_COLORS),
        'levels': np.arange(-40, 121, 2), 
        'unit_conv': lambda x: (x - 273.15) * 9/5 + 32, 'unit_label': '°F',
        'key': 't2m', 
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
        'key': 'wind_speed', 
        'filter': {'typeOfLevel': 'heightAboveGround', 'level': 10} 
    }
}

def process_file(file_path):
    try:
        # Check memory
        mem = psutil.virtual_memory()
        if mem.available < (MIN_FREE_RAM_GB * 1024**3): 
            print("Skipping due to low RAM")
            return False

        basename = os.path.basename(file_path)
        parts = basename.split('.')
        run = parts[1][1:3]
        fhr_str = parts[3][1:]
        fhr_int = int(fhr_str)
        date_str = os.path.basename(os.path.dirname(file_path)).split('_')[0]
        output_dir = os.path.join("static", "maps")
        
        # Determine needed tasks
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
        
        # 1. Load Data
        data_cache = {}
        
        def load_var(filter_keys, internal_name):
            try:
                index_path = f"{file_path}.{internal_name}.{os.getpid()}.idx"
                ds = xr.open_dataset(file_path, engine='cfgrib', 
                                    backend_kwargs={'filter_by_keys': filter_keys, 'indexpath': index_path})
                if not ds.data_vars:
                    ds.close()
                    if os.path.exists(index_path): os.remove(index_path)
                    return

                var = list(ds.data_vars)[0]
                val = ds[var]
                # Fix Longitude: GFS is 0-360. Cartopy handles this, BUT standardizing to -180/180 is safer for cropping.
                val = val.assign_coords(longitude=(((val.longitude + 180) % 360) - 180)).sortby(['latitude', 'longitude'])
                data_cache[internal_name] = val
                ds.close()
                if os.path.exists(index_path): os.remove(index_path)
            except Exception as e:
                pass

        load_var(VAR_CONFIG['t2m']['filter'], 't2m')
        load_var(VAR_CONFIG['tp']['filter'], 'tp')
        load_var(VAR_CONFIG['prmsl']['filter'], 'prmsl')
        load_var({'typeOfLevel': 'heightAboveGround', 'level': 10, 'shortName': 'u10'}, 'u10')
        load_var({'typeOfLevel': 'heightAboveGround', 'level': 10, 'shortName': 'v10'}, 'v10')

        if 'u10' in data_cache and 'v10' in data_cache:
            data_cache['wind_speed'] = np.sqrt(data_cache['u10']**2 + data_cache['v10']**2)
            data_cache['wind_speed'] = data_cache['wind_speed'].assign_coords(latitude=data_cache['u10'].latitude, longitude=data_cache['u10'].longitude)

        if not data_cache:
            return True

        # 2. Generate Maps with Cartopy
        generated_count = 0
        for reg_name, reg_cfg in REGIONS.items():
            if fhr_int > reg_cfg['max_fhr']: continue
            
            lon_min, lon_max, lat_min, lat_max = reg_cfg['extent']

            for var_key, config in VAR_CONFIG.items():
                out_filename = f"aigfs_{reg_name}_{date_str}_{run}_{fhr_str}_{var_key}.png"
                out_path = os.path.join(output_dir, out_filename)
                json_path = out_path.replace('.png', '.json')

                if not REPROCESS and os.path.exists(out_path):
                    continue
                if config['key'] not in data_cache:
                    continue
                if config['key'] == 'wind_speed' and 'wind_speed' not in data_cache:
                    continue

                raw_data = data_cache[config['key']]
                data = config['unit_conv'](raw_data)

                # Loose crop to speed up plotting (add buffer)
                data_crop = data.sel(latitude=slice(lat_min - 2, lat_max + 2), longitude=slice(lon_min - 2, lon_max + 2))
                if data_crop.size == 0: continue

                # Stats Calculation
                valid_vals = data_crop.values[~np.isnan(data_crop.values)]
                if len(valid_vals) > 0:
                    min_val, max_val = float(np.min(valid_vals)), float(np.max(valid_vals))
                else:
                    min_val, max_val = 0.0, 0.0

                # PLOTTING
                # Figure setup: We want high-res output
                # DPI 100, Size 10x10 -> 1000px
                fig = plt.figure(figsize=(10, 10), dpi=100)
                
                # Projection: Google Web Mercator (matches Leaflet)
                ax = plt.axes(projection=ccrs.Mercator.GOOGLE)
                
                # Set Extent: This is the critical part for alignment
                ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
                
                # Colormap setup
                cmap = config['cmap']
                if isinstance(cmap, str): cmap = plt.get_cmap(cmap)
                norm = mcolors.BoundaryNorm(config['levels'], ncolors=cmap.N, extend='both')

                # Plot Data
                # transform=ccrs.PlateCarree() tells Cartopy the data is Lat/Lon
                mesh = ax.pcolormesh(data_crop.longitude, data_crop.latitude, data_crop.values, 
                                     transform=ccrs.PlateCarree(),
                                     cmap=cmap, norm=norm, shading='auto')

                # Hide Axes/Borders completely
                ax.axis('off')
                
                # Special handling for Precip (Alpha)
                if var_key == 'tp':
                    # pcolormesh doesn't support alpha per-pixel easily with arrays, but we can use the colormap
                    # Or simple masking: mask values < 0.01
                    # Cartopy pcolormesh creates a collection.
                    # Better approach: Mask the data before plotting?
                    # Masked array:
                    masked_data = np.ma.masked_less(data_crop.values, 0.01)
                    ax.clear() # Clear the previous unmasked plot
                    ax.axis('off')
                    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
                    mesh = ax.pcolormesh(data_crop.longitude, data_crop.latitude, masked_data,
                                        transform=ccrs.PlateCarree(),
                                        cmap=cmap, norm=norm, shading='auto')
                else:
                     # Global transparency for other layers
                     # pcolormesh supports alpha=...
                     mesh.set_alpha(0.7)

                # Save tightly
                plt.savefig(out_path, transparent=True, bbox_inches='tight', pad_inches=0, dpi=100)
                plt.close(fig)
                
                # Save Stats
                with open(json_path, 'w') as jf:
                    json.dump({'min': min_val, 'max': max_val, 'unit': config['unit_label']}, jf)

                generated_count += 1
        
        del data_cache
        gc.collect()
        if generated_count > 0:
            print(f"Processed {basename}: Generated {generated_count} maps")
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
    print(f"--- AIGFS Raster Processor (Cartopy) Started ({MAX_WORKERS} Workers) ---")
    data_dir, output_dir = "data", os.path.join("static", "maps")
    os.makedirs(output_dir, exist_ok=True)
    generate_legends(output_dir)

    while True:
        files_to_process = []
        for root, dirs, files in os.walk(data_dir):
            for f in sorted(files):
                if f.endswith('.grib2'):
                    files_to_process.append(os.path.join(root, f))
        
        if files_to_process:
            print(f"\n[Parallel Cycle] Scanning {len(files_to_process)} files...")
            with Pool(MAX_WORKERS) as pool:
                pool.map(process_file, files_to_process)
        
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                if f.endswith('.idx') and not os.path.exists(os.path.join(root, f.replace('.idx', ''))):
                    try: os.remove(os.path.join(root, f))
                    except: pass
        
        print("Cycle complete. Sleeping...")
        time.sleep(60)

if __name__ == "__main__":
    run_processor_service()
