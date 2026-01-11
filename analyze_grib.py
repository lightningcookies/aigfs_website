import cfgrib
import xarray as xr
import os
import sys

def analyze_grib(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    print(f"--- Analyzing GRIB2 file: {file_path} ---")
    
    try:
        # Use open_datasets to get all hypercubes
        datasets = cfgrib.open_datasets(file_path)
        print(f"Total datasets (hypercubes) found: {len(datasets)}\n")

        for i, ds in enumerate(datasets):
            print(f"--- DATASET {i} ---")
            
            # Variables
            vars_info = []
            for var_name in ds.data_vars:
                var = ds[var_name]
                desc = var.attrs.get('GRIB_name', 'N/A')
                units = var.attrs.get('units', 'N/A')
                vars_info.append(f"  - {var_name}: {desc} ({units})")
            
            print("Variables:")
            print("\n".join(vars_info))

            # Coordinates & Levels
            print("Dimensions:", ds.dims)
            
            level_info = "N/A"
            for coord in ds.coords:
                if coord not in ['latitude', 'longitude', 'time', 'step', 'valid_time']:
                    val = ds[coord].values
                    level_info = f"{coord} = {val}"
            
            print("Level:", level_info)
            
            # Time
            ref_time = ds.time.values
            step = ds.step.values
            valid_time = ds.valid_time.values
            print(f"Reference Time (Run): {ref_time}")
            print(f"Forecast Step: {step}")
            print(f"Valid Time: {valid_time}")

            # Spatial
            if 'latitude' in ds.coords and 'longitude' in ds.coords:
                lat_min, lat_max = ds.latitude.min().item(), ds.latitude.max().item()
                lon_min, lon_max = ds.longitude.min().item(), ds.longitude.max().item()
                print(f"Grid: {len(ds.latitude)}x{len(ds.longitude)} (Lat: {lat_min} to {lat_max}, Lon: {lon_min} to {lon_max})")
                
                # Test selection for Alta
                try:
                    ALTA_LAT, ALTA_LON = 40.57, -111.63
                    # Convert lon to 0-360 if needed
                    grib_lon = ALTA_LON if ALTA_LON >= 0 else ALTA_LON + 360
                    
                    sample = ds.sel(latitude=ALTA_LAT, longitude=grib_lon, method='nearest')
                    print(f"  Sample lookup for Alta ({ALTA_LAT}, {ALTA_LON} -> GRIB Lon {grib_lon}):")
                    for v in ds.data_vars:
                        print(f"    {v}: {sample[v].values.item():.4f}")
                except Exception as e:
                    print(f"  Sample lookup failed: {e}")
            
            print("-" * 30 + "\n")

    except Exception as e:
        print(f"Error analyzing GRIB file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    target = "aigfs_example.grib2"
    if len(sys.argv) > 1:
        target = sys.argv[1]
    analyze_grib(target)
