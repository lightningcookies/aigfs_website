import os
import xarray as xr
import sys

# Define filters to test
VAR_FILTERS = {
    't2m': {'typeOfLevel': 'heightAboveGround', 'level': 2},
    'u10': {'typeOfLevel': 'heightAboveGround', 'level': 10, 'shortName': 'u10'},
    'v10': {'typeOfLevel': 'heightAboveGround', 'level': 10, 'shortName': 'v10'},
    'prmsl': {'shortName': 'prmsl'},
    'tp': {'shortName': 'tp'}
}

def test_read(file_path):
    print(f"Testing file: {file_path}")
    if not os.path.exists(file_path):
        print("File not found!")
        return

    for var_name, filter_keys in VAR_FILTERS.items():
        print(f"--- Testing {var_name} ---")
        try:
            ds = xr.open_dataset(file_path, engine='cfgrib', backend_kwargs={'filter_by_keys': filter_keys})
            print(f"  Success! Variables: {list(ds.data_vars)}")
            ds.close()
        except Exception as e:
            print(f"  Failed: {e}")

if __name__ == "__main__":
    # Find first grib file
    target_file = None
    for root, dirs, files in os.walk("data"):
        for f in files:
            if f.endswith(".grib2"):
                target_file = os.path.join(root, f)
                break
        if target_file: break
    
    if target_file:
        test_read(target_file)
    else:
        print("No GRIB2 files found in data/ directory.")

