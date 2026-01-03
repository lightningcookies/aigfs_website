import os
import sys
from backend.scraper import download_aigfs_data, get_latest_runs
from backend.processor import process_grib_file

def main():
    # Forecast hours: 0 to 384 in 6h increments
    # To save time/bandwidth, you could reduce this (e.g., range(0, 126, 6))
    forecast_hours = [f"{h:03d}" for h in range(0, 390, 6)]
    
    # Get the last 4 potential runs to see if new data is available
    latest_runs = get_latest_runs()
    
    print(f"--- Checking for new AIGFS runs ---")
    
    for date_str, run_str in latest_runs:
        print(f"\nChecking Run: {date_str} {run_str}Z")
        
        # 1. Download Data
        # Returns True if new files were actually downloaded
        download_aigfs_data(date_str, run_str, forecast_hours)
        
        # 2. Process Data
        data_dir = os.path.join("data", f"{date_str}_{run_str}")
        output_dir = os.path.join("static", "maps")
        
        if os.path.exists(data_dir):
            files = sorted([f for f in os.listdir(data_dir) if f.endswith('.grib2')])
            for f in files:
                grib_path = os.path.join(data_dir, f)
                # Check if we already processed this file by checking for one of the output images
                # e.g., aigfs_00_000_t2m.png
                # Actually, the processor should be smart enough or we just overwrite.
                # Let's add a date prefix to the map filenames to distinguish different days.
                
                # We'll pass the date_str to process_grib_file if we want, 
                # but for now let's just update processor.py to include date in filename.
                process_grib_file(grib_path, output_dir, date_str=date_str)
        else:
            print(f"No data for {date_str} {run_str}Z")

    print("\n--- All Done! ---")

if __name__ == "__main__":
    main()
