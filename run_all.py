import os
import sys
from datetime import datetime
from backend.scraper import download_aigfs_data
from backend.processor import process_grib_file

def main():
    # 1. Configuration
    # You can change these to get different dates/runs
    target_date = "20260103" 
    target_run = "00"
    
    # Forecast hours: 0 to 384 in 6h increments as requested
    # For a quick test, you might want to reduce this list (e.g., [0, 6, 12])
    forecast_hours = [f"{h:03d}" for h in range(0, 390, 6)]
    
    print(f"--- Starting AIGFS Data Retrieval for {target_date} Run {target_run}Z ---")
    
    # 2. Download Data
    download_aigfs_data(target_date, target_run, forecast_hours)
    
    # 3. Process Data
    print("\n--- Starting Data Processing ---")
    data_dir = os.path.join("data", f"{target_date}_{target_run}")
    output_dir = os.path.join("static", "maps")
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.exists(data_dir):
        # Only process GRIB2 files that were downloaded
        files = sorted([f for f in os.listdir(data_dir) if f.endswith('.grib2')])
        for f in files:
            process_grib_file(os.path.join(data_dir, f), output_dir)
    else:
        print(f"Data directory {data_dir} not found. Check if downloads were successful.")

    print("\n--- All Done! ---")
    print("You can now start the web server by running: python app.py")

if __name__ == "__main__":
    main()

