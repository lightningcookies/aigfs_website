import os
import sys
from backend.scraper import download_aigfs_data, get_latest_runs
from backend.processor import process_grib_file

def main():
    forecast_hours = [f"{h:03d}" for h in range(0, 390, 6)]
    latest_runs = get_latest_runs()
    
    print(f"--- Checking for new AIGFS runs ---")
    
    total_processed = 0
    for date_str, run_str in latest_runs:
        print(f"\n[Run Check] {date_str} {run_str}Z")
        
        # 1. Download
        downloaded = download_aigfs_data(date_str, run_str, forecast_hours)
        
        # 2. Process
        data_dir = os.path.join("data", f"{date_str}_{run_str}")
        output_dir = os.path.join("static", "maps")
        os.makedirs(output_dir, exist_ok=True)
        
        if os.path.exists(data_dir):
            files = [f for f in os.listdir(data_dir) if f.endswith('.grib2')]
            if not files:
                print(f"  No GRIB2 files found in {data_dir}")
                continue
                
            print(f"  Found {len(files)} files to process in {data_dir}")
            for f in sorted(files):
                grib_path = os.path.join(data_dir, f)
                process_grib_file(grib_path, output_dir, date_str=date_str)
                total_processed += 1
        else:
            print(f"  Data directory {data_dir} does not exist.")

    print(f"\n--- Processing Finished. Total files handled: {total_processed} ---")
    print(f"Check your 'static/maps' folder for .png files.")

if __name__ == "__main__":
    main()
