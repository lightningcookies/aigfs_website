import os
import requests
from datetime import datetime, timedelta

def download_aigfs_data(date_str, run, forecast_hours):
    """
    Downloads AIGFS GRIB2 files from NOAA NOMADS.
    
    Args:
        date_str (str): Date in yyyymmdd format.
        run (str): Model run (00, 06, 12, 18).
        forecast_hours (list): List of forecast hours (strings like '000', '006', etc.)
    """
    base_url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/aigfs/prod/aigfs.{date_str}/{run}/model/atmos/grib2/"
    download_dir = os.path.join("data", f"{date_str}_{run}")
    os.makedirs(download_dir, exist_ok=True)

    for fhr in forecast_hours:
        filename = f"aigfs.t{run}z.sfc.f{fhr}.grib2"
        url = base_url + filename
        target_path = os.path.join(download_dir, filename)

        if os.path.exists(target_path):
            print(f"Skipping {filename}, already exists.")
            continue

        print(f"Downloading {url}...")
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            with open(target_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Saved to {target_path}")
        except Exception as e:
            print(f"Failed to download {filename}: {e}")

if __name__ == "__main__":
    # Example usage: Download latest run
    # For now, let's just use the current date or the one from the prompt
    now = datetime.utcnow()
    # AIGFS runs every 6 hours. Latest run might be a few hours behind.
    # The prompt used 20260103 and run 00.
    target_date = "20260103" 
    target_run = "00"
    
    # Forecast hours: 0 to 384 in 6h increments
    hours = [f"{h:03d}" for h in range(0, 390, 6)]
    
    download_aigfs_data(target_date, target_run, hours)

