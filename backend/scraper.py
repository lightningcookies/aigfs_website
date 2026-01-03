import os
import requests
from datetime import datetime, timedelta

def download_aigfs_data(date_str, run, forecast_hours):
    """
    Downloads AIGFS GRIB2 files from NOAA NOMADS.
    """
    base_url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/aigfs/prod/aigfs.{date_str}/{run}/model/atmos/grib2/"
    download_dir = os.path.join("data", f"{date_str}_{run}")
    os.makedirs(download_dir, exist_ok=True)

    downloaded_any = False
    for fhr in forecast_hours:
        filename = f"aigfs.t{run}z.sfc.f{fhr}.grib2"
        url = base_url + filename
        target_path = os.path.join(download_dir, filename)

        if os.path.exists(target_path):
            # Check if file is not empty (sometimes downloads fail)
            if os.path.getsize(target_path) > 1000:
                continue

        print(f"Downloading {url}...")
        try:
            response = requests.get(url, stream=True, timeout=30)
            if response.status_code == 404:
                print(f"File not yet available: {filename}")
                continue
            response.raise_for_status()
            with open(target_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Saved to {target_path}")
            downloaded_any = True
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
    
    return downloaded_any

def get_latest_runs():
    """
    Returns a list of the last 4 potential (date, run) tuples.
    AIGFS runs at 00, 06, 12, 18 UTC.
    """
    now = datetime.utcnow()
    runs = []
    
    # Check the last 4 possible run times
    for i in range(4):
        check_time = now - timedelta(hours=i*6)
        # Round down to the nearest 6h run (0, 6, 12, 18)
        run_hour = (check_time.hour // 6) * 6
        date_str = check_time.strftime("%Y%m%d")
        run_str = f"{run_hour:02d}"
        runs.append((date_str, run_str))
    
    return runs

if __name__ == "__main__":
    # Test latest runs
    latest = get_latest_runs()
    print(f"Checking runs: {latest}")
    hours = ["000", "006"] # Small test
    for d, r in latest:
        download_aigfs_data(d, r, hours)
