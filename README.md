# AIGFS Weather Model Viewer

This project allows you to download, process, and visualize the AIGFS global weather model data on your local network.

## Prerequisites

You will need a Linux server with Python 3 installed. It is recommended to use a virtual environment.

### System Dependencies

GRIB2 processing requires the `eccodes` library. On Ubuntu/Debian, you can install it with:

```bash
sudo apt-get update
sudo apt-get install libeccodes-dev
```

### Python Dependencies

It is highly recommended (and often required on modern Linux) to use a virtual environment:

```bash
# Create the environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install the required Python packages
pip install -r requirements.txt
```

Note: If you have trouble installing `cartopy`, you may need additional system libraries like `libgeos-dev` and `libproj-dev`.

## Usage (As Services)

The project is now designed to run as three separate background services.

### 1. Configure Services

The templates are located in the `services/` folder. Before installing, you must edit them to replace `{{USER}}` with your actual Linux username (e.g., `tayta`).

```bash
# Example: Replace {{USER}} with your username
sed -i 's/{{USER}}/your_username/g' services/*.service
```

### 2. Install Services

Copy the files to the Systemd directory and enable them:

```bash
sudo cp services/*.service /etc/systemd/system/
sudo systemctl daemon-reload

# Start the services
sudo systemctl enable --now aigfs-downloader
sudo systemctl enable --now aigfs-processor
sudo systemctl enable --now aigfs-web
```

### 3. Monitoring

You can check the logs of any service using `journalctl`:

```bash
journalctl -u aigfs-downloader -f
```

## Standardized Scales

All maps now use a fixed color scale (VMIN/VMAX) to ensure consistency across different runs and forecast hours. These are centrally managed in `backend/processor.py`.

## Project Structure

- `run_all.py`: Master script to download and process data.
- `app.py`: Flask web server.
- `backend/scraper.py`: Logic for downloading from NOAA NOMADS.
- `backend/processor.py`: Logic for reading GRIB2 and generating maps.
- `templates/index.html`: Dashboard frontend.
- `data/`: Downloaded GRIB2 files (temporary).
- `static/maps/`: Generated map images.
```

