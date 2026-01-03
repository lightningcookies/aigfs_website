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

## Usage

### 1. Download and Process Data

Run the main script to fetch the latest data and generate the maps. By default, it looks for the run specified in your request (20260103 00Z).

```bash
python run_all.py
```

This will:
- Download GRIB2 files into the `data/` directory.
- Generate PNG map images into `static/maps/`.

### 2. Start the Web Server

Run the Flask application:

```bash
python app.py
```

The website will be accessible on your local network at `http://<your-server-ip>:5000`.

## Project Structure

- `run_all.py`: Master script to download and process data.
- `app.py`: Flask web server.
- `backend/scraper.py`: Logic for downloading from NOAA NOMADS.
- `backend/processor.py`: Logic for reading GRIB2 and generating maps.
- `templates/index.html`: Dashboard frontend.
- `data/`: Downloaded GRIB2 files (temporary).
- `static/maps/`: Generated map images.
```

