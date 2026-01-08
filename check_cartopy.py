try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    print("Cartopy is installed and working!")
except ImportError:
    print("Cartopy is NOT installed.")
except Exception as e:
    print(f"Cartopy failed: {e}")
