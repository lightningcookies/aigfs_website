import sqlite3
import numpy as np
import os
import logging
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_data.db")

def train_models():
    if not os.path.exists(DB_PATH):
        logger.warning("Database not found. Skipping training.")
        return

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Fetch Data
    # Format: obs_temp, gfs_temp, obs_wind, gfs_wind
    c.execute("SELECT obs_temp, gfs_temp, obs_wind, gfs_wind FROM training_data WHERE obs_temp IS NOT NULL AND gfs_temp IS NOT NULL")
    rows = c.fetchall()
    
    if len(rows) < 10:
        logger.warning(f"Not enough data to train (only {len(rows)} samples). Need at least 10.")
        conn.close()
        return
        
    data = np.array(rows)
    obs_temp = data[:, 0]
    gfs_temp = data[:, 1]
    obs_wind = data[:, 2]
    gfs_wind = data[:, 3]
    
    # Train Temperature Model
    # We want to predict Bias (Obs - GFS) based on GFS? 
    # Or just predict Obs from GFS? 
    # Let's predict Corrected Value (Obs) from GFS.
    # Model: Obs = m * GFS + b
    
    # Temperature
    X_temp = gfs_temp.reshape(-1, 1)
    y_temp = obs_temp
    
    model_t = LinearRegression()
    model_t.fit(X_temp, y_temp)
    
    pred_t = model_t.predict(X_temp)
    rmse_t = np.sqrt(mean_squared_error(y_temp, pred_t))
    
    # Wind
    # Filter NaNs if any (though SQL query handled main nulls, sometimes wind is null differently)
    # Simple check
    valid_wind = ~np.isnan(obs_wind) & ~np.isnan(gfs_wind)
    if np.sum(valid_wind) > 10:
        X_wind = gfs_wind[valid_wind].reshape(-1, 1)
        y_wind = obs_wind[valid_wind]
        
        model_w = LinearRegression()
        model_w.fit(X_wind, y_wind)
        
        pred_w = model_w.predict(X_wind)
        rmse_w = np.sqrt(mean_squared_error(y_wind, pred_w))
    else:
        model_w = None
        rmse_w = 0.0

    # Save Results
    timestamp = datetime.now().isoformat()
    
    # Save Temp
    c.execute('''
        INSERT OR REPLACE INTO model_coefficients 
        (variable, slope, intercept, rmse, last_updated, sample_count)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', ('temperature', float(model_t.coef_[0]), float(model_t.intercept_), float(rmse_t), timestamp, len(y_temp)))
    
    # Save Wind
    if model_w:
        c.execute('''
            INSERT OR REPLACE INTO model_coefficients 
            (variable, slope, intercept, rmse, last_updated, sample_count)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', ('wind_speed', float(model_w.coef_[0]), float(model_w.intercept_), float(rmse_w), timestamp, np.sum(valid_wind)))
    
    conn.commit()
    conn.close()
    
    logger.info(f"Training Complete. Temp RMSE: {rmse_t:.2f}C, Wind RMSE: {rmse_w:.2f}m/s")

if __name__ == "__main__":
    train_models()
