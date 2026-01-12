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

def save_model(cursor, var_name, model, rmse, count):
    timestamp = datetime.now().isoformat()
    cursor.execute('''
        INSERT OR REPLACE INTO model_coefficients 
        (variable, slope, intercept, rmse, last_updated, sample_count)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (var_name, float(model.coef_[0]), float(model.intercept_), float(rmse), timestamp, count))

def train_models():
    if not os.path.exists(DB_PATH):
        logger.warning("Database not found. Skipping training.")
        return

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Check if table exists
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='model_coefficients'")
    if not c.fetchone():
        logger.warning("model_coefficients table not found. Ensure ml_collector.py has run at least once.")
        conn.close()
        return

    # 1. Temperature Model
    c.execute("SELECT obs_temp, gfs_temp FROM training_data WHERE obs_temp IS NOT NULL AND gfs_temp IS NOT NULL")
    rows = c.fetchall()
    if len(rows) >= 10:
        data = np.array(rows)
        X = data[:, 1].reshape(-1, 1) # AIGFS
        y = data[:, 0]               # OBS
        model = LinearRegression().fit(X, y)
        rmse = np.sqrt(mean_squared_error(y, model.predict(X)))
        save_model(c, 'temperature', model, rmse, len(y))
        logger.info(f"Updated Temperature model (RMSE: {rmse:.2f}C from {len(y)} samples)")
    else:
        logger.warning(f"Not enough temperature data ({len(rows)} samples)")

    # 2. Wind U-component
    c.execute("SELECT obs_u10, gfs_u10 FROM training_data WHERE obs_u10 IS NOT NULL AND gfs_u10 IS NOT NULL")
    rows = c.fetchall()
    if len(rows) >= 10:
        data = np.array(rows)
        X = data[:, 1].reshape(-1, 1)
        y = data[:, 0]
        model = LinearRegression().fit(X, y)
        rmse = np.sqrt(mean_squared_error(y, model.predict(X)))
        save_model(c, 'u10', model, rmse, len(y))
        logger.info(f"Updated Wind U model (RMSE: {rmse:.2f}m/s from {len(y)} samples)")

    # 3. Wind V-component
    c.execute("SELECT obs_v10, gfs_v10 FROM training_data WHERE obs_v10 IS NOT NULL AND gfs_v10 IS NOT NULL")
    rows = c.fetchall()
    if len(rows) >= 10:
        data = np.array(rows)
        X = data[:, 1].reshape(-1, 1)
        y = data[:, 0]
        model = LinearRegression().fit(X, y)
        rmse = np.sqrt(mean_squared_error(y, model.predict(X)))
        save_model(c, 'v10', model, rmse, len(y))
        logger.info(f"Updated Wind V model (RMSE: {rmse:.2f}m/s from {len(y)} samples)")

    # 4. Pressure Model
    c.execute("SELECT obs_pressure, gfs_pressure FROM training_data WHERE obs_pressure IS NOT NULL AND gfs_pressure IS NOT NULL")
    rows = c.fetchall()
    if len(rows) >= 10:
        data = np.array(rows)
        X = data[:, 1].reshape(-1, 1)
        y = data[:, 0]
        model = LinearRegression().fit(X, y)
        rmse = np.sqrt(mean_squared_error(y, model.predict(X)))
        save_model(c, 'pressure', model, rmse, len(y))
        logger.info(f"Updated Pressure model (RMSE: {rmse:.2f}Pa from {len(y)} samples)")

    conn.commit()
    conn.close()

if __name__ == "__main__":
    train_models()
