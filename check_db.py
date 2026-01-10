import sqlite3
import pandas as pd
import os

DB_PATH = os.path.join("backend", "ml_data.db")

def check_db():
    if not os.path.exists(DB_PATH):
        print("Database not found.")
        return

    conn = sqlite3.connect(DB_PATH)
    
    # 1. Training Data Summary
    print("\n--- Training Data Summary ---")
    df = pd.read_sql_query("SELECT * FROM training_data ORDER BY timestamp DESC", conn)
    
    if df.empty:
        print("No training data found.")
    else:
        print(f"Total Rows: {len(df)}")
        print("\nColumns:")
        print(df.columns.tolist())
        
        print("\nLatest 5 Entries:")
        print(df.head(5)[['timestamp', 'obs_temp', 'gfs_temp', 'gfs_run_date', 'gfs_fhr']].to_string(index=False))
        
        # Stats
        print("\nDate Range:")
        print(f"Start: {df['timestamp'].min()}")
        print(f"End:   {df['timestamp'].max()}")

    # 2. Model Coefficients
    print("\n--- Model Coefficients ---")
    try:
        model_df = pd.read_sql_query("SELECT * FROM model_coefficients", conn)
        if model_df.empty:
            print("No models trained yet.")
        else:
            print(model_df.to_string(index=False))
    except Exception as e:
        print(f"Could not read model coefficients: {e}")

    conn.close()

if __name__ == "__main__":
    check_db()
