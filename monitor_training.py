"""
Real-time Training Monitor
Watch training progress without TensorBoard
"""

import os
import time
import pandas as pd
from pathlib import Path

def find_latest_checkpoint():
    """Find the most recent checkpoint folder"""
    checkpoint_dir = Path(r'c:\Users\likit\OneDrive\Documents\projects\glucamo\checkpoints')
    if not checkpoint_dir.exists():
        return None

    folders = [f for f in checkpoint_dir.iterdir() if f.is_dir()]
    if not folders:
        return None

    latest = max(folders, key=lambda f: f.stat().st_mtime)
    return latest

def monitor_training(checkpoint_path=None, refresh_seconds=5):
    """Monitor training progress from CSV log"""

    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path is None:
            print("No checkpoint folder found. Start training first!")
            return

    log_file = Path(checkpoint_path) / 'training_log.csv'

    print(f"Monitoring: {log_file}")
    print("Press Ctrl+C to stop\n")

    last_size = 0

    try:
        while True:
            if log_file.exists():
                current_size = log_file.stat().st_size

                if current_size != last_size:
                    try:
                        df = pd.read_csv(log_file)

                        # Clear screen (Windows)
                        os.system('cls')

                        print("="*80)
                        print(f"TRAINING MONITOR - {checkpoint_path.name}")
                        print("="*80)
                        print(f"\nEpoch: {len(df)}/{50}")  # Assuming 50 epochs

                        if len(df) > 0:
                            latest = df.iloc[-1]

                            print(f"\nLatest Epoch ({len(df)}):")
                            print(f"  Loss:          {latest['loss']:.4f}")
                            print(f"  Accuracy:      {latest['accuracy']:.4f} ({latest['accuracy']*100:.2f}%)")
                            print(f"  Val Loss:      {latest['val_loss']:.4f}")
                            print(f"  Val Accuracy:  {latest['val_accuracy']:.4f} ({latest['val_accuracy']*100:.2f}%)")

                            if 'val_auc' in latest:
                                print(f"  Val AUC:       {latest['val_auc']:.4f}")
                            if 'val_recall' in latest:
                                print(f"  Val Recall:    {latest['val_recall']:.4f}")
                            if 'val_precision' in latest:
                                print(f"  Val Precision: {latest['val_precision']:.4f}")

                            print(f"\nBest Results So Far:")
                            print(f"  Best Val Acc:  {df['val_accuracy'].max():.4f} (Epoch {df['val_accuracy'].idxmax() + 1})")
                            if 'val_auc' in df.columns:
                                print(f"  Best Val AUC:  {df['val_auc'].max():.4f} (Epoch {df['val_auc'].idxmax() + 1})")

                            # Show trend
                            if len(df) >= 2:
                                acc_trend = df['val_accuracy'].iloc[-1] - df['val_accuracy'].iloc[-2]
                                trend_icon = "[UP]" if acc_trend > 0 else "[DOWN]" if acc_trend < 0 else "[STABLE]"
                                print(f"\nTrend: {trend_icon} Val Accuracy {'improved' if acc_trend > 0 else 'decreased' if acc_trend < 0 else 'unchanged'}")

                            print("\n" + "="*80)
                            print(f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                            print(f"Refreshing every {refresh_seconds} seconds... (Ctrl+C to stop)")

                        last_size = current_size

                    except pd.errors.EmptyDataError:
                        print("Waiting for training data...")
                    except Exception as e:
                        print(f"Error reading log: {e}")
            else:
                print(f"Waiting for training to start...\nLooking for: {log_file}")

            time.sleep(refresh_seconds)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

if __name__ == "__main__":
    import sys

    # Check if checkpoint path provided
    if len(sys.argv) > 1:
        checkpoint_path = Path(sys.argv[1])
    else:
        checkpoint_path = None

    monitor_training(checkpoint_path, refresh_seconds=5)
