from pathlib import Path
import time

import pandas as pd


PREDICTIONS_PATH = Path("results/realtime/predictions.csv")

df = pd.read_csv(PREDICTIONS_PATH)

for i in range(200):
    row = df.sample(1).copy()

    now = pd.Timestamp.now()
    row["processed_at"] = now.isoformat()
    row["event_time"] = now.isoformat()
    row["timestamp"] = now

    row["particle_index"] = int(df["particle_index"].max()) + i + 1

    row.to_csv(
        PREDICTIONS_PATH,
        mode="a",
        header=False,
        index=False,
    )

    print(f"Appended simulated particle {i + 1}")
    time.sleep(1)