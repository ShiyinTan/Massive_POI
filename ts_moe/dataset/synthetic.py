"""Small trajectory tables with the Massive-STEPS column layout.

Used by the smoke config and tests. Not a substitute for the city CSVs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_frames(
    n_users: int = 12,
    n_venues: int = 10,
    trails_per_split: tuple[int, int, int] = (24, 8, 8),
    length: int = 6,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    """Users prefer a home venue; time and location follow that venue.

    The preference is strong enough that a SURD estimate on the training
    split attributes next-POI information to the user source.
    """
    rng = np.random.default_rng(seed)
    frames = {}
    offset = 0
    for split, n_trails in zip(("train", "val", "test"), trails_per_split):
        rows = []
        for t in range(n_trails):
            user = int(t % n_users)
            trail_id = f"{split}_{offset + t}"
            home = user % n_venues
            for step in range(length):
                if rng.random() < 0.8:
                    venue = home
                else:
                    venue = int(rng.integers(0, n_venues))
                hour = (8 + home) % 24
                lat = 39.9 + 0.01 * venue
                lon = 116.4 + 0.01 * venue
                rows.append(
                    {
                        "trail_id": trail_id,
                        "user_id": f"U{user}",
                        "venue_id": venue,
                        "latitude": lat,
                        "longitude": lon,
                        "venue_category": f"cat{venue % 4}",
                        "venue_city_latitude": 39.9,
                        "venue_city_longitude": 116.4,
                        "timestamp": f"2020-01-{(step % 27) + 1:02d} {hour:02d}:00:00",
                    }
                )
        frames[split] = pd.DataFrame(rows)
        offset += n_trails
    return frames
