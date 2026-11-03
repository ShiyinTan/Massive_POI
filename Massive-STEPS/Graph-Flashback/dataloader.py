import os.path
import sys
from datetime import datetime
import token

import pandas as pd

from dataset import PoiDataset, Usage, Split


class PoiDataloader:
    """Creates datasets from our prepared Gowalla/Foursquare data files.
    The file consist of one check-in per line in the following format (tab separated):

    <user-id> <timestamp> <latitude> <longitude> <location-id>

    Check-ins for the same user have to be on continuous lines.
    Ids for users and locations are recreated and continuous from 0.
    """

    def __init__(self, max_users=0, min_checkins=0):
        """max_users limits the amount of users to load.
        min_checkins discards users with less than this amount of checkins.
        """

        self.max_users = max_users
        self.min_checkins = min_checkins

        self.user2id = {}
        self.poi2id = {}
        self.poi2gps = {}

        self.users = []
        self.times = []
        self.timeslots = []
        self.coords = []

        self.train_data = {"users": [], "times": [], "coords": [], "locs": [], "timeslots": []}
        self.val_data = {"users": [], "times": [], "coords": [], "locs": [], "timeslots": []}
        self.test_data = {"users": [], "times": [], "coords": [], "locs": [], "timeslots": []}

    def create_dataset(self, sequence_length, batch_size, split, usage=Usage.MAX_SEQ_LENGTH, custom_seq_count=1):
        data = self.train_data if split == Split.TRAIN else self.val_data if split == Split.VAL else self.test_data
        return PoiDataset(
            data["users"].copy(),
            data["times"].copy(),
            data["timeslots"].copy(),
            data["coords"].copy(),
            data["locs"].copy(),
            batch_size,
            len(self.poi2id),
        )

    def user_count(self):
        return len(self.user2id)

    def locations(self):
        return len(self.poi2id)

    def checkins_count(self):
        return (
            sum(len(locs) for locs in self.train_data["locs"])
            + sum(len(locs) for locs in self.val_data["locs"])
            + sum(len(locs) for locs in self.test_data["locs"])
        )

    def read(self, train_file, val_file, test_file):
        df_train = pd.read_csv(train_file)
        df_val = pd.read_csv(val_file)
        df_test = pd.read_csv(test_file)

        for _df in [df_train, df_val, df_test]:
            _df["latitude"] = _df["latitude"].fillna(_df["venue_city_latitude"])
            _df["longitude"] = _df["longitude"].fillna(_df["venue_city_longitude"])
            _df["timestamp"] = pd.to_datetime(_df["timestamp"]).astype("int64") // 10**9
            _df["timeslot"] = (
                pd.to_datetime(_df["timestamp"], unit="s").dt.hour
                + pd.to_datetime(_df["timestamp"], unit="s").dt.weekday * 24
            )

        all_df = pd.concat([df_train, df_val, df_test])
        self.poi2id = {val: i for i, val in enumerate(all_df["venue_id"].unique())}
        self.poi2gps = {
            self.poi2id[row["venue_id"]]: (row["latitude"], row["longitude"]) for _, row in all_df.iterrows()
        }
        self.user2id = {val: i for i, val in enumerate(all_df["user_id"].unique())}
        self.users = list(self.user2id.values())

        def process_split(df, target_dict):
            # Group by trail_id to keep trajectories together
            for (user_id, trail_id), group in df.groupby(["user_id", "trail_id"]):
                group = group.sort_values("timestamp")
                if len(group) < 2:
                    continue  # Need at least input + label

                target_dict["users"].append(self.user2id[user_id])
                target_dict["locs"].append([self.poi2id[v] for v in group["venue_id"]])
                target_dict["times"].append(group["timestamp"].tolist())
                target_dict["coords"].append(list(zip(group["latitude"], group["longitude"])))
                target_dict["timeslots"].append(group["timeslot"].tolist())

        process_split(df_train, self.train_data)
        process_split(df_val, self.val_data)
        process_split(df_test, self.test_data)
