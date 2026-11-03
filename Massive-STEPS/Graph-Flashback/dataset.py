import logging
import random
from enum import Enum
import torch
from torch.utils.data import Dataset


class Split(Enum):
    """Defines whether to split for train or test."""

    TRAIN = 0
    VAL = 1
    TEST = 2


class Usage(Enum):
    """
    Each user has a different amount of sequences. The usage defines
    how many sequences are used:

    MAX: each sequence of any user is used (default)
    MIN: only as many as the minimal user has
    CUSTOM: up to a fixed amount if available.

    The unused sequences are discarded. This setting applies after the train/test split.
    """

    MIN_SEQ_LENGTH = 0
    MAX_SEQ_LENGTH = 1
    CUSTOM = 2


class PoiDataset(Dataset):
    def __init__(self, users, times, time_slots, coords, locs, batch_size, loc_count):
        self.users = users
        self.batch_size = batch_size
        self.loc_count = loc_count
        self.time_slots = time_slots

        # We store trajectories as they come from the loader (pre-grouped by trail_id)
        self.user_trajectories = locs  # List of lists (trajectories)
        self.user_times = times
        self.user_coords = coords
        self.user_ids = users

        self.capacity = len(self.user_trajectories)
        self.reset()

    def reset(self):
        self.next_traj_idx = 0

    def __len__(self):
        return (self.capacity + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        batch_locs, batch_times, batch_timeslots, batch_coords = [], [], [], []
        batch_labels, batch_users, lengths = [], [], []
        batch_traj_ids = []
        reset_h = [True] * self.batch_size  # Every trajectory is independent now

        base_traj_idx = idx * self.batch_size

        for i in range(self.batch_size):
            traj_idx = (base_traj_idx + i) % self.capacity

            loc = self.user_trajectories[traj_idx]
            t = self.user_times[traj_idx]
            t_slots = self.time_slots[traj_idx]
            c = self.user_coords[traj_idx]
            u = self.user_ids[traj_idx]

            # In trajectory prediction:
            # Input: Traj[:-1], Label: Traj[1:]
            # But per your requirement: for eval, label is the last POI.
            # To keep it consistent for RNN training:
            batch_locs.append(torch.tensor(loc[:-1]))
            batch_labels.append(torch.tensor(loc[1:]))
            batch_times.append(torch.tensor(t[:-1]))
            batch_timeslots.append(torch.tensor(t_slots[:-1]))
            batch_coords.append(torch.tensor(c[:-1]))
            batch_users.append(u)
            lengths.append(len(loc) - 1)
            batch_traj_ids.append(traj_idx)

            # Move to next trajectory in global list
            self.next_traj_idx += 1

        # Pad sequences with -100
        x = torch.nn.utils.rnn.pad_sequence(batch_locs, batch_first=False, padding_value=self.loc_count)
        y = torch.nn.utils.rnn.pad_sequence(batch_labels, batch_first=False, padding_value=-100)
        t = torch.nn.utils.rnn.pad_sequence(batch_times, batch_first=False, padding_value=0)
        t_slots = torch.nn.utils.rnn.pad_sequence(batch_timeslots, batch_first=False, padding_value=0)
        # Coords padding (0,0)
        s = torch.nn.utils.rnn.pad_sequence(batch_coords, batch_first=False, padding_value=0)

        return (
            x,
            t,
            t_slots,
            s,
            y,
            torch.tensor(lengths),
            torch.tensor(batch_users),
            reset_h,
            torch.tensor(batch_traj_ids),
        )
