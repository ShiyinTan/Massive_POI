import os
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

import random


class TrajDataset(Dataset):
    def __init__(self, data_root, split, B, T, few_shot):
        self.B = B
        self.T = T
        self.split = split
        self.few_shot = few_shot

        # load the shards
        shards = os.listdir(data_root)
        shards = [s for s in shards if any(x in s for x in split)]
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        print(f"found {len(shards)} shards for split {split}")

        self.data_city = defaultdict()
        for shard in self.shards:
            self.data_city[shard] = self.load_traj(shard)
        self.data = []

        batches = {
            shard: [
                self.data_city[shard][i : i + self.B]
                for i in range(0, len(self.data_city[shard]) - len(self.data_city[shard]) % self.B, self.B)
            ] # drop last batch to avoid cross-city batches
            for shard in self.shards
        }

        total_batches = sum(len(batches[shard]) for shard in self.shards)

        shard_indices = {shard: 0 for shard in self.shards}

        remaining_batches = {shard: len(batches[shard]) for shard in self.shards}

        for _ in range(total_batches):

            shard = random.choice(self.shards)

            if remaining_batches[shard] > 0:

                batch = batches[shard][shard_indices[shard]]

                self.data.extend(batch)

                shard_indices[shard] += 1
                remaining_batches[shard] -= 1

    def load_traj(self, filename):
        with open(filename, "r", encoding="utf-8") as file:
            lines = file.readlines()
            data = []
            for line in lines:
                traj = []
                line = line.strip()
                userid = line.split(" ")[0]
                trajs = line.split(" ")[1]
                parts = trajs.strip().split(";")
                for part in parts:
                    if part:
                        location, day, time = part.split(",")
                        day = int(day)
                        time = int(time)
                        traj.append([int(location) + 2, time])

                traj.append([int(1), int(0)])
                for _ in range(self.T + 1 - len(traj)):
                    traj.append([int(0), int(0)])
                traj = torch.tensor(traj, dtype=torch.long)
                data.append([traj, "_".join(filename.split("/")[-1].split("_")[:-1])])
            if self.few_shot:
                length = int(self.few_shot * len(data))
                data = data[:length]
            return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        traj, file = self.data[idx]
        x = traj[:-1, 0]
        y = traj[1:, 0]
        ts_his = traj[:-1, 1]
        return x, y, ts_his, file


def get_dataloader(data_root, split, B, T, few_shot):
    dataset = TrajDataset(data_root, split, B, T, few_shot)
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=B, shuffle=False)
    return dataloader


# 示例用法
if __name__ == "__main__":
    data_root = "traj_dataset/massive_steps/val"
    city = ["melbourne"]  # 或 'valid', 'test'
    B = 16
    T = 144
    dataloader = get_dataloader(data_root, city, B, T, few_shot=1.0)
    for batch_no, train_batch in enumerate(dataloader, start=1):
        print(train_batch[0].size())
        print(train_batch[1].size())
        print(train_batch[2].size())
        print(train_batch[3][0])
        break
