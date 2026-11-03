#  先构造user-item，划分train/test集
#  定义relation2id.txt
#  根据user/item构造entity2id.txt
#  然后构造train/test 三元组
#  去掉重复三元组

import os
from setting import Setting
from dataloader import PoiDataloader
from math import radians, cos, sin, asin, sqrt
from tqdm import tqdm
from collections import defaultdict
from constant import SCHEME
from dataset import Split
import numpy as np
from sklearn.neighbors import BallTree


def generate_checkin_files(train_file, test_file, valid_file):
    with open(train_file, "w+") as f_train:
        for user, locs in zip(train_dataset.users, train_dataset.user_trajectories):
            locs.insert(0, user)
            for train_elem in locs:
                f_train.write(str(train_elem) + " ")
            f_train.write("\n")

    with open(test_file, "w+") as f_test:
        for user, locs in zip(test_dataset.users, test_dataset.user_trajectories):
            locs.insert(0, user)
            for test_elem in locs:
                f_test.write(str(test_elem) + " ")
            f_test.write("\n")

    with open(valid_file, "w+") as f_valid:
        for user, locs in zip(val_dataset.users, val_dataset.user_trajectories):
            locs.insert(0, user)
            for valid_elem in locs:
                f_valid.write(str(valid_elem) + " ")
            f_valid.write("\n")

    print("Successfully generate train/test/valid checkins!")


def generate_entity_file(entity2id_file):  # 构造entity2id文件
    if os.path.exists(entity2id_file):
        print("entity2id.txt has existed!!!")
        return
    with open(entity2id_file, "w+") as f:
        # users_count = len(users)
        for i in range(users_count):
            f.write(str(i) + " ")
            f.write(str(i) + " ")
            f.write("\n")
        for value in poi2id.values():
            poi_id = value + users_count
            f.write(str(poi_id) + " ")
            f.write(str(poi_id) + " ")
            f.write("\n")
    print("Successfully generate entity2id.txt!")


def generate_triplets(train_file, train_triplets_file):  # 构造train/test 三元组
    f_train_triplets = open(train_triplets_file, "w+")
    print("Construct interact relation and temporal relation......")
    with tqdm(total=users_count) as bar:
        with open(train_file, "r") as f:
            for line in f.readlines():
                line = line.strip().split(" ")  # 以空格形式分隔
                user_id = line[0]  # str
                poi_ids = line[1:]  # poi2id字典中的org_id，在entity2id中对应id是org_id + users_count

                # 构建interact关系

                for poi_id in poi_ids:
                    poi_id = str(int(poi_id) + users_count)
                    f_train_triplets.write(user_id + "\t")
                    f_train_triplets.write(poi_id + "\t")
                    f_train_triplets.write("0" + "\n")  # 0代表interact relation

                # 构建temporal关系  相邻poi相连
                # print('Construct temporal relation......')
                for i in range(len(poi_ids) - 1):
                    poi_prev = str(int(poi_ids[i]) + users_count)
                    poi_next = str(int(poi_ids[i + 1]) + users_count)
                    if poi_prev != poi_next:
                        f_train_triplets.write(poi_prev + "\t")
                        f_train_triplets.write(poi_next + "\t")
                        f_train_triplets.write("1" + "\n")  # 1代表temporal relation
                bar.update(1)

    # 构建spatial关系  两个poi的距离小于距离阈值lambda_d，就相连
    print("Construct spatial relation......")
    pois_items = list(poi2gps.items())
    poi_org_ids = np.array([item[0] + users_count for item in pois_items])
    coords = np.array([item[1] for item in pois_items])

    # BallTree haversine requires coordinates in RADIANS
    coords_rad = np.radians(coords)
    tree = BallTree(coords_rad, metric="haversine")

    EARTH_RADIUS = 6371.0

    # 方案1
    if SCHEME == 1:
        lambda_d = 0.2  # 距离阈值为0.2千米
        radius_rad = lambda_d / EARTH_RADIUS

        # query_radius returns indices of all points within distance
        indices = tree.query_radius(coords_rad, r=radius_rad)

        for i, neighbors in enumerate(tqdm(indices)):
            this_poi = poi_org_ids[i]
            output = []
            for neighbor_idx in neighbors:
                if i < neighbor_idx:  # Ensure we only process each pair once
                    neighbor_poi = poi_org_ids[neighbor_idx]
                    output.append(f"{this_poi}\t{neighbor_poi}\t2\n")
                    output.append(f"{neighbor_poi}\t{this_poi}\t2\n")
            f_train_triplets.write("".join(output))

    # 方案2
    else:
        lambda_d = 3  # 距离阈值为3千米, 再取top k, 即双重限制
        max_k = 51  # top 50 + 1 (itself)
        radius_rad = lambda_d / EARTH_RADIUS
        # query returns (distances, indices) for top k nearest
        for i in tqdm(range(len(coords_rad))):
            this_poi = poi_org_ids[i]
            # Find neighbors within 3km, then pick top 50
            dist, ind = tree.query(coords_rad[i : i + 1], k=max_k)

            output = []
            for d, idx in zip(dist[0], ind[0]):
                neighbor_poi = poi_org_ids[idx]
                # Check if within 3km and not itself
                if d <= radius_rad and this_poi != neighbor_poi:
                    output.append(f"{this_poi}\t{neighbor_poi}\t2\n")
                    output.append(f"{neighbor_poi}\t{this_poi}\t2\n")

            # Since Scheme 2 is top-k per POI, it's naturally directed/symmetric here
            f_train_triplets.write("".join(output[:100]))  # 50 pairs * 2 lines

    # NOTE: ignore friend relation construction: https://github.com/kevin-xuan/Graph-Flashback/issues/9
    # 构建friend关系  互为朋友的user相连  这个train/test会重复构造一次,可以选择生成一个friend_triplet文件,然后再将其内容放入train/test
    # 但因为数量很少,构造很快,所以放在一起
    # print("Construct friend relation......")
    # with open(friendship_file, "r") as f_friend:
    #     for friend_line in f_friend.readlines():
    #         tokens = friend_line.strip("\n").split("\t")
    #         if user2id.get(int(tokens[0])) and user2id.get(int(tokens[1])):  # only focus on active users
    #             user_id1 = str(user2id.get(int(tokens[0])))
    #             user_id2 = str(user2id.get(int(tokens[1])))
    #             f_train_triplets.write(user_id1 + "\t")
    #             f_train_triplets.write(user_id2 + "\t")
    #             f_train_triplets.write("3" + "\n")  # 2代表friend relation
    #             # friend relation是对称的
    #             f_train_triplets.write(user_id2 + "\t")
    #             f_train_triplets.write(user_id1 + "\t")
    #             f_train_triplets.write("3" + "\n")
    f_train_triplets.close()


# 可能会重复添加triplet，所以要进行去重操作，得到最终train triplets
def filter_train_triplet(read_file, write_file):
    filter_set = set()
    print("Filter repeated triplets......")
    count = 0
    with open(read_file, "r") as f_read, open(write_file, "w+") as f_write:
        for f_read_line in f_read.readlines():
            count += 1
            f_read_line = f_read_line.strip("\n")
            if f_read_line not in filter_set:
                filter_set.add(f_read_line)
        for triplet in filter_set:
            f_write.write(triplet + "\n")
    print("Original triplets: ", count)
    print("Final triplets: ", len(filter_set))
    return filter_set


# 去重且保证test triplets与train triplets不同
def filter_test_triplet(read_file, write_file, train_filter_set):
    filter_set = set()
    print("Filter repeated triplets......")
    count = 0
    with open(read_file, "r") as f_read, open(write_file, "w+") as f_write:
        for f_read_line in f_read.readlines():
            count += 1
            f_read_line = f_read_line.strip("\n")
            if f_read_line not in filter_set and f_read_line not in train_filter_set:
                filter_set.add(f_read_line)
        for triplet in filter_set:
            f_write.write(triplet + "\n")
    print("Original triplets: ", count)
    print("Final triplets: ", len(filter_set))


if __name__ == "__main__":
    # parse settings
    setting = Setting()
    setting.parse()
    print(setting)

    # load dataset
    poi_loader = PoiDataloader(setting.max_users, setting.min_checkins)  # 0， 5*20+1
    poi_loader.read(setting.dataset_train_file, setting.dataset_val_file, setting.dataset_test_file)

    train_dataset = poi_loader.create_dataset(setting.sequence_length, setting.batch_size, Split.TRAIN)
    val_dataset = poi_loader.create_dataset(setting.sequence_length, setting.batch_size, Split.VAL)
    test_dataset = poi_loader.create_dataset(setting.sequence_length, setting.batch_size, Split.TEST)

    # poi_loader.read(setting.dataset_file)
    print("Active POI number: ", poi_loader.locations())
    print("Active User number: ", poi_loader.user_count())
    print("Total Checkins number: ", poi_loader.checkins_count())

    # global mapping
    user2id = poi_loader.user2id
    poi2id = poi_loader.poi2id
    poi2gps = poi_loader.poi2gps
    users_count = len(poi_loader.users)

    data_path = "./dataset/{}/{}_scheme{}".format(setting.city, setting.city, SCHEME)
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # generate train & test file

    train_file = os.path.join(data_path, "train.txt")
    test_file = os.path.join(data_path, "test.txt")
    valid_file = os.path.join(data_path, "valid.txt")
    entity2id_file = os.path.join(data_path, "entity2id.txt")

    train_triplets = os.path.join(data_path, "train_triplets.txt")
    test_triplets = os.path.join(data_path, "test_triplets.txt")
    valid_triplets = os.path.join(data_path, "valid_triplets.txt")

    final_train_triplets = os.path.join(data_path, "final_train_triplets.txt")
    final_test_triplets = os.path.join(data_path, "final_test_triplets.txt")
    final_valid_triplets = os.path.join(data_path, "final_valid_triplets.txt")

    print("Generate train/test checkins......")
    generate_checkin_files(train_file, test_file, valid_file)

    print("Generate entity2id......")
    generate_entity_file(entity2id_file)

    print("Construct train triplets......")
    generate_triplets(train_file, train_triplets)
    print("Construct test triplets......")
    generate_triplets(test_file, test_triplets)
    print("Construct valid triplets......")
    generate_triplets(valid_file, valid_triplets)

    train_filter_triplets = filter_train_triplet(train_triplets, final_train_triplets)
    filter_test_triplet(test_triplets, final_test_triplets, train_filter_triplets)
    filter_test_triplet(valid_triplets, final_valid_triplets, train_filter_triplets)
