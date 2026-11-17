import os
import pickle
import pandas as pd
from datetime import datetime
import os.path as osp
from utils import Cfg, get_root_dir
from preprocess import (
    ignore_first,
    only_keep_last,
    id_encode,
    remove_unseen_user_poi,
    FileReader,
    generate_hypergraph_from_file,
)
import logging


def preprocess_nyc(path: bytes, preprocessed_path: bytes) -> pd.DataFrame:
    raw_path = osp.join(path, "raw")

    df_train = pd.read_csv(osp.join(raw_path, "NYC_train.csv"))
    df_val = pd.read_csv(osp.join(raw_path, "NYC_val.csv"))
    df_test = pd.read_csv(osp.join(raw_path, "NYC_test.csv"))
    df_train["SplitTag"] = "train"
    df_val["SplitTag"] = "validation"
    df_test["SplitTag"] = "test"
    df = pd.concat([df_train, df_val, df_test])
    df.columns = [
        "UserId",
        "PoiId",
        "PoiCategoryId",
        "PoiCategoryCode",
        "PoiCategoryName",
        "Latitude",
        "Longitude",
        "TimezoneOffset",
        "UTCTime",
        "UTCTimeOffset",
        "UTCTimeOffsetWeekday",
        "UTCTimeOffsetNormInDayTime",
        "pseudo_session_trajectory_id",
        "UTCTimeOffsetNormDayShift",
        "UTCTimeOffsetNormRelativeTime",
        "SplitTag",
    ]

    # data transformation
    df["trajectory_id"] = df["pseudo_session_trajectory_id"]
    df["UTCTimeOffset"] = df["UTCTimeOffset"].apply(lambda x: datetime.strptime(x[:19], "%Y-%m-%d %H:%M:%S"))
    df["UTCTimeOffsetEpoch"] = df["UTCTimeOffset"].apply(lambda x: x.strftime("%s"))
    df["UTCTimeOffsetWeekday"] = df["UTCTimeOffset"].apply(lambda x: x.weekday())
    df["UTCTimeOffsetHour"] = df["UTCTimeOffset"].apply(lambda x: x.hour)
    df["UTCTimeOffsetDay"] = df["UTCTimeOffset"].apply(lambda x: x.strftime("%Y-%m-%d"))
    df["UserRank"] = df.groupby("UserId")["UTCTimeOffset"].rank(method="first")
    df = df.sort_values(by=["UserId", "UTCTimeOffset"], ascending=True)

    # id encoding
    df["check_ins_id"] = df["UTCTimeOffset"].rank(ascending=True, method="first") - 1
    traj_id_le, padding_traj_id = id_encode(df, df, "pseudo_session_trajectory_id")

    df_train = df[df["SplitTag"] == "train"]
    poi_id_le, padding_poi_id = id_encode(df_train, df, "PoiId")
    poi_category_le, padding_poi_category = id_encode(df_train, df, "PoiCategoryId")
    user_id_le, padding_user_id = id_encode(df_train, df, "UserId")
    hour_id_le, padding_hour_id = id_encode(df_train, df, "UTCTimeOffsetHour")
    weekday_id_le, padding_weekday_id = id_encode(df_train, df, "UTCTimeOffsetWeekday")

    # save mapping logic
    with open(osp.join(preprocessed_path, "label_encoding.pkl"), "wb") as f:
        pickle.dump(
            [
                poi_id_le,
                poi_category_le,
                user_id_le,
                hour_id_le,
                weekday_id_le,
                padding_poi_id,
                padding_poi_category,
                padding_user_id,
                padding_hour_id,
                padding_weekday_id,
            ],
            f,
        )

    # ignore the first for train/validate/test and keep the last for validata/test
    df = ignore_first(df)
    df = only_keep_last(df)
    return df


def preprocess_tky_ca(cfg: Cfg, path: bytes) -> pd.DataFrame:
    if cfg.dataset_args.dataset_name == "tky":
        raw_file = "dataset_TSMC2014_TKY.txt"
    else:
        raw_file = "dataset_gowalla_ca_ne.csv"

    FileReader.root_path = path
    data = FileReader.read_dataset(raw_file, cfg.dataset_args.dataset_name)
    data = FileReader.do_filter(data, cfg.dataset_args.min_poi_freq, cfg.dataset_args.min_user_freq)
    data = FileReader.split_train_test(data)

    # for ca dataset, after one round of filter, there still be many low frequency pois and users, so do twice
    if cfg.dataset_args.dataset_name == "ca":
        data = FileReader.do_filter(data, cfg.dataset_args.min_poi_freq, cfg.dataset_args.min_user_freq)
        data = FileReader.split_train_test(data)

    data = FileReader.generate_id(
        data,
        cfg.dataset_args.session_time_interval,
        cfg.dataset_args.do_label_encode,
        cfg.dataset_args.only_last_metric,
    )
    return data


def preprocess_std(data_path, dataset_name, preprocessed_path):
    train_csv_path = osp.join(data_path, f"{dataset_name}_checkins_train.csv")
    val_csv_path = osp.join(data_path, f"{dataset_name}_checkins_validation.csv")
    test_csv_path = osp.join(data_path, f"{dataset_name}_checkins_test.csv")

    df_train = pd.read_csv(train_csv_path)
    df_val = pd.read_csv(val_csv_path)
    df_test = pd.read_csv(test_csv_path)

    df_train["SplitTag"] = "train"
    df_val["SplitTag"] = "validation"
    df_test["SplitTag"] = "test"

    df = pd.concat([df_train, df_val, df_test])
    df = df.rename(
        columns={
            "user_id": "UserId",
            "venue_id": "PoiId",
            "venue_category_id_code": "PoiCategoryId",
            "venue_category": "PoiCategoryName",
            "venue_city_latitude": "Latitude",
            "venue_city_longitude": "Longitude",
            "trail_id": "pseudo_session_trajectory_id",
        }
    )

    df["trajectory_id"] = df["pseudo_session_trajectory_id"]
    df["UTCTimeOffset"] = df["timestamp"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    df["UTCTimeOffsetEpoch"] = df["UTCTimeOffset"].apply(lambda x: int(x.strftime("%s")))
    df["UTCTimeOffsetWeekday"] = df["UTCTimeOffset"].apply(lambda x: x.weekday())
    df["UTCTimeOffsetHour"] = df["UTCTimeOffset"].apply(lambda x: x.hour)
    df["UserRank"] = df.groupby("UserId")["UTCTimeOffset"].rank(method="first")
    df = df.sort_values(by=["UserId", "UTCTimeOffset"], ascending=True)

    # id encoding
    df["check_ins_id"] = df["UTCTimeOffset"].rank(ascending=True, method="first") - 1
    traj_id_le, padding_traj_id = id_encode(df, df, "pseudo_session_trajectory_id")

    df_train = df[df["SplitTag"] == "train"]
    poi_id_le, padding_poi_id = id_encode(df_train, df, "PoiId")
    poi_category_le, padding_poi_category = id_encode(df_train, df, "PoiCategoryId")
    user_id_le, padding_user_id = id_encode(df_train, df, "UserId")
    hour_id_le, padding_hour_id = id_encode(df_train, df, "UTCTimeOffsetHour")
    weekday_id_le, padding_weekday_id = id_encode(df_train, df, "UTCTimeOffsetWeekday")

    # save mapping logic
    with open(osp.join(preprocessed_path, "label_encoding.pkl"), "wb") as f:
        pickle.dump(
            [
                poi_id_le,
                poi_category_le,
                user_id_le,
                hour_id_le,
                weekday_id_le,
                padding_poi_id,
                padding_poi_category,
                padding_user_id,
                padding_hour_id,
                padding_weekday_id,
            ],
            f,
        )

    # ignore the first for train/validate/test and keep the last for validata/test
    df = ignore_first(df)
    df = only_keep_last(df)
    return df


def _ablation_preprocessed_dirname(cfg: Cfg) -> str:
    """Isolate preprocess caches that depend on graph_scope / unseen_prefix_poi."""
    graph_scope = getattr(cfg.dataset_args, "graph_scope", "full")
    unseen_prefix = getattr(cfg.dataset_args, "unseen_prefix_poi", "unk")
    if graph_scope == "full" and unseen_prefix == "unk":
        return "preprocessed"
    return f"preprocessed_scope-{graph_scope}_prefix-{unseen_prefix}"


def _drop_samples_with_unseen_prefix_poi(sample_df, eval_df, padding_poi_id):
    """Drop eval rows whose current-trail prefix contains padding_poi_id (train-unseen)."""
    if eval_df is None or len(eval_df) == 0:
        return eval_df
    traj_groups = {}
    for traj_id, g in sample_df.groupby("pseudo_session_trajectory_id", sort=False):
        g = g.sort_values("UTCTimeOffset")
        traj_groups[traj_id] = g["PoiId"].tolist()

    keep_idx = []
    for i, row in eval_df.iterrows():
        tid = row["pseudo_session_trajectory_id"]
        seq = traj_groups.get(tid, [])
        # Prefix = all but last (target). Drop sample if any prefix POI is UNK/padding.
        prefix = seq[:-1] if len(seq) >= 2 else []
        if any(int(p) == int(padding_poi_id) for p in prefix):
            continue
        keep_idx.append(i)
    out = eval_df.loc[keep_idx].reset_index(drop=True)
    logging.info(
        f"[Preprocess] unseen_prefix_poi=drop: kept {len(out)}/{len(eval_df)} eval samples "
        f"(padding_poi_id={padding_poi_id})."
    )
    return out


def preprocess(cfg: Cfg):
    root_path = get_root_dir()
    dataset_name = cfg.dataset_args.dataset_name
    data_path = osp.join(root_path, "data", dataset_name)
    preprocessed_path = osp.join(data_path, _ablation_preprocessed_dirname(cfg))
    sample_file = osp.join(preprocessed_path, "sample.csv")
    train_file = osp.join(preprocessed_path, "train_sample.csv")
    validate_file = osp.join(preprocessed_path, "validate_sample.csv")
    test_file = osp.join(preprocessed_path, "test_sample.csv")
    graph_csv = osp.join(preprocessed_path, "graph_sample.csv")

    keep_cols = [
        "check_ins_id",
        "UTCTimeOffset",
        "UTCTimeOffsetEpoch",
        "pseudo_session_trajectory_id",
        "query_pseudo_session_trajectory_id",
        "UserId",
        "Latitude",
        "Longitude",
        "PoiId",
        "PoiCategoryId",
        "PoiCategoryName",
        "last_checkin_epoch_time",
        "trajectory_id",
    ]

    graph_scope = getattr(cfg.dataset_args, "graph_scope", "full")
    unseen_prefix = getattr(cfg.dataset_args, "unseen_prefix_poi", "unk")

    if not osp.exists(preprocessed_path):
        os.makedirs(preprocessed_path)

    # Step 1. preprocess raw files and create sample files including
    # 1. data transformation; 2. id encoding; 3.train/validate/test splitting; 4. remove unseen user or poi
    if not osp.exists(sample_file):
        preprocessed_data = preprocess_std(data_path, dataset_name, preprocessed_path)

        preprocessed_result = remove_unseen_user_poi(preprocessed_data)

        # Ablation: drop eval samples whose trail prefix has a train-unseen POI.
        if unseen_prefix == "drop":
            with open(osp.join(preprocessed_path, "label_encoding.pkl"), "rb") as f:
                le_pack = pickle.load(f)
            padding_poi_id = le_pack[5]
            sample_df = preprocessed_result["sample"]
            preprocessed_result["validate_sample"] = _drop_samples_with_unseen_prefix_poi(
                sample_df, preprocessed_result["validate_sample"], padding_poi_id
            )
            preprocessed_result["test_sample"] = _drop_samples_with_unseen_prefix_poi(
                sample_df, preprocessed_result["test_sample"], padding_poi_id
            )

        preprocessed_result["sample"].to_csv(sample_file, index=False)
        preprocessed_result["train_sample"][keep_cols].to_csv(train_file, index=False)
        preprocessed_result["validate_sample"][keep_cols].to_csv(validate_file, index=False)
        preprocessed_result["test_sample"][keep_cols].to_csv(test_file, index=False)

        # Build the CSV used for hypergraph construction (may be scoped to train neighbors).
        sample_df = preprocessed_result["sample"]
        if graph_scope == "train":
            train_trajs = set(
                sample_df.loc[sample_df["SplitTag"] == "train", "pseudo_session_trajectory_id"].tolist()
            )
            eval_trajs = set(preprocessed_result["validate_sample"]["pseudo_session_trajectory_id"].tolist()) | set(
                preprocessed_result["test_sample"]["pseudo_session_trajectory_id"].tolist()
            )
            keep_trajs = train_trajs | eval_trajs
            graph_df = sample_df[sample_df["pseudo_session_trajectory_id"].isin(keep_trajs)].copy()
            graph_df.to_csv(graph_csv, index=False)
            with open(osp.join(preprocessed_path, "train_traj_ids.pkl"), "wb") as f:
                pickle.dump(train_trajs, f)
            logging.info(
                f"[Preprocess] graph_scope=train: graph checkins={len(graph_df)} "
                f"(train_trajs={len(train_trajs)}, eval_query_trajs={len(eval_trajs)})."
            )
        else:
            sample_df.to_csv(graph_csv, index=False)

    # Step 2. generate hypergraph related data
    if not osp.exists(osp.join(preprocessed_path, "ci2traj_pyg_data.pt")):
        if not osp.exists(graph_csv):
            # Backward compatible: older caches without graph_sample.csv
            graph_csv = sample_file
        neighbor_traj_ids = None
        train_traj_path = osp.join(preprocessed_path, "train_traj_ids.pkl")
        if graph_scope == "train" and osp.exists(train_traj_path):
            with open(train_traj_path, "rb") as f:
                neighbor_traj_ids = pickle.load(f)
        generate_hypergraph_from_file(
            graph_csv, preprocessed_path, cfg.dataset_args, neighbor_traj_ids=neighbor_traj_ids
        )

    logging.info("[Preprocess] Done preprocessing.")


if __name__ == "__main__":
    cfg = Cfg("best_conf/beijing.yml")
    preprocess(cfg)
