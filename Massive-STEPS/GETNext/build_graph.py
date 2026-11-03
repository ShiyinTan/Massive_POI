"""Build the user-agnostic global trajectory flow map from the sequence data"""

import os
import pickle
from argparse import ArgumentParser

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--csv_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--poi_id_column", type=str, default="venue_id")
    parser.add_argument("--poi_catid_column", type=str, default="venue_category_id")
    parser.add_argument("--poi_catid_code_column", type=str, default="venue_category_id_code")
    parser.add_argument("--poi_catname_column", type=str, default="venue_category")
    parser.add_argument("--user_id_column", type=str, default="user_id")
    parser.add_argument("--trajectory_id_column", type=str, default="trail_id")
    parser.add_argument("--latitude_column", type=str, default="venue_city_latitude")
    parser.add_argument("--longitude_column", type=str, default="venue_city_longitude")
    return parser.parse_args()


def build_global_POI_checkin_graph(df, args, exclude_user=None):
    G = nx.DiGraph()
    users = list(set(df[args.user_id_column].to_list()))
    if exclude_user in users:
        users.remove(exclude_user)
    loop = tqdm(users)
    for user_id in loop:
        user_df = df[df[args.user_id_column] == user_id]

        # Add node (POI)
        for i, row in user_df.iterrows():
            node = row[args.poi_id_column]
            if node not in G.nodes():
                G.add_node(
                    row[args.poi_id_column],
                    checkin_cnt=1,
                    poi_catid=row[args.poi_catid_column],
                    poi_catid_code=row[args.poi_catid_code_column],
                    poi_catname=row[args.poi_catname_column],
                    latitude=row[args.latitude_column],
                    longitude=row[args.longitude_column],
                )
            else:
                G.nodes[node]["checkin_cnt"] += 1

        # Add edges (Check-in seq)
        previous_poi_id = 0
        previous_traj_id = 0
        for i, row in user_df.iterrows():
            poi_id = row[args.poi_id_column]
            traj_id = row[args.trajectory_id_column]
            # No edge for the begin of the seq or different traj
            if (previous_poi_id == 0) or (previous_traj_id != traj_id):
                previous_poi_id = poi_id
                previous_traj_id = traj_id
                continue

            # Add edges
            if G.has_edge(previous_poi_id, poi_id):
                G.edges[previous_poi_id, poi_id]["weight"] += 1
            else:  # Add new edge
                G.add_edge(previous_poi_id, poi_id, weight=1)
            previous_traj_id = traj_id
            previous_poi_id = poi_id

    return G


def save_graph_to_csv(G, dst_dir):
    # Save graph to an adj matrix file and a nodes file
    # Adj matrix file: edge from row_idx to col_idx with weight; Rows and columns are ordered according to nodes file.
    # Nodes file: node_name/poi_id, node features (category, location); Same node order with adj matrix.

    # Save adj matrix
    nodelist = G.nodes()
    A = nx.adjacency_matrix(G, nodelist=nodelist)
    # np.save(os.path.join(dst_dir, 'adj_mtx.npy'), A.todense())
    np.savetxt(os.path.join(dst_dir, "graph_A.csv"), A.todense(), delimiter=",")

    # Save nodes list
    nodes_data = list(G.nodes.data())  # [(node_name, {attr1, attr2}),...]
    with open(os.path.join(dst_dir, "graph_X.csv"), "w") as f:
        print(
            "node_name/poi_id,checkin_cnt,poi_catid,poi_catid_code,poi_catname,latitude,longitude",
            file=f,
        )
        for each in nodes_data:
            node_name = each[0]
            checkin_cnt = each[1]["checkin_cnt"]
            poi_catid = each[1]["poi_catid"]
            poi_catid_code = each[1]["poi_catid_code"]
            poi_catname = each[1]["poi_catname"]
            latitude = each[1]["latitude"]
            longitude = each[1]["longitude"]
            print(
                f"{node_name},{checkin_cnt}," f"{poi_catid},{poi_catid_code},{poi_catname}," f"{latitude},{longitude}",
                file=f,
            )


def save_graph_to_pickle(G, dst_dir):
    pickle.dump(G, open(os.path.join(dst_dir, "graph.pkl"), "wb"))


def save_graph_edgelist(G, dst_dir):
    nodelist = G.nodes()
    node_id2idx = {k: v for v, k in enumerate(nodelist)}

    with open(os.path.join(dst_dir, "graph_node_id2idx.txt"), "w") as f:
        for i, node in enumerate(nodelist):
            print(f"{node}, {i}", file=f)

    with open(os.path.join(dst_dir, "graph_edge.edgelist"), "w") as f:
        for edge in nx.generate_edgelist(G, data=["weight"]):
            src_node, dst_node, weight = edge.split(" ")
            print(f"{node_id2idx[int(src_node)]} {node_id2idx[int(dst_node)]} {weight}", file=f)


def load_graph_adj_mtx(path):
    """A.shape: (num_node, num_node), edge from row_index to col_index with weight"""
    A = np.loadtxt(path, delimiter=",")
    return A


def load_graph_node_features(
    path,
    feature1="checkin_cnt",
    feature2="poi_catid_code",
    feature3="latitude",
    feature4="longitude",
):
    """X.shape: (num_node, 4), four features: checkin cnt, poi cat, latitude, longitude"""
    df = pd.read_csv(path)
    rlt_df = df[[feature1, feature2, feature3, feature4]]
    X = rlt_df.to_numpy()

    return X


def print_graph_statisics(G):
    print(f"Num of nodes: {G.number_of_nodes()}")
    print(f"Num of edges: {G.number_of_edges()}")

    # Node degrees (mean and percentiles)
    node_degrees = [each[1] for each in G.degree]
    print(f"Node degree (mean): {np.mean(node_degrees):.2f}")
    for i in range(0, 101, 20):
        print(f"Node degree ({i} percentile): {np.percentile(node_degrees, i)}")

    # Edge weights (mean and percentiles)
    edge_weights = []
    for n, nbrs in G.adj.items():
        for nbr, attr in nbrs.items():
            weight = attr["weight"]
            edge_weights.append(weight)
    print(f"Edge frequency (mean): {np.mean(edge_weights):.2f}")
    for i in range(0, 101, 20):
        print(f"Edge frequency ({i} percentile): {np.percentile(edge_weights, i)}")


if __name__ == "__main__":
    args = parse_args()

    # Build POI checkin trajectory graph
    train_df = pd.read_csv(args.csv_path)
    G = build_global_POI_checkin_graph(train_df, args)

    # Save graph to disk
    save_graph_to_pickle(G, dst_dir=args.output_dir)
    save_graph_to_csv(G, dst_dir=args.output_dir)
    save_graph_edgelist(G, dst_dir=args.output_dir)
