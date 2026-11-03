import torch
import argparse
import sys

from network import RnnFactory


class Setting:
    """Defines all settings in a single place using a command line interface."""

    def parse(self):
        parser = argparse.ArgumentParser()
        self.parse_arguments(parser)
        args = parser.parse_args()

        ###### settings ######
        # training
        self.gpu = args.gpu
        self.hidden_dim = args.hidden_dim  # 10
        self.weight_decay = args.weight_decay  # 0.0
        self.learning_rate = args.lr  # 0.01
        self.epochs = args.epochs  # 100
        self.rnn_factory = RnnFactory(args.rnn)  # RNN:0, GRU:1, LSTM:2
        self.is_lstm = self.rnn_factory.is_lstm()  # True or False
        self.lambda_t = args.lambda_t  # 0.01
        self.lambda_s = args.lambda_s  # 100 or 1000

        # data management
        self.city = args.city
        self.dataset_train_file = args.dataset_train_file
        self.dataset_val_file = args.dataset_val_file
        self.dataset_test_file = args.dataset_test_file
        self.max_users = 0  # 0 = use all available users
        self.sequence_length = args.sequence_length
        self.batch_size = args.batch_size
        self.min_checkins = args.sequence_length + 1
        self.patience = 5

        self.trans_loc_file = args.trans_loc_file  # 时间POI graph
        self.trans_loc_spatial_file = args.trans_loc_spatial_file  # 空间POI graph
        self.trans_user_file = args.trans_user_file
        self.trans_interact_file = args.trans_interact_file

        self.lambda_user = args.lambda_user
        self.lambda_loc = args.lambda_loc

        self.use_weight = args.use_weight
        self.use_graph_user = args.use_graph_user
        self.use_spatial_graph = args.use_spatial_graph

        ### CUDA Setup ###
        self.device = torch.device("cpu") if args.gpu == -1 else torch.device("cuda", args.gpu)

    def parse_arguments(self, parser):
        # training
        parser.add_argument("--gpu", default=0, type=int, help="the gpu to use")  # -1
        parser.add_argument("--hidden-dim", default=10, type=int, help="hidden dimensions to use")  # 10
        parser.add_argument("--weight_decay", default=0, type=float, help="weight decay regularization")
        parser.add_argument("--lr", default=0.01, type=float, help="learning rate")  # 0.01
        parser.add_argument("--epochs", default=100, type=int, help="amount of epochs")  # 100
        parser.add_argument("--rnn", default="rnn", type=str, help="the GRU implementation to use: [rnn|gru|lstm]")
        parser.add_argument("--sequence_length", default=20, type=int, help="length of input sequences")

        # data management
        parser.add_argument("--city", required=True, type=str)
        parser.add_argument("--dataset_train_file", required=True, type=str)
        parser.add_argument("--dataset_val_file", required=True, type=str)
        parser.add_argument("--dataset_test_file", required=True, type=str)

        parser.add_argument(
            "--trans_loc_file",
            default="./KGE/POI_graph/gowalla_scheme2_transe_loc_temporal_100.pkl",
            type=str,
            help="使用transe方法构造的时间POI转换图",
        )
        parser.add_argument("--trans_user_file", default="", type=str, help="使用transe方法构造的user转换图")
        parser.add_argument("--trans_loc_spatial_file", default="", type=str, help="使用transe方法构造的空间POI转换图")
        parser.add_argument(
            "--trans_interact_file",
            default="./KGE/POI_graph/gowalla_scheme2_transe_user-loc_100.pkl",
            type=str,
            help="使用transe方法构造的用户-POI交互图",
        )
        parser.add_argument("--use_weight", default=False, type=bool, help="应用于GCN的AXW中是否使用W")
        parser.add_argument("--use_graph_user", default=False, type=bool, help="是否使用user graph")
        parser.add_argument("--use_spatial_graph", default=False, type=bool, help="是否使用空间POI graph")

        parser.add_argument(
            "--batch-size", default=100, type=int, help="amount of users to process in one pass (batching)"
        )
        parser.add_argument("--lambda_t", default=0.1, type=float, help="decay factor for temporal data")
        parser.add_argument("--lambda_s", default=100, type=float, help="decay factor for spatial data")
        parser.add_argument("--lambda_loc", default=1.0, type=float, help="weight factor for transition graph")
        parser.add_argument("--lambda_user", default=1.0, type=float, help="weight factor for user graph")

    def __str__(self):
        settings = vars(self)
        result = "Settings:\n"
        for key in sorted(settings.keys()):
            result += f"{key}: {settings[key]}\n"
        return result
