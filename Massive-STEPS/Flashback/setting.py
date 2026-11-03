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
        self.hidden_dim = args.hidden_dim
        self.weight_decay = args.weight_decay
        self.learning_rate = args.lr
        self.epochs = args.epochs
        self.rnn_factory = RnnFactory(args.rnn)
        self.is_lstm = self.rnn_factory.is_lstm()
        self.lambda_t = args.lambda_t
        self.lambda_s = args.lambda_s

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

        ### CUDA Setup ###
        self.device = torch.device("cpu") if args.gpu == -1 else torch.device("cuda", args.gpu)

    def parse_arguments(self, parser):
        # training
        parser.add_argument("--gpu", default=0, type=int, help="the gpu to use")
        parser.add_argument("--hidden-dim", default=10, type=int, help="hidden dimensions to use")
        parser.add_argument("--weight_decay", default=0.0, type=float, help="weight decay regularization")
        parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
        parser.add_argument("--epochs", default=100, type=int, help="amount of epochs")
        parser.add_argument("--rnn", default="rnn", type=str, help="the GRU implementation to use: [rnn|gru|lstm]")
        parser.add_argument("--sequence_length", default=20, type=int, help="length of input sequences")

        # data management
        parser.add_argument("--city", required=True, type=str)
        parser.add_argument("--dataset_train_file", required=True, type=str)
        parser.add_argument("--dataset_val_file", required=True, type=str)
        parser.add_argument("--dataset_test_file", required=True, type=str)

        parser.add_argument(
            "--batch-size", default=100, type=int, help="amount of users to process in one pass (batching)"
        )
        parser.add_argument("--lambda_t", default=0.1, type=float, help="decay factor for temporal data")
        parser.add_argument("--lambda_s", default=100, type=float, help="decay factor for spatial data")

    def __str__(self):
        settings = vars(self)
        result = "Settings:\n"
        for key in sorted(settings.keys()):
            result += f"{key}: {settings[key]}\n"
        return result
