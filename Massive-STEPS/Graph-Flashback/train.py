from pathlib import Path
import logging
import torch
from torch.utils.data import DataLoader
import numpy as np
import time, os
import pickle
from setting import Setting
from trainer import FlashbackTrainer
from dataloader import PoiDataloader
from dataset import Split
from utils import *
from network import create_h0_strategy
from evaluation import Evaluation
from tqdm import tqdm
from scipy.sparse import coo_matrix


class EarlyStopping:
    def __init__(self, exp_path, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.exp_path = exp_path
        self.best_model_path = exp_path / "best_model.pth"

    def __call__(self, current_score, model):
        if self.best_score is None:
            self.best_score = current_score
            self.save_checkpoint(model)
        elif current_score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.best_model_path)


def main():
    # parse settings
    setting = Setting()
    setting.parse()

    exp_path = Path("logs") / setting.city
    exp_path.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO, filename=exp_path / "training_log.log", filemode="w", format="%(asctime)s - %(message)s"
    )
    logging.info(setting)

    # load dataset
    poi_loader = PoiDataloader(setting.max_users, setting.min_checkins)  # 0， 5*20+1
    poi_loader.read(setting.dataset_train_file, setting.dataset_val_file, setting.dataset_test_file)

    logging.info("Active POI number:{}".format(poi_loader.locations()))
    logging.info("Active User number:{}".format(poi_loader.user_count()))
    logging.info("Total Checkins number:{}".format(poi_loader.checkins_count()))

    train_dataset = poi_loader.create_dataset(setting.sequence_length, setting.batch_size, Split.TRAIN)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=32, pin_memory=True)

    val_dataset = poi_loader.create_dataset(setting.sequence_length, setting.batch_size, Split.VAL)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=32, pin_memory=True)

    test_dataset = poi_loader.create_dataset(setting.sequence_length, setting.batch_size, Split.TEST)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=32, pin_memory=True)

    assert setting.batch_size < poi_loader.user_count(), "batch size must be lower than the amount of available users"

    # create flashback trainer
    with open(setting.trans_loc_file, "rb") as f:  # transition POI graph
        transition_graph = pickle.load(f)

    transition_graph = coo_matrix(transition_graph)

    if setting.use_spatial_graph:
        with open(setting.trans_loc_spatial_file, "rb") as f:  # spatial POI graph
            spatial_graph = pickle.load(f)
        spatial_graph = coo_matrix(spatial_graph)
    else:
        spatial_graph = None

    if setting.use_graph_user:
        with open(setting.trans_user_file, "rb") as f:
            friend_graph = pickle.load(f)
        friend_graph = coo_matrix(friend_graph)
    else:
        friend_graph = None

    with open(setting.trans_interact_file, "rb") as f:  # User-POI interaction graph
        interact_graph = pickle.load(f)  # 在cpu上
    interact_graph = csr_matrix(interact_graph)

    logging.info("Successfully load graph")

    early_stopping = EarlyStopping(exp_path=exp_path, patience=setting.patience)
    trainer = FlashbackTrainer(
        setting.lambda_t,
        setting.lambda_s,
        setting.lambda_loc,
        setting.lambda_user,
        setting.use_weight,
        transition_graph,
        spatial_graph,
        friend_graph,
        setting.use_graph_user,
        setting.use_spatial_graph,
        interact_graph,
    )
    h0_strategy = create_h0_strategy(setting.hidden_dim, setting.is_lstm)
    loc_count, user_count = poi_loader.locations(), poi_loader.user_count()

    trainer.prepare(
        loc_count=loc_count,
        user_count=user_count,
        hidden_size=setting.hidden_dim,
        gru_factory=setting.rnn_factory,
        padding_idx=loc_count,
        device=setting.device,
    )
    evaluation_val = Evaluation(val_dataset, val_dataloader, user_count, h0_strategy, trainer, setting)
    evaluation_test = Evaluation(test_dataset, test_dataloader, user_count, h0_strategy, trainer, setting)
    logging.info("{} {}".format(trainer, setting.rnn_factory))

    #  training loop
    optimizer = torch.optim.Adam(trainer.parameters(), lr=setting.learning_rate, weight_decay=setting.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=0.2)

    param_count = trainer.count_parameters()
    logging.info(f"In total: {param_count} trainable parameters")

    pbar = tqdm(range(len(train_dataloader) * setting.epochs))
    for e in range(setting.epochs):
        pbar.set_description(f"Epoch {e+1}/{setting.epochs}")
        h = h0_strategy.on_init(setting.batch_size, setting.device)

        losses = []

        for i, (x, t, t_slot, s, y, lengths, active_users, reset_h, _) in enumerate(train_dataloader):
            x = x.squeeze(0).to(setting.device)
            t = t.squeeze(0).to(setting.device)
            t_slot = t_slot.squeeze(0).to(setting.device)
            s = s.squeeze(0).to(setting.device)
            y = y.squeeze(0).to(setting.device)
            lengths = lengths.squeeze(0).to(setting.device)
            active_users = active_users.to(setting.device)

            # reset hidden states for newly added users
            for j, reset in enumerate(reset_h):
                if reset:
                    if setting.is_lstm:
                        hc = h0_strategy.on_reset(active_users[0][j])
                        h[0][0, j] = hc[0]
                        h[1][0, j] = hc[1]
                    else:
                        h[0, j] = h0_strategy.on_reset(active_users[0][j])

            optimizer.zero_grad()
            loss = trainer.loss(x, t, t_slot, s, y, lengths, h, active_users)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()

            pbar.set_postfix(loss=loss.item())
            pbar.update(1)

        # schedule learning rate:
        scheduler.step()

        # statistics:
        epoch_loss = np.mean(losses)
        logging.info(f"Epoch: {e + 1}/{setting.epochs}")
        logging.info(f"Used learning rate: {scheduler.get_last_lr()[0]}")
        logging.info(f"Avg Loss: {epoch_loss}")

        logging.info(f"~~~ Validation Set Evaluation (Epoch: {e + 1}) ~~~")
        val_metrics = evaluation_val.evaluate()

        # We use Recall@10 as the lead metric for Early Stopping
        early_stopping(current_score=val_metrics["recall1"], model=trainer.model)

        if early_stopping.early_stop:
            logging.info("Early stopping triggered. Training finished.")
            break

    logging.info("~~~ Test Set Evaluation ~~~")
    trainer.model.load_state_dict(torch.load(early_stopping.best_model_path))
    evaluation_test.evaluate()


if __name__ == "__main__":
    main()
