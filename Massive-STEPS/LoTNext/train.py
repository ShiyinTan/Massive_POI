from pathlib import Path
import logging
import torch
from torch.utils.data import DataLoader
import numpy as np
import time, os
import pickle
import copy
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
        self.best_epoch = None

    def __call__(self, current_score, current_epoch, model):
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = current_epoch
            self.save_checkpoint(model)
        elif current_score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_score
            self.best_epoch = current_epoch
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.best_model_path)


def compute_adjustment(label_freq, setting):
    """compute the base probabilities"""

    label_freq_array = np.array(list(label_freq.values()))
    max_freq = label_freq_array.max()
    tau = 1.2
    adjustments = tau * (1 - (np.log(label_freq_array + 1e-4) / np.log(max_freq + 1e-4)))
    adjustments = torch.from_numpy(adjustments)
    adjustments = adjustments.to(setting.device)
    return adjustments


seed = 3407  # 或任何你选择的数字
random.seed(seed)  # 为Python内建的随机数生成器设置种子
np.random.seed(seed)  # 为NumPy的随机数生成器设置种子
torch.manual_seed(seed)  # 为Torch的随机数生成器设置种子


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
        interact_graph = pickle.load(f)
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
        setting=setting,
    )

    evaluation_val = Evaluation(val_dataset, val_dataloader, user_count, h0_strategy, trainer, setting)
    evaluation_test = Evaluation(test_dataset, test_dataloader, user_count, h0_strategy, trainer, setting)
    logging.info("{} {}".format(trainer, setting.rnn_factory))

    logits = compute_adjustment(train_dataset.freq, setting)

    #  training loop
    optimizer = torch.optim.AdamW(trainer.parameters(), lr=setting.learning_rate, weight_decay=setting.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=0.2)

    param_count = trainer.count_parameters()
    logging.info(f"In total: {param_count} trainable parameters")

    pbar = tqdm(range(len(train_dataloader) * setting.epochs))

    for e in range(setting.epochs):
        pbar.set_description(f"Epoch {e+1}/{setting.epochs}")
        h = h0_strategy.on_init(setting.batch_size, setting.device)

        losses = []

        for i, (x, t, t_slot, s, y, y_t_slot, lengths, active_users, reset_h, _, f, y_f) in enumerate(train_dataloader):
            x = x.squeeze(0).to(setting.device)
            t = t.squeeze(0).to(setting.device)
            t_slot = t_slot.squeeze(0).to(setting.device)
            s = s.squeeze(0).to(setting.device)
            y = y.squeeze(0).to(setting.device)
            y_t_slot = y_t_slot.squeeze(0).to(setting.device)
            lengths = lengths.squeeze(0).to(setting.device)
            active_users = active_users.to(setting.device)
            f = f.squeeze(0).to(setting.device)
            y_f = y_f.squeeze(0).to(setting.device)

            optimizer.zero_grad()
            loss = trainer.loss(x, t, t_slot, s, y, y_t_slot, lengths, reset_h, active_users, f, y_f, logits, e)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()

            pbar.set_postfix(loss=loss.item())
            pbar.update(1)

        # schedule learning rate:
        scheduler.step()

        epoch_loss = np.mean(losses)
        logging.info(f"Epoch: {e + 1}/{setting.epochs}")
        logging.info(f"Used learning rate: {scheduler.get_last_lr()[0]}")
        logging.info(f"Avg Loss: {epoch_loss}")

        logging.info(f"~~~ Validation Set Evaluation (Epoch: {e + 1}) ~~~")
        val_metrics = evaluation_val.evaluate(e)

        # We use Recall@1 as the lead metric for Early Stopping
        early_stopping(current_score=val_metrics["recall1"], current_epoch=e, model=trainer.model)

        if early_stopping.early_stop:
            logging.info("Early stopping triggered. Training finished.")
            break

    logging.info("~~~ Test Set Evaluation ~~~")
    trainer.model.load_state_dict(torch.load(early_stopping.best_model_path))
    evaluation_test.evaluate(early_stopping.best_epoch)


if __name__ == "__main__":
    main()
