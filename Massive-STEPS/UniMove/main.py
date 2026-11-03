import os
import numpy as np
import torch
from torch.nn import functional as F

torch.cuda.empty_cache()  # Clear the cache
from dataloader import TrajDataset, get_dataloader
import random
import argparse
from torch.utils.data import DataLoader
from utils import evaluate, train_stop
from model import Traj_Config, Traj_Model


def set_random_seed(seed: int):
    """
    固定随机种子以确保结果的可重复性

    参数:
    seed (int): 要设置的随机种子
    """
    random.seed(seed)  # Python 随机数生成器
    np.random.seed(seed)  # NumPy 随机数生成器
    torch.manual_seed(seed)  # PyTorch 随机数生成器（CPU）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # PyTorch 随机数生成器（当前 GPU）
        torch.cuda.manual_seed_all(seed)  # PyTorch 随机数生成器（所有 GPU）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser(description="traj_MOE_MODEL")

parser.add_argument("--device", default="cuda:0", help="Device for Attack")
# 随机种子
parser.add_argument("--seed", type=int, default=520)
# Model 参数
parser.add_argument("--n_embd", type=int, default=512)
parser.add_argument("--n_head", type=int, default=4)
parser.add_argument("--n_layer", type=int, default=4)
parser.add_argument("--num_experts", type=int, default=4)
parser.add_argument("--top_k", type=int, default=2)

# data load
parser.add_argument("--B", type=int, default=16, help="batch size")
parser.add_argument("--T", type=int, default=144, help="max length 48*3days")
parser.add_argument("--city", nargs="+", default=["nanchang", "shanghai", "lasa"])
parser.add_argument("--target_city", nargs="+", default=["nanchang"])
parser.add_argument("--train_root", type=str, default="traj_dataset/massive_steps/train")
parser.add_argument("--val_root", type=str, default="traj_dataset/massive_steps/val")
parser.add_argument("--test_root", type=str, default="traj_dataset/massive_steps/test")
parser.add_argument("--few_shot", type=float, default=1.0)
# train
parser.add_argument("--epoch", type=int, default=50)
parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")

args = parser.parse_args()
print(args)
set_random_seed(args.seed)


log_dir = f"models/{'_'.join(args.city)}/{args.n_layer}_{args.n_embd}/log_{args.few_shot}"
os.makedirs(log_dir, exist_ok=True)


model = Traj_Model(
    Traj_Config(
        n_embd=args.n_embd, n_head=args.n_head, n_layer=args.n_layer, num_experts=args.num_experts, top_k=args.top_k
    )
).to(args.device)


train_dataset = TrajDataset(args.train_root, args.city, args.B, args.T, args.few_shot)
train_loader = DataLoader(train_dataset, batch_size=args.B, shuffle=False)


valid_step_interval = len(train_dataset) // args.B

val_loader = []
for city in args.target_city:
    city = [f"{city}"]
    val_loader_city = get_dataloader(args.val_root, city, args.B, args.T, few_shot=False)
    val_loader.append(val_loader_city)

train_stop(
    model,
    train_loader=train_loader,
    valid_loaders=val_loader,
    log_dir=log_dir,
    lr=args.lr,
    epoch=args.epoch,
    valid_step_interval=valid_step_interval,
    device=args.device,
    citys=args.target_city,
    patience=3,
)

for city in args.target_city:
    city = [f"{city}"]
    test_loader = get_dataloader(args.test_root, city, args.B, args.T, few_shot=False)
    evaluate(model, test_loader=test_loader, log_dir=log_dir, B=args.B, city=city, device=args.device)
