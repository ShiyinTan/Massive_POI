import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import os
from torch.amp import autocast, GradScaler


def train_stop(model, train_loader, valid_loaders, log_dir, lr, epoch, valid_step_interval, device, citys, patience=10):
    log_file_train = os.path.join(log_dir, f"log_train.txt")
    with open(log_file_train, "w") as f:  # open for writing to clear the file
        pass

    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=lr)

    p1 = int(0.4 * epoch)
    p2 = int(0.6 * epoch)
    p3 = int(0.9 * epoch)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[p1, p2, p3], gamma=0.4)
    best_valid_loss_dict = {}
    patience_counter_dict = {}

    scaler = GradScaler()

    for city in citys:
        best_valid_loss_dict[f"{city}"] = 1e10
        patience_counter_dict[f"{city}"] = 0
        log_file_val = os.path.join(log_dir, f"log_val_{city}.txt")
        with open(log_file_val, "w") as f:  # open for writing to clear the file
            pass

    for epoch_no in range(epoch):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()
                x_train = train_batch[0]
                y_train = train_batch[1]
                ts = train_batch[2]
                train_city = train_batch[3][0]
                vocab = np.load(f"./location_feature/vocab_{train_city}.npy")
                vocab = np.pad(vocab, ((2, 0), (0, 0)), mode="constant", constant_values=0)
                vocab = torch.from_numpy(vocab)
                vocab = vocab.to(torch.float32)

                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(x_train, ts, y_train, vocab, device)
                    loss = output["loss"]

                # Scale the loss and call backward
                scaler.scale(loss).backward()

                avg_loss += loss.item()
                loss_avg = avg_loss / batch_no

                scaler.step(optimizer)
                scaler.update()

                with open(log_file_train, "a") as f:
                    f.write(f"{epoch_no}\t{batch_no}\t train \t{loss_avg:.6f}\n")
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": loss_avg,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )

                if valid_loaders is not None and (batch_no + 1) % valid_step_interval == 0:
                    model.eval()
                    with torch.no_grad():
                        for valid_loader in valid_loaders:
                            avg_loss_valid = 0
                            with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                                for batch_no_val, valid_batch in enumerate(it, start=1):
                                    x_val = valid_batch[0]
                                    y_val = valid_batch[1]
                                    ts = valid_batch[2]
                                    val_city = valid_batch[3][0]
                                    vocab = np.load(f"./location_feature/vocab_{val_city}.npy")
                                    vocab = np.pad(vocab, ((2, 0), (0, 0)), mode="constant", constant_values=0)
                                    vocab = torch.from_numpy(vocab)
                                    vocab = vocab.to(torch.float32)

                                    with autocast(device_type="cuda", dtype=torch.bfloat16):
                                        output = model(x_val, ts, y_val, vocab, device)
                                        loss = output["loss"]

                                    avg_loss_valid += loss.item()
                                    loss_avg_valid = avg_loss_valid / batch_no_val
                                    it.set_postfix(
                                        ordered_dict={
                                            "valid_avg_loss": loss_avg_valid,
                                            "epoch": epoch_no,
                                        },
                                        refresh=False,
                                    )
                                log_file_val = os.path.join(log_dir, f"log_val_{val_city}.txt")
                                with open(log_file_val, "a") as f:
                                    f.write(f"{epoch_no}\t{batch_no}\t val \t{loss_avg_valid:.6f}\n")

                                if best_valid_loss_dict[f"{val_city}"] > avg_loss_valid:
                                    output_path = log_dir + f"/model_{val_city}.pth"
                                    torch.save(model.state_dict(), output_path)
                                    best_valid_loss_dict[f"{val_city}"] = avg_loss_valid
                                    patience_counter_dict[f"{val_city}"] = 0  # Reset patience counter
                                    print(
                                        "\n best loss is updated to ",
                                        avg_loss_valid / batch_no_val,
                                        "at",
                                        epoch_no,
                                        val_city,
                                    )
                                else:
                                    patience_counter_dict[f"{val_city}"] += 1
                                    # if patience_counter_dict[f'{val_city}'] >= patience:
                                    if all(value > patience for value in patience_counter_dict.values()):
                                        print(f"\n Early stopping triggered for {val_city} at epoch {epoch_no}")
                                        return  # Stop training

                lr_scheduler.step()


def evaluate(model, test_loader, log_dir, B, city, device):
    log_file_test = os.path.join(log_dir, f"log_{'_'.join(city)}_test.txt")
    with open(log_file_test, "w") as f:  # open for writing to clear the file
        pass
    model.load_state_dict(torch.load(log_dir + f"/model_{city[0]}.pth"))
    model.eval()
    acc1 = 0
    acc3 = 0
    acc5 = 0
    ndcg5 = 0
    size = 0
    val_loss_accum = 0.0
    batch = 0
    with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
        for batch_no, test_batch in enumerate(it, start=1):
            batch = batch_no + 1
            x_test = test_batch[0]
            y_test = test_batch[1]
            ts = test_batch[2]
            test_city = test_batch[3][0]
            vocab = np.load(f"./location_feature/vocab_{test_city}.npy")
            vocab = np.pad(vocab, ((2, 0), (0, 0)), mode="constant", constant_values=0)
            vocab = torch.from_numpy(vocab)
            vocab = vocab.to(torch.float32)

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(x_test, ts, y_test, vocab, device)
                loss = output["loss"]

            val_loss_accum += loss.detach()
            pred = output["logits"]  # [B T vocab_size]
            pred[:, :, 0] = float("-inf")
            y_test = y_test.to(device)
            batch_size = pred.size(0)
            for b in range(batch_size):
                _, pred_indices = torch.topk(pred[b], 100)
                valid_mask = y_test[b] > 0
                valid_y_val = y_test[b][valid_mask]
                valid_pred_indices = pred_indices[valid_mask]

                valid_y_val_expanded = valid_y_val.unsqueeze(1)
                l = valid_y_val_expanded.size(0)
                size += l

                a1 = torch.sum(valid_pred_indices[:, 0:1] == valid_y_val_expanded).item()
                a3 = torch.sum(valid_pred_indices[:, 0:3] == valid_y_val_expanded).item()

                a5 = torch.sum(valid_pred_indices[:, 0:5] == valid_y_val_expanded).item()
                acc1 += a1
                acc3 += a3
                acc5 += a5

                # NDCG@5
                for i in range(l):
                    ground_truth = valid_y_val[i].item()
                    predictions_top5 = valid_pred_indices[i, 0:5].tolist()
                    if ground_truth in predictions_top5:
                        ndcg5 += 1 / np.log2(predictions_top5.index(ground_truth) + 1 + 1)

    val_loss_accum = val_loss_accum / batch
    acc1 = acc1 / size
    acc3 = acc3 / size
    acc5 = acc5 / size
    ndcg5 = ndcg5 / size

    with open(log_file_test, "a") as f:
        f.write(f"{val_loss_accum}\t{acc1:.6f}\t{acc5:.6f}\t{ndcg5:.6f}\t{size}\n")
