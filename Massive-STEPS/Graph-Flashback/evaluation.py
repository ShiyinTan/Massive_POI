import logging
import torch
import numpy as np


class Evaluation:
    """
    Handles evaluation on a given POI dataset and loader.

    The two metrics are MAP and recall@n. Our model predicts sequence of
    next locations determined by the sequence_length at one pass. During evaluation we
    treat each entry of the sequence as single prediction. One such prediction
    is the ranked list of all available locations and we can compute the two metrics.

    As a single prediction is of the size of all available locations,
    evaluation takes its time to compute. The code here is optimized.

    Using the --report_user argument one can access the statistics per user.
    """

    def __init__(self, dataset, dataloader, user_count, h0_strategy, trainer, setting):
        self.dataset = dataset
        self.dataloader = dataloader
        self.user_count = user_count
        self.h0_strategy = h0_strategy
        self.trainer = trainer
        self.setting = setting

    def evaluate(self):
        self.dataset.reset()
        h = self.h0_strategy.on_init(self.setting.batch_size, self.setting.device)

        evaluated_traj_indices = set()

        with torch.no_grad():
            iter_cnt = 0
            recall1 = 0
            recall5 = 0
            recall10 = 0
            average_precision = 0.0
            ndcg5 = 0.0

            u_iter_cnt = np.zeros(self.user_count)
            u_recall1 = np.zeros(self.user_count)
            u_recall5 = np.zeros(self.user_count)
            u_recall10 = np.zeros(self.user_count)
            u_average_precision = np.zeros(self.user_count)
            u_ndcg5 = np.zeros(self.user_count)
            reset_count = torch.zeros(self.user_count)

            for i, (x, t, t_slot, s, y, lengths, active_users, reset_h, traj_ids) in enumerate(self.dataloader):
                x = x.squeeze(0).to(self.setting.device)
                t = t.squeeze(0).to(self.setting.device)
                t_slot = t_slot.squeeze(0).to(self.setting.device)
                s = s.squeeze(0).to(self.setting.device)
                y = y.squeeze(0)
                lengths = lengths.squeeze(0).to(self.setting.device)
                active_users = active_users.squeeze(0).to(self.setting.device)
                traj_ids = traj_ids.squeeze(0)

                for j, reset in enumerate(reset_h):
                    if reset:
                        if self.setting.is_lstm:
                            hc = self.h0_strategy.on_reset_test(active_users[j], self.setting.device)
                            h[0][0, j] = hc[0]
                            h[1][0, j] = hc[1]
                        else:
                            h[0, j] = self.h0_strategy.on_reset_test(active_users[j], self.setting.device)
                        reset_count[active_users[j]] += 1

                # evaluate:
                out, h = self.trainer.evaluate(x, t, t_slot, s, lengths, h, active_users)

                for j in range(self.setting.batch_size):
                    t_idx = traj_ids[j].item()

                    if t_idx in evaluated_traj_indices:
                        continue
                    evaluated_traj_indices.add(t_idx)

                    user_idx = active_users[j].item()

                    # 3. Find the last valid index for this trajectory
                    # If length is L, last input is at L-1, last prediction is at index L-1
                    last_idx = lengths[j].item() - 1

                    # Logits for the last predicted POI
                    o_n = out[j, last_idx].cpu().detach().numpy()

                    # Target POI
                    target_poi = y[last_idx, j].item()

                    # If the target is the padding value (-100), skip (shouldn't happen with trail_id logic)
                    if target_poi == -100:
                        continue

                    # Metrics calculation
                    ind = np.argpartition(o_n, -10)[-10:]
                    r = ind[np.argsort(-o_n[ind])]  # Top 10 sorted

                    # NDCG@5
                    curr_ndcg5 = 0.0
                    for rank, pred in enumerate(r[:5]):
                        if pred == target_poi:
                            curr_ndcg5 = 1.0 / np.log2(rank + 2)
                            break

                    # MAP (Precision at the rank of the true POI)
                    t_val = o_n[target_poi]
                    upper = np.where(o_n > t_val)[0]
                    precision = 1.0 / (1 + len(upper))

                    # Accumulate
                    u_iter_cnt[user_idx] += 1
                    u_recall1[user_idx] += target_poi in r[:1]
                    u_recall5[user_idx] += target_poi in r[:5]
                    u_recall10[user_idx] += target_poi in r[:10]
                    u_ndcg5[user_idx] += curr_ndcg5
                    u_average_precision[user_idx] += precision

            formatter = "{0:.8f}"
            for j in range(self.user_count):
                iter_cnt += u_iter_cnt[j]
                recall1 += u_recall1[j]
                recall5 += u_recall5[j]
                recall10 += u_recall10[j]
                ndcg5 += u_ndcg5[j]
                average_precision += u_average_precision[j]

            logging.info(f"recall@1: {formatter.format(recall1 / iter_cnt)}")
            logging.info(f"recall@5: {formatter.format(recall5 / iter_cnt)}")
            logging.info(f"recall@10: {formatter.format(recall10 / iter_cnt)}")
            logging.info(f"NDCG@5: {formatter.format(ndcg5 / iter_cnt)}")
            logging.info(f"MAP: {formatter.format(average_precision / iter_cnt)}")
            logging.info(f"predictions: {iter_cnt}")

            return {
                "recall1": recall1 / iter_cnt,
                "recall5": recall5 / iter_cnt,
                "recall10": recall10 / iter_cnt,
                "ndcg5": ndcg5 / iter_cnt,
                "map": average_precision / iter_cnt,
            }
