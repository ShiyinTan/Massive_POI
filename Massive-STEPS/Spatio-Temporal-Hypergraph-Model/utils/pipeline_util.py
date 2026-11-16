import csv
import json
import torch
import logging
import numpy as np
from tqdm import tqdm
import os
import os.path as osp
from metric import (
    recall,
    ndcg,
    map_k,
    mrr
)
from utils.log_format import (
    DEFAULT_KS,
    format_metrics_block,
    format_metrics_line,
    metrics_dict,
    metrics_to_serializable,
)


def save_model(model, optimizer, save_variable_list, run_args, argparse_dict):
    """Save best-model checkpoint to disk (skipped unless run_args.save_checkpoint).

    Default contents are weights-only (no optimizer) to avoid huge files.
    Set run_args.checkpoint_contents to ``full`` to also store optimizer state.
    """
    with open(osp.join(run_args.log_path, 'config.json'), 'w') as fjson:
        for key, value in argparse_dict.items():
            if isinstance(value, torch.Tensor):
                argparse_dict[key] = value.numpy().tolist()
        json.dump(argparse_dict, fjson, indent=2, sort_keys=True)

    if not getattr(run_args, 'save_checkpoint', False):
        return

    contents = getattr(run_args, 'checkpoint_contents', 'weights')
    payload = {
        **save_variable_list,
        'model_state_dict': model.state_dict(),
    }
    if contents == 'full' and optimizer is not None:
        payload['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(payload, osp.join(run_args.save_path, 'checkpoint.pt'))
    logging.info(
        f"[Training] Wrote {'full' if 'optimizer_state_dict' in payload else 'weights-only'} "
        f"checkpoint → {osp.join(run_args.save_path, 'checkpoint.pt')}"
    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_eval_results(
    results_path,
    split,
    recall_res,
    ndcg_res,
    map_res,
    mrr_res,
    *,
    loss=None,
    epoch=None,
    step=None,
    extra=None,
    run_tag=None,
    ks=DEFAULT_KS,
):
    """Write structured eval results: JSON, CSV row history, and markdown summary.

    Filenames are human-readable, e.g.:
      beijing_sthgcn_test_metrics.json
      beijing_sthgcn_valid_history.csv
      beijing_sthgcn_test_summary.md
    """
    os.makedirs(results_path, exist_ok=True)
    tag = run_tag or "run"
    prefix = f"{tag}_{split}"

    flat = metrics_dict(recall_res, ndcg_res, map_res, mrr_res, loss=loss, ks=ks)
    payload = {
        "split": split,
        "epoch": epoch,
        "step": step,
        "metrics": metrics_to_serializable(flat),
    }
    if extra:
        payload["extra"] = metrics_to_serializable(extra)

    json_path = osp.join(results_path, f"{prefix}_metrics.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    # Append one row to a history CSV (useful for repeated valid evals)
    csv_path = osp.join(results_path, f"{tag}_{split}_history.csv")
    row = {"split": split, "epoch": epoch if epoch is not None else "", "step": step if step is not None else ""}
    row.update({k: flat[k] for k in flat if k.startswith("Acc@") or k in ("MRR", "loss")})
    for k in ks:
        if f"NDCG@{k}" in flat:
            row[f"NDCG@{k}"] = flat[f"NDCG@{k}"]
    write_header = not osp.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    md_path = osp.join(results_path, f"{prefix}_summary.md")
    lines = [
        f"# STHGCN {split} results",
        "",
        f"- run: `{tag}`",
    ]
    if epoch is not None:
        lines.append(f"- epoch: {epoch}")
    if step is not None:
        lines.append(f"- step: {step}")
    if loss is not None:
        lines.append(f"- loss: {float(loss):.4f}")
    lines.extend([
        "",
        "Acc@K ≡ Recall@K (single-label next-POI).",
        "",
        "| Metric | Value |",
        "|--------|------:|",
    ])
    for k in ks:
        if f"Acc@{k}" in flat:
            lines.append(f"| Acc@{k} | {flat[f'Acc@{k}']:.4f} |")
    lines.append(f"| MRR | {flat['MRR']:.4f} |")
    for k in ks:
        if f"NDCG@{k}" in flat:
            lines.append(f"| NDCG@{k} | {flat[f'NDCG@{k}']:.4f} |")
    for k in ks:
        if f"MAP@{k}" in flat:
            lines.append(f"| MAP@{k} | {flat[f'MAP@{k}']:.4f} |")
    lines.append("")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    logging.info(f"[Results] Saved {split} metrics → {results_path}/ ({prefix}_*)")
    return json_path, csv_path, md_path


def test_step(model, data, ks=(1, 5, 10, 20), *, split="eval", epoch=None, step=None):
    model.eval()
    loss_list = []
    pred_list = []
    label_list = []
    with torch.no_grad():
        for row in tqdm(data):
            split_index = torch.max(row.adjs_t[1].storage.row()).tolist()
            row = row.to(model.device)

            input_data = {
                'x': row.x,
                'edge_index': row.adjs_t,
                'edge_attr': row.edge_attrs,
                'split_index': split_index,
                'delta_ts': row.edge_delta_ts,
                'delta_ss': row.edge_delta_ss,
                'edge_type': row.edge_types
            }

            out, loss = model(input_data, label=row.y[:, 0], mode='test')
            loss_list.append(loss.cpu().detach().numpy().tolist())
            ranking = torch.sort(out, descending=True)[1]
            pred_list.append(ranking.cpu().detach())
            label_list.append(row.y[:, :1].cpu())
    pred_ = torch.cat(pred_list, dim=0)
    label_ = torch.cat(label_list, dim=0)
    recalls, NDCGs, MAPs = {}, {}, {}
    mean_loss = float(np.mean(loss_list))
    for k_ in ks:
        recalls[k_] = recall(label_, pred_, k_).cpu().detach().numpy().tolist()
        NDCGs[k_] = ndcg(label_, pred_, k_).cpu().detach().numpy().tolist()
        MAPs[k_] = map_k(label_, pred_, k_).cpu().detach().numpy().tolist()
    mrr_res = mrr(label_, pred_).cpu().detach().numpy().tolist()

    logging.info(
        format_metrics_block(
            split,
            recalls,
            NDCGs,
            MAPs,
            mrr_res,
            loss=mean_loss,
            epoch=epoch,
            step=step,
            ks=ks,
        )
    )
    logging.info(
        f"[{split}] "
        + format_metrics_line(recalls, mrr_res, ndcg_res=NDCGs, loss=mean_loss, ks=ks)
    )
    return recalls, NDCGs, MAPs, mrr_res, mean_loss
