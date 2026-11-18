import csv
import logging
import os
import os.path as osp
import datetime
import torch
import random
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from preprocess import preprocess
from utils import seed_torch, set_logger, Cfg, count_parameters, test_step, save_model, get_root_dir
from utils.ablation_cli import build_arg_parser, apply_ablation_args, ablation_run_name
from layer import NeighborSampler
from dataset import LBSNDataset
from model import STHGCN, SequentialTransformer


SUMMARY_CSV_HEADER = [
    'time', 'city', 'run_name', 'seed', 'use_user_embed', 'hypergraph_edges',
    'graph_scope', 'unseen_prefix_poi', 'late_sequence', 'gpu',
    'Recall@1', 'Recall@5', 'Recall@10', 'Recall@20', 'MRR', 'log_dir',
]


def _fsync_file(f):
    """Flush Python buffers and fsync so a kill keeps the last summary row."""
    f.flush()
    os.fsync(f.fileno())


def append_summary_csv(root_dir, row: dict):
    """Append one finished-test row to results/summary.csv (create/upgrade header if needed)."""
    path = osp.join(root_dir, 'results', 'summary.csv')
    os.makedirs(osp.dirname(path), exist_ok=True)
    existing_rows = []
    if osp.exists(path) and osp.getsize(path) > 0:
        with open(path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and list(reader.fieldnames) != SUMMARY_CSV_HEADER:
                existing_rows = list(reader)
            elif reader.fieldnames:
                # Header already current; append only.
                with open(path, 'a', newline='') as af:
                    writer = csv.DictWriter(af, fieldnames=SUMMARY_CSV_HEADER)
                    writer.writerow({k: row.get(k, '') for k in SUMMARY_CSV_HEADER})
                    _fsync_file(af)
                return
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_CSV_HEADER)
        writer.writeheader()
        for old in existing_rows:
            writer.writerow({k: old.get(k, '') for k in SUMMARY_CSV_HEADER})
        writer.writerow({k: row.get(k, '') for k in SUMMARY_CSV_HEADER})
        _fsync_file(f)


if __name__ == '__main__':
    # Parse arguments (ablation defaults = paper std_conf behavior)
    parser = build_arg_parser()
    args = parser.parse_args()
    conf_file = args.yaml_file

    cfg = Cfg(conf_file)
    cfg = apply_ablation_args(cfg, args)

    sizes = [int(i) for i in cfg.model_args.sizes.split('-')]
    cfg.model_args.sizes = sizes

    # Device: yaml run_args.gpu or CLI --gpu (-1 = CPU)
    if int(cfg.run_args.gpu) >= 0:
        device = 'cuda:' + str(cfg.run_args.gpu)
    else:
        device = 'cpu'
    cfg.run_args.device = device

    # --seed overrides yaml; --multi_run_mode draws a random seed when neither is set
    if getattr(args, 'seed', None) is not None:
        seed = int(args.seed)
    elif args.multi_run_mode or cfg.run_args.seed is None:
        seed = random.randint(0, 100000000)
    else:
        seed = int(cfg.run_args.seed)
    cfg.run_args.seed = seed

    seed_torch(seed)

    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    city = cfg.dataset_args.dataset_name
    run_name = ablation_run_name(args)
    # results/<city>/<run_name>/<timestamp>/ — ablation (or paper) visible before timestamp
    run_dir = osp.join('results', city, run_name, current_time)
    cfg.run_args.save_path = osp.join(run_dir, 'checkpoints')  # checkpoint.pt
    cfg.run_args.log_path = osp.join(run_dir, 'log')           # train.log + metrics
    tb_path = osp.join(run_dir, 'tensorboard')

    if not osp.isdir(cfg.run_args.save_path):
        os.makedirs(cfg.run_args.save_path)
    if not osp.isdir(cfg.run_args.log_path):
        os.makedirs(cfg.run_args.log_path)
    if not osp.isdir(tb_path):
        os.makedirs(tb_path)

    set_logger(cfg.run_args)
    summary_writer = SummaryWriter(log_dir=tb_path)

    hparam_dict = {}
    for group, hparam in cfg.__dict__.items():
        hparam_dict.update(hparam.__dict__)
    hparam_dict['seed'] = seed
    hparam_dict['sizes'] = '-'.join([str(item) for item in cfg.model_args.sizes])
    hparam_dict['use_user_embed'] = args.use_user_embed
    hparam_dict['hypergraph_edges'] = args.hypergraph_edges
    hparam_dict['graph_scope'] = args.graph_scope
    hparam_dict['unseen_prefix_poi'] = args.unseen_prefix_poi
    hparam_dict['late_sequence'] = args.late_sequence
    hparam_dict['run_name'] = run_name

    logging.info(
        f'[Ablation] run_name={run_name} use_user_embed={args.use_user_embed} '
        f'hypergraph_edges={args.hypergraph_edges} graph_scope={args.graph_scope} '
        f'unseen_prefix_poi={args.unseen_prefix_poi} '
        f'late_sequence={args.late_sequence} run_dir={run_dir}'
    )

    # Preprocess data
    preprocess(cfg)

    # Initialize dataset
    lbsn_dataset = LBSNDataset(cfg)
    cfg.dataset_args.spatial_slots = lbsn_dataset.spatial_slots
    cfg.dataset_args.num_user = lbsn_dataset.num_user
    cfg.dataset_args.num_poi = lbsn_dataset.num_poi
    cfg.dataset_args.num_category = lbsn_dataset.num_category
    cfg.dataset_args.padding_poi_id = lbsn_dataset.padding_poi_id
    cfg.dataset_args.padding_user_id = lbsn_dataset.padding_user_id
    cfg.dataset_args.padding_poi_category = lbsn_dataset.padding_poi_category
    cfg.dataset_args.padding_hour_id = lbsn_dataset.padding_hour_id
    cfg.dataset_args.padding_weekday_id = lbsn_dataset.padding_weekday_id

    # Initialize neighbor sampler(dataloader)
    sampler_train, sampler_validate, sampler_test = None, None, None

    if cfg.run_args.do_train:
        sampler_train = NeighborSampler(
            lbsn_dataset.x,
            lbsn_dataset.edge_index,
            lbsn_dataset.edge_attr,
            intra_jaccard_threshold=cfg.model_args.intra_jaccard_threshold,
            inter_jaccard_threshold=cfg.model_args.inter_jaccard_threshold,
            edge_t=lbsn_dataset.edge_t,
            edge_delta_t=lbsn_dataset.edge_delta_t,
            edge_type=lbsn_dataset.edge_type,
            sizes=sizes,
            sample_idx=lbsn_dataset.sample_idx_train,
            node_idx=lbsn_dataset.node_idx_train,
            edge_delta_s=lbsn_dataset.edge_delta_s,
            max_time=lbsn_dataset.max_time_train,
            label=lbsn_dataset.label_train,
            batch_size=cfg.run_args.batch_size,
            num_workers=0 if device == 'cpu' else cfg.run_args.num_workers,
            shuffle=True,
            pin_memory=True
        )

    if cfg.run_args.do_validate:
        sampler_validate = NeighborSampler(
            lbsn_dataset.x,
            lbsn_dataset.edge_index,
            lbsn_dataset.edge_attr,
            intra_jaccard_threshold=cfg.model_args.intra_jaccard_threshold,
            inter_jaccard_threshold=cfg.model_args.inter_jaccard_threshold,
            edge_t=lbsn_dataset.edge_t,
            edge_delta_t=lbsn_dataset.edge_delta_t,
            edge_type=lbsn_dataset.edge_type,
            sizes=sizes,
            sample_idx=lbsn_dataset.sample_idx_valid,
            node_idx=lbsn_dataset.node_idx_valid,
            edge_delta_s=lbsn_dataset.edge_delta_s,
            max_time=lbsn_dataset.max_time_valid,
            label=lbsn_dataset.label_valid,
            batch_size=cfg.run_args.eval_batch_size,
            num_workers=0 if device == 'cpu' else cfg.run_args.num_workers,
            shuffle=False,
            pin_memory=True
        )

    if cfg.run_args.do_test:
        sampler_test = NeighborSampler(
            lbsn_dataset.x,
            lbsn_dataset.edge_index,
            lbsn_dataset.edge_attr,
            intra_jaccard_threshold=cfg.model_args.intra_jaccard_threshold,
            inter_jaccard_threshold=cfg.model_args.inter_jaccard_threshold,
            edge_t=lbsn_dataset.edge_t,
            edge_delta_t=lbsn_dataset.edge_delta_t,
            edge_type=lbsn_dataset.edge_type,
            sizes=sizes,
            sample_idx=lbsn_dataset.sample_idx_test,
            node_idx=lbsn_dataset.node_idx_test,
            edge_delta_s=lbsn_dataset.edge_delta_s,
            max_time=lbsn_dataset.max_time_test,
            label=lbsn_dataset.label_test,
            batch_size=cfg.run_args.eval_batch_size,
            num_workers=0 if device == 'cpu' else cfg.run_args.num_workers,
            shuffle=False,
            pin_memory=True
        )

    if cfg.model_args.model_name == 'sthgcn':
        model = STHGCN(cfg)
    elif cfg.model_args.model_name == 'seq_transformer':
        model = SequentialTransformer(cfg)
    else:
        raise NotImplementedError(
            f'[Training] Model {cfg.model_args.name}, please choose from ["sthgcn", "seq_transformer"]'
        )

    model = model.to(device)
    logging.info(f'[Training] Seed: {seed}')
    logging.info('[Training] Model Parameter Configuration:')
    for name, param in model.named_parameters():
        logging.info(f'[Training] Parameter {name}: {param.size()}, require_grad = {param.requires_grad}')
    logging.info(f'[Training] #Parameters: {count_parameters(model)}')

    if cfg.run_args.do_train:
        current_learning_rate = cfg.run_args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=current_learning_rate
        )
        if cfg.run_args.warm_up_steps:
            warm_up_steps = cfg.run_args.warm_up_steps
        else:
            num_train_samples = lbsn_dataset.sample_idx_train.shape[0]
            num_epochs = cfg.run_args.epoch
            total_steps = num_train_samples // cfg.run_args.batch_size * num_epochs
            warm_up_steps = total_steps // 2

        init_step = 0
        if cfg.run_args.init_checkpoint:
            # Restore model from checkpoint directory
            # manually set in yml
            logging.info(f'[Training] Loading checkpoint %s...' % cfg.run_args.init_checkpoint)
            checkpoint = torch.load(osp.join(cfg.run_args.init_checkpoint, 'checkpoint.pt'))
            init_step = checkpoint['step']
            model.load_state_dict(checkpoint['model_state_dict'])
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            cooldown_rate = checkpoint['cooldown_rate']
            sizes = checkpoint['sizes']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            logging.info(f'[Training] Randomly Initializing Model...')
            init_step = 0
        step = init_step

        # Set valid dataloader as it would be evaluated during training
        logging.info(f'[Training] Initial learning rate: {current_learning_rate}')

        # Training Loop
        best_metrics = 0.0
        global_step = 0
        for eph in range(cfg.run_args.epoch):
            training_logs = []
            if global_step >= cfg.run_args.max_steps:
                break
            for data in tqdm(sampler_train):
                model.train()
                split_index = torch.max(data.adjs_t[1].storage.row()).tolist()
                data = data.to(device)
                input_data = {
                    'x': data.x,
                    'edge_index': data.adjs_t,
                    'edge_attr': data.edge_attrs,
                    'split_index': split_index,
                    'delta_ts': data.edge_delta_ts,
                    'delta_ss': data.edge_delta_ss,
                    'edge_type': data.edge_types
                }

                out, loss = model(input_data, label=data.y[:, 0])
                training_logs.append(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                summary_writer.add_scalar(f'train/loss_step', loss, global_step)

                if cfg.run_args.do_validate and global_step % cfg.run_args.valid_steps == 0:
                    logging.info(f'[Evaluating] Evaluating on Valid Dataset...')

                    logging.info(f'[Evaluating] Epoch {eph}, step {global_step}:')
                    recall_res, ndcg_res, map_res, mrr_res, eval_loss = test_step(model, data=sampler_validate)
                    summary_writer.add_scalar(f'validate/Recall@1', 100*recall_res[1], global_step)
                    summary_writer.add_scalar(f'validate/Recall@5', 100*recall_res[5], global_step)
                    summary_writer.add_scalar(f'validate/Recall@10', 100*recall_res[10], global_step)
                    summary_writer.add_scalar(f'validate/Recall@20', 100*recall_res[20], global_step)
                    summary_writer.add_scalar(f'validate/MRR', mrr_res, global_step)
                    summary_writer.add_scalar(f'validate/eval_loss', eval_loss, global_step)
                    summary_writer.add_scalar('train/learning_rate', current_learning_rate, global_step)

                    metrics = 4 * recall_res[1] + recall_res[20]

                    # save model based on compositional recall metrics
                    if metrics > best_metrics:
                        save_variable_list = {
                            'step': global_step,
                            'current_learning_rate': current_learning_rate,
                            'warm_up_steps': warm_up_steps,
                            'cooldown_rate': cfg.run_args.cooldown_rate,
                            'sizes': sizes
                        }
                        logging.info(f'[Training] Save model at step {global_step} epoch {eph}')
                        save_model(model, optimizer, save_variable_list, cfg.run_args, hparam_dict)
                        best_metrics = metrics

                # learning rate schedule
                if global_step >= warm_up_steps:
                    current_learning_rate = current_learning_rate / 10
                    logging.info(f'[Training] Change learning_rate to {current_learning_rate} at step {global_step}')
                    optimizer = torch.optim.Adam(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        lr=current_learning_rate
                    )
                    warm_up_steps = warm_up_steps * cfg.run_args.cooldown_rate

                if global_step >= cfg.run_args.max_steps:
                    break
                global_step += 1

            epoch_loss = sum([loss for loss in training_logs]) / len(training_logs)
            logging.info(f'[Training] Average train loss at step {global_step} is {epoch_loss}:')
            summary_writer.add_scalar('train/loss_epoch', epoch_loss, eph)

    if cfg.run_args.do_test:
        logging.info('[Evaluating] Start evaluating on test set...')

        checkpoint = torch.load(
            osp.join(cfg.run_args.save_path, 'checkpoint.pt'),
            map_location=device,
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        recall_res, ndcg_res, map_res, mrr_res, loss = test_step(model, sampler_test)
        num_params = count_parameters(model)
        metric_dict = {
            'hparam/num_params': num_params,
            'hparam/Recall@1': recall_res[1],
            'hparam/Recall@5': recall_res[5],
            'hparam/Recall@10': recall_res[10],
            'hparam/Recall@20': recall_res[20],
            'hparam/NDCG@1': ndcg_res[1],
            'hparam/NDCG@5': ndcg_res[5],
            'hparam/NDCG@10': ndcg_res[10],
            'hparam/NDCG@20': ndcg_res[20],
            'hparam/MAP@1': map_res[1],
            'hparam/MAP@5': map_res[5],
            'hparam/MAP@10': map_res[10],
            'hparam/MAP@20': map_res[20],
            'hparam/MRR': mrr_res,
        }
        logging.info(f'[Evaluating] Test evaluation result : {metric_dict}')
        summary_writer.add_hparams(hparam_dict, metric_dict)
        summary_writer.close()

        append_summary_csv(
            get_root_dir(),
            {
                'time': current_time,
                'city': city,
                'run_name': run_name,
                'seed': seed,
                'use_user_embed': args.use_user_embed,
                'hypergraph_edges': args.hypergraph_edges,
                'graph_scope': args.graph_scope,
                'unseen_prefix_poi': args.unseen_prefix_poi,
                'late_sequence': args.late_sequence,
                'gpu': cfg.run_args.gpu,
                'Recall@1': recall_res[1],
                'Recall@5': recall_res[5],
                'Recall@10': recall_res[10],
                'Recall@20': recall_res[20],
                'MRR': mrr_res,
                'log_dir': run_dir,
            },
        )
        logging.info(f'[Evaluating] Appended row to results/summary.csv (run_dir={run_dir})')
