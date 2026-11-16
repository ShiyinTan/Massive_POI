import logging
import os
import os.path as osp
import datetime
import torch
import random
from tqdm import tqdm
import argparse
from torch.utils.tensorboard import SummaryWriter
from preprocess import preprocess
from utils import seed_torch, set_logger, Cfg, count_parameters, test_step, save_model, save_eval_results
from utils.log_format import format_epoch_train, format_metrics_line
from layer import NeighborSampler
from dataset import LBSNDataset
from model import STHGCN, SequentialTransformer


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--yaml_file', help='The configuration file.', required=True)
    parser.add_argument('--multi_run_mode', help='Run multiple experiments with the same config.', action='store_true')
    args = parser.parse_args()
    conf_file = args.yaml_file

    cfg = Cfg(conf_file)

    sizes = [int(i) for i in cfg.model_args.sizes.split('-')]
    cfg.model_args.sizes = sizes

    # cuda setting
    if int(cfg.run_args.gpu) >= 0:
        device = 'cuda:' + str(cfg.run_args.gpu)
    else:
        device = 'cpu'
    cfg.run_args.device = device

    # for multiple runs, seed is replaced with random value
    if args.multi_run_mode:
        cfg.run_args.seed = None
    if cfg.run_args.seed is None:
        seed = random.randint(0, 100000000)
    else:
        seed = int(cfg.run_args.seed)

    seed_torch(seed)

    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    # Clear run folder: outputs/<dataset>_<model>_<timestamp>/{checkpoint,logs,results}
    run_tag = f'{cfg.dataset_args.dataset_name}_{cfg.model_args.model_name}_{current_time}'
    cfg.run_args.output_dir = osp.join('outputs', run_tag)
    cfg.run_args.save_path = osp.join(cfg.run_args.output_dir, 'checkpoint')
    cfg.run_args.log_path = osp.join(cfg.run_args.output_dir, 'logs')
    cfg.run_args.results_path = osp.join(cfg.run_args.output_dir, 'results')
    cfg.run_args.run_tag = run_tag

    for path in (cfg.run_args.save_path, cfg.run_args.log_path, cfg.run_args.results_path):
        os.makedirs(path, exist_ok=True)

    set_logger(cfg.run_args)
    logging.info(f'[Setup] Output directory: {cfg.run_args.output_dir}')
    summary_writer = SummaryWriter(log_dir=cfg.run_args.save_path)

    hparam_dict = {}
    for group, hparam in cfg.__dict__.items():
        hparam_dict.update(hparam.__dict__)
    hparam_dict['seed'] = seed
    hparam_dict['sizes'] = '-'.join([str(item) for item in cfg.model_args.sizes])

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

    # Best weights for do_test when disk checkpointing is disabled (default).
    best_checkpoint = None

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
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                logging.info('[Training] Weights-only checkpoint — optimizer freshly initialized')
        else:
            logging.info(f'[Training] Randomly Initializing Model...')
            init_step = 0
        step = init_step

        # Disk checkpoints are off by default (too large); keep best weights in RAM for test.
        save_checkpoint = bool(getattr(cfg.run_args, 'save_checkpoint', False))
        checkpoint_contents = getattr(cfg.run_args, 'checkpoint_contents', 'weights')
        cfg.run_args.save_checkpoint = save_checkpoint
        cfg.run_args.checkpoint_contents = checkpoint_contents
        logging.info(
            f'[Training] Checkpoint disk save: {save_checkpoint} '
            f'(contents={checkpoint_contents}; best-only when enabled)'
        )

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
                    logging.info(f'[Evaluating] Valid  epoch={eph + 1}/{cfg.run_args.epoch}  step={global_step}')
                    recall_res, ndcg_res, map_res, mrr_res, eval_loss = test_step(
                        model, data=sampler_validate, split='valid', epoch=eph + 1, step=global_step
                    )
                    summary_writer.add_scalar(f'validate/Acc@1', 100 * recall_res[1], global_step)
                    summary_writer.add_scalar(f'validate/Acc@5', 100 * recall_res[5], global_step)
                    summary_writer.add_scalar(f'validate/Acc@10', 100 * recall_res[10], global_step)
                    summary_writer.add_scalar(f'validate/Acc@20', 100 * recall_res[20], global_step)
                    # Keep legacy Recall@* TB tags for older dashboards
                    summary_writer.add_scalar(f'validate/Recall@1', 100 * recall_res[1], global_step)
                    summary_writer.add_scalar(f'validate/Recall@5', 100 * recall_res[5], global_step)
                    summary_writer.add_scalar(f'validate/Recall@10', 100 * recall_res[10], global_step)
                    summary_writer.add_scalar(f'validate/Recall@20', 100 * recall_res[20], global_step)
                    summary_writer.add_scalar(f'validate/MRR', mrr_res, global_step)
                    summary_writer.add_scalar(f'validate/eval_loss', eval_loss, global_step)
                    summary_writer.add_scalar('train/learning_rate', current_learning_rate, global_step)

                    save_eval_results(
                        cfg.run_args.results_path,
                        'valid',
                        recall_res,
                        ndcg_res,
                        map_res,
                        mrr_res,
                        loss=eval_loss,
                        epoch=eph + 1,
                        step=global_step,
                        run_tag=cfg.run_args.run_tag,
                    )

                    metrics = 4 * recall_res[1] + recall_res[20]

                    # keep / optionally persist best-only weights (not every step)
                    if metrics > best_metrics:
                        save_variable_list = {
                            'step': global_step,
                            'current_learning_rate': current_learning_rate,
                            'warm_up_steps': warm_up_steps,
                            'cooldown_rate': cfg.run_args.cooldown_rate,
                            'sizes': sizes
                        }
                        # CPU copy avoids pinning a large GPU tensor graph in RAM
                        best_checkpoint = {
                            **save_variable_list,
                            'model_state_dict': {
                                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                            },
                        }
                        logging.info(
                            f'[Training] Best valid so far — keep weights'
                            f'{" + write disk" if save_checkpoint else " (no disk)"}  '
                            f'epoch={eph + 1} step={global_step}  '
                            + format_metrics_line(recall_res, mrr_res)
                        )
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
            logging.info(format_epoch_train(eph, cfg.run_args.epoch, global_step, epoch_loss))
            summary_writer.add_scalar('train/loss_epoch', epoch_loss, eph)

    if cfg.run_args.do_test:
        logging.info('[Evaluating] Test set (best checkpoint)...')

        ckpt_path = osp.join(cfg.run_args.save_path, 'checkpoint.pt')
        if best_checkpoint is not None:
            checkpoint = best_checkpoint
            logging.info('[Evaluating] Loading best weights from memory')
        elif osp.isfile(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            logging.info(f'[Evaluating] Loading best weights from {ckpt_path}')
        else:
            raise FileNotFoundError(
                f'No best checkpoint in memory or at {ckpt_path}. '
                'Train with do_validate, or set save_checkpoint: true / provide init_checkpoint.'
            )
        model.load_state_dict(checkpoint['model_state_dict'])
        recall_res, ndcg_res, map_res, mrr_res, loss = test_step(
            model, sampler_test, split='test', step=checkpoint.get('step')
        )
        num_params = count_parameters(model)
        metric_dict = {
            'hparam/num_params': num_params,
            'hparam/Acc@1': recall_res[1],
            'hparam/Acc@5': recall_res[5],
            'hparam/Acc@10': recall_res[10],
            'hparam/Acc@20': recall_res[20],
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
        save_eval_results(
            cfg.run_args.results_path,
            'test',
            recall_res,
            ndcg_res,
            map_res,
            mrr_res,
            loss=loss,
            step=checkpoint.get('step'),
            extra={'num_params': num_params},
            run_tag=cfg.run_args.run_tag,
        )
        logging.info(
            f'[Evaluating] Test final  '
            + format_metrics_line(recall_res, mrr_res, ndcg_res=ndcg_res, loss=loss)
        )
        logging.info(f'[Results] Full run folder: {cfg.run_args.output_dir}')
        summary_writer.add_hparams(hparam_dict, metric_dict)
        summary_writer.close()
