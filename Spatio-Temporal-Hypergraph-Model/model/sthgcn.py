import torch
from torch import nn
from layer import (
    CheckinEmbedding,
    EdgeEmbedding,
    HypergraphTransformer,
    TimeEncoder,
    DistanceEncoderHSTLSTM,
    DistanceEncoderSTAN,
    DistanceEncoderSimple
)


class STHGCN(nn.Module):
    """STHGCN with optional single-sequence late pull-out.

    Default (``late_sequence=none``): paper path — CheckinEmbedding concat → one
    HypergraphTransformer stack → Linear → POI logits.

    Ablation (``poi`` / ``category`` / ``time`` / ``location``): that one sequence
    is encoded on its own HG stack; the others stay on the paper fused path
    (pulled P/C/T slots zeroed in the early concat; location was never in it).
    Just before softmax: concat(main_rep, pulled_rep) → MLP → POI logits.
    User embedding always stays on the fused path.
    """

    LATE_CHOICES = ('poi', 'category', 'time', 'location')

    def __init__(self, cfg):
        super(STHGCN, self).__init__()
        self.device = cfg.run_args.device
        self.batch_size = cfg.run_args.batch_size
        self.eval_batch_size = cfg.run_args.eval_batch_size
        # Ablation: hypergraph_edges in {both, ci2traj, traj2traj, none}; default both.
        self.hypergraph_edges = getattr(cfg.model_args, 'hypergraph_edges', 'both')
        self.do_traj2traj = bool(cfg.model_args.do_traj2traj) and self.hypergraph_edges in ('both', 'traj2traj')
        self.do_ci2traj = self.hypergraph_edges in ('both', 'ci2traj')
        # Ablation: late_sequence in {none, poi, category, time, location}; default none.
        late_sequence = getattr(cfg.model_args, 'late_sequence', 'none')
        if isinstance(late_sequence, str):
            late_sequence = late_sequence.lower()
        self.late_sequence = late_sequence
        self.distance_encoder_type = cfg.model_args.distance_encoder_type
        self.dropout_rate = cfg.model_args.dropout_rate
        self.generate_edge_attr = cfg.model_args.generate_edge_attr
        self.num_conv_layers = len(cfg.model_args.sizes)
        self.num_poi = cfg.dataset_args.num_poi
        self.embed_fusion_type = cfg.model_args.embed_fusion_type
        self.embed_size = cfg.model_args.embed_size
        use_user_embed = getattr(cfg.model_args, 'use_user_embed', True)
        if isinstance(use_user_embed, str):
            use_user_embed = use_user_embed.lower() in ('1', 'true', 'on', 'yes')
        self.use_user_embed = bool(use_user_embed)
        self.checkin_embedding_layer = CheckinEmbedding(
            embed_size=cfg.model_args.embed_size,
            fusion_type=self.embed_fusion_type,
            dataset_args=cfg.dataset_args,
            use_user_embed=self.use_user_embed,
        )

        if cfg.model_args.activation == 'elu':
            self.act = nn.ELU()
        elif cfg.model_args.activation == 'relu':
            self.act = nn.RReLU()
        elif cfg.model_args.activation == 'leaky_relu':
            self.act = nn.LeakyReLU()
        else:
            self.act = torch.tanh

        # Paper / fused path: always present (same tensors when late_sequence=none).
        self.checkin_embed_size = self.checkin_embedding_layer.output_embed_size
        continuous_encoder_dim = (
            self.checkin_embed_size
            if cfg.conv_args.time_fusion_mode == 'add'
            else cfg.model_args.st_embed_size
        )
        main_stack = self._build_hg_stack(self.checkin_embed_size, continuous_encoder_dim, cfg)
        self.conv_for_time_filter = main_stack['conv_for_time_filter']
        self.norms_for_time_filter = main_stack['norms_for_time_filter']
        self.dropout_for_time_filter = main_stack['dropout_for_time_filter']
        self.conv_list = main_stack['conv_list']
        self.norms_list = main_stack['norms_list']
        self.dropout_list = main_stack['dropout_list']
        self._init_edge_modules(
            self.checkin_embed_size, continuous_encoder_dim, cfg, prefix=''
        )
        self.linear = nn.Linear(self.checkin_embed_size, self.num_poi)

        self.late_stack = None
        self.late_mlp = None
        self.location_proj = None
        if self.late_sequence in self.LATE_CHOICES:
            late_cont_dim = (
                self.embed_size
                if cfg.conv_args.time_fusion_mode == 'add'
                else cfg.model_args.st_embed_size
            )
            self.late_stack = self._build_hg_stack(self.embed_size, late_cont_dim, cfg)
            self._init_edge_modules(self.embed_size, late_cont_dim, cfg, prefix='late_')
            if self.late_sequence == 'location':
                self.location_proj = nn.Linear(2, self.embed_size)
            late_in = self.checkin_embed_size + self.embed_size
            self.late_mlp = nn.Sequential(
                nn.Linear(late_in, late_in),
                nn.ReLU(),
                nn.Linear(late_in, self.num_poi),
            )

        self.loss_func = nn.CrossEntropyLoss()

    def _init_edge_modules(self, channel_size, continuous_encoder_dim, cfg, prefix=''):
        """Create edge/time/distance modules; ``prefix=''`` (paper) or ``'late_'``."""
        setattr(
            self,
            f'{prefix}edge_type_embedding_layer',
            EdgeEmbedding(
                embed_size=channel_size,
                fusion_type=self.embed_fusion_type,
                num_edge_type=cfg.model_args.num_edge_type,
            ),
        )
        if self.generate_edge_attr:
            edge_attr = EdgeEmbedding(
                embed_size=channel_size,
                fusion_type=self.embed_fusion_type,
                num_edge_type=cfg.model_args.num_edge_type,
            )
        else:
            if cfg.conv_args.edge_fusion_mode == 'add':
                edge_attr = nn.Linear(3, channel_size)
            else:
                edge_attr = None
        setattr(self, f'{prefix}edge_attr_embedding_layer', edge_attr)

        setattr(
            self,
            f'{prefix}continuous_time_encoder',
            TimeEncoder(cfg.model_args, continuous_encoder_dim),
        )
        if self.distance_encoder_type == 'stan':
            dist = DistanceEncoderSTAN(
                cfg.model_args, continuous_encoder_dim, cfg.dataset_args.spatial_slots
            )
        elif self.distance_encoder_type == 'time':
            dist = TimeEncoder(cfg.model_args, continuous_encoder_dim)
        elif self.distance_encoder_type == 'hstlstm':
            dist = DistanceEncoderHSTLSTM(
                cfg.model_args, continuous_encoder_dim, cfg.dataset_args.spatial_slots
            )
        elif self.distance_encoder_type == 'simple':
            dist = DistanceEncoderSimple(
                cfg.model_args, continuous_encoder_dim, cfg.dataset_args.spatial_slots
            )
        else:
            raise ValueError(
                f"Get wrong distance_encoder_type argument: {cfg.model_args.distance_encoder_type}!"
            )
        setattr(self, f'{prefix}continuous_distance_encoder', dist)

    def _edge_bundle(self, prefix=''):
        return {
            'edge_type_embedding_layer': getattr(self, f'{prefix}edge_type_embedding_layer'),
            'edge_attr_embedding_layer': getattr(self, f'{prefix}edge_attr_embedding_layer'),
            'continuous_time_encoder': getattr(self, f'{prefix}continuous_time_encoder'),
            'continuous_distance_encoder': getattr(self, f'{prefix}continuous_distance_encoder'),
        }

    def _build_hg_stack(self, channel_size, continuous_encoder_dim, cfg):
        """One hypergraph encoder stack (time-filter ci2traj + optional traj2traj hops)."""
        stack = nn.ModuleDict()
        stack['conv_for_time_filter'] = HypergraphTransformer(
            in_channels=channel_size,
            out_channels=channel_size,
            attn_heads=cfg.conv_args.num_attention_heads,
            residual_beta=cfg.conv_args.residual_beta,
            learn_beta=cfg.conv_args.learn_beta,
            dropout=cfg.conv_args.conv_dropout_rate,
            trans_method=cfg.conv_args.trans_method,
            edge_fusion_mode=cfg.conv_args.edge_fusion_mode,
            time_fusion_mode=cfg.conv_args.time_fusion_mode,
            head_fusion_mode=cfg.conv_args.head_fusion_mode,
            residual_fusion_mode=None,
            edge_dim=None,
            rel_embed_dim=channel_size,
            time_embed_dim=continuous_encoder_dim,
            dist_embed_dim=continuous_encoder_dim,
            negative_slope=cfg.conv_args.negative_slope,
            have_query_feature=False
        )
        stack['norms_for_time_filter'] = nn.BatchNorm1d(channel_size)
        stack['dropout_for_time_filter'] = nn.Dropout(self.dropout_rate)

        conv_list = nn.ModuleList()
        norms_list = nn.ModuleList()
        dropout_list = nn.ModuleList()
        if self.do_traj2traj:
            for i in range(self.num_conv_layers):
                if i == 0:
                    have_query_feature = False
                    residual_fusion_mode = None
                    edge_size = None
                else:
                    have_query_feature = True
                    residual_fusion_mode = cfg.conv_args.residual_fusion_mode
                    if self.generate_edge_attr or cfg.conv_args.edge_fusion_mode == 'add':
                        edge_size = channel_size
                    else:
                        edge_size = 3
                conv_list.append(
                    HypergraphTransformer(
                        in_channels=channel_size,
                        out_channels=channel_size,
                        attn_heads=cfg.conv_args.num_attention_heads,
                        residual_beta=cfg.conv_args.residual_beta,
                        learn_beta=cfg.conv_args.learn_beta,
                        dropout=cfg.conv_args.conv_dropout_rate,
                        trans_method=cfg.conv_args.trans_method,
                        edge_fusion_mode=cfg.conv_args.edge_fusion_mode,
                        time_fusion_mode=cfg.conv_args.time_fusion_mode,
                        head_fusion_mode=cfg.conv_args.head_fusion_mode,
                        residual_fusion_mode=residual_fusion_mode,
                        edge_dim=edge_size,
                        rel_embed_dim=channel_size,
                        time_embed_dim=continuous_encoder_dim,
                        dist_embed_dim=continuous_encoder_dim,
                        negative_slope=cfg.conv_args.negative_slope,
                        have_query_feature=have_query_feature
                    )
                )
                norms_list.append(nn.BatchNorm1d(channel_size))
                dropout_list.append(nn.Dropout(self.dropout_rate))
        stack['conv_list'] = conv_list
        stack['norms_list'] = norms_list
        stack['dropout_list'] = dropout_list
        return stack

    def _main_stack(self):
        return {
            'conv_for_time_filter': self.conv_for_time_filter,
            'norms_for_time_filter': self.norms_for_time_filter,
            'dropout_for_time_filter': self.dropout_for_time_filter,
            'conv_list': self.conv_list,
            'norms_list': self.norms_list,
            'dropout_list': self.dropout_list,
        }

    def _batch_size(self, label, mode):
        if label is not None:
            return int(label.size(0))
        if mode in ('test', 'validate'):
            return self.eval_batch_size
        return self.batch_size

    def _hypergraph_encode(self, x, data, label, mode, stack, channel_size, edge_prefix=''):
        """Run a feature tensor through a hypergraph stack (+ matching edge modules)."""
        edges = self._edge_bundle(edge_prefix)
        if self.hypergraph_edges == 'none':
            batch_size = self._batch_size(label, mode)
            if x.size(0) > 0:
                return x.mean(dim=0, keepdim=True).expand(batch_size, -1)
            return torch.zeros(batch_size, channel_size, device=x.device, dtype=x.dtype)

        edge_time_embed = edges['continuous_time_encoder'](data['delta_ts'][0] / (60 * 60))
        if self.distance_encoder_type == 'stan':
            edge_distance_embed = edges['continuous_distance_encoder'](
                data['delta_ss'][0], dist_type='ch2tj'
            )
        else:
            edge_distance_embed = edges['continuous_distance_encoder'](data['delta_ss'][0])

        edge_attr_embed, edge_type_embed = None, None
        if data['edge_type'][0] is not None:
            if self.generate_edge_attr:
                edge_attr_embed = edges['edge_attr_embedding_layer'](data['edge_type'][0])
            edge_type_embed = edges['edge_type_embedding_layer'](data['edge_type'][0])

        if self.do_ci2traj and data['edge_index'][0] is not None:
            x_for_time_filter = stack['conv_for_time_filter'](
                x,
                edge_index=data['edge_index'][0],
                edge_attr_embed=edge_attr_embed,
                edge_time_embed=edge_time_embed,
                edge_dist_embed=edge_distance_embed,
                edge_type_embed=edge_type_embed
            )
            x_for_time_filter = stack['norms_for_time_filter'](x_for_time_filter)
            x_for_time_filter = self.act(x_for_time_filter)
            x_for_time_filter = stack['dropout_for_time_filter'](x_for_time_filter)
        else:
            # traj2traj-only: residual skip must match this batch's query count
            # (edge_index[0] dest dim), not configured eval_batch_size.
            if data['edge_index'][0] is not None:
                n_query = data['edge_index'][0].sparse_sizes()[0]
            elif label is not None:
                n_query = int(label.size(0))
            elif data.get('split_index') is not None:
                n_query = int(data['split_index']) + 1
            else:
                n_query = x.size(0)
            x_for_time_filter = torch.zeros(
                n_query, channel_size, device=x.device, dtype=x.dtype
            )

        if data['edge_index'][-1] is not None and self.do_traj2traj:
            for idx, (edge_index, edge_attr, delta_ts, delta_dis, edge_type) in enumerate(
                    zip(data["edge_index"][1:], data["edge_attr"][1:], data["delta_ts"][1:], data["delta_ss"][1:],
                        data["edge_type"][1:])
            ):
                if self.hypergraph_edges == 'traj2traj' and idx == 0:
                    x = x[: edge_index.sparse_sizes()[0]]
                    continue
                edge_time_embed = edges['continuous_time_encoder'](delta_ts / (60 * 60))
                if self.distance_encoder_type == 'stan':
                    edge_distance_embed = edges['continuous_distance_encoder'](
                        delta_dis, dist_type='tj2tj'
                    )
                else:
                    edge_distance_embed = edges['continuous_distance_encoder'](delta_dis)

                edge_attr_embed, edge_type_embed = None, None
                if edge_type is not None:
                    edge_type_embed = edges['edge_type_embedding_layer'](edge_type)
                    if self.generate_edge_attr:
                        edge_attr_embed = edges['edge_attr_embedding_layer'](edge_type)
                    else:
                        if edges['edge_attr_embedding_layer']:
                            edge_attr_embed = edges['edge_attr_embedding_layer'](
                                edge_attr.to(torch.float32)
                            )
                        else:
                            edge_attr_embed = edge_attr.to(torch.float32)

                n_dst = edge_index.sparse_sizes()[0]
                if idx == len(data['edge_index']) - 2:
                    x_target = x_for_time_filter[:n_dst]
                else:
                    x_target = x[:n_dst]

                x = stack['conv_list'][idx](
                    (x, x_target),
                    edge_index=edge_index,
                    edge_attr_embed=edge_attr_embed,
                    edge_time_embed=edge_time_embed,
                    edge_dist_embed=edge_distance_embed,
                    edge_type_embed=edge_type_embed
                )
                x = stack['norms_list'][idx](x)
                x = self.act(x)
                x = stack['dropout_list'][idx](x)
        else:
            x = x_for_time_filter
        return x

    def _encode_checkin_nodes(self, checkin_feature, data, label, mode, stack, channel_size, edge_prefix=''):
        """Wrap check-in features with traj zeros (or mean-pool when edges=none)."""
        split_idx = data['split_index']
        if self.hypergraph_edges == 'none':
            return self._hypergraph_encode(
                checkin_feature, data, label, mode, stack, channel_size, edge_prefix
            )
        traj_zeros = torch.zeros(
            split_idx + 1,
            channel_size,
            device=checkin_feature.device,
            dtype=checkin_feature.dtype,
        )
        x = torch.cat([traj_zeros, checkin_feature], dim=0)
        return self._hypergraph_encode(
            x, data, label, mode, stack, channel_size, edge_prefix
        )

    def forward(self, data, label=None, mode='train'):
        input_x = data['x']  # [?, 8]
        split_idx = data['split_index']
        check_in_x = input_x[split_idx + 1:]

        # Fused / paper path: exclude the pulled-out P/C/T slot (location not in concat).
        exclude = self.late_sequence if self.late_sequence in ('poi', 'category', 'time') else None
        checkin_feature = self.checkin_embedding_layer(check_in_x, exclude=exclude)
        main_out = self._encode_checkin_nodes(
            checkin_feature,
            data,
            label,
            mode,
            self._main_stack(),
            self.checkin_embed_size,
            edge_prefix='',
        )

        if self.late_sequence not in self.LATE_CHOICES:
            # Paper scorer: Linear only (no extra MLP).
            logits = self.linear(main_out)
            loss = self.loss_func(logits, label.long())
            return logits, loss

        # Pulled-out sequence on its own HG stack.
        pulled = self.checkin_embedding_layer.sequence_feature(check_in_x, self.late_sequence)
        if self.late_sequence == 'location':
            pulled = self.location_proj(pulled)
        late_out = self._encode_checkin_nodes(
            pulled,
            data,
            label,
            mode,
            self.late_stack,
            self.embed_size,
            edge_prefix='late_',
        )
        # Align rows if a partial batch somehow differs (should match).
        n = min(main_out.size(0), late_out.size(0))
        fused = torch.cat([main_out[:n], late_out[:n]], dim=-1)
        logits = self.late_mlp(fused)
        loss = self.loss_func(logits, label.long())
        return logits, loss
