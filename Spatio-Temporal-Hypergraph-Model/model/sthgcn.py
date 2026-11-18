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
    def __init__(self, cfg):
        super(STHGCN, self).__init__()
        self.device = cfg.run_args.device
        self.batch_size = cfg.run_args.batch_size
        self.eval_batch_size = cfg.run_args.eval_batch_size
        # Ablation: hypergraph_edges in {both, ci2traj, traj2traj, none}; default both.
        self.hypergraph_edges = getattr(cfg.model_args, 'hypergraph_edges', 'both')
        self.do_traj2traj = bool(cfg.model_args.do_traj2traj) and self.hypergraph_edges in ('both', 'traj2traj')
        self.do_ci2traj = self.hypergraph_edges in ('both', 'ci2traj')
        # Ablation: modality_fusion in {early, late}; default early = paper CheckinEmbedding concat.
        self.modality_fusion = getattr(cfg.model_args, 'modality_fusion', 'early')
        if isinstance(self.modality_fusion, str):
            self.modality_fusion = self.modality_fusion.lower()
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

        self.cfg_conv_args = cfg.conv_args
        self.cfg_model_args = cfg.model_args
        self.cfg_dataset_args = cfg.dataset_args

        if self.modality_fusion == 'late':
            # Late fusion: P/C/T/L each use embed_size channels through own hypergraph stacks.
            self.checkin_embed_size = self.embed_size
            continuous_encoder_dim = (
                self.checkin_embed_size
                if cfg.conv_args.time_fusion_mode == 'add'
                else cfg.model_args.st_embed_size
            )
            self.location_proj = nn.Linear(2, self.embed_size)
            self.late_streams = nn.ModuleDict({
                name: self._build_hg_stack(self.embed_size, continuous_encoder_dim, cfg)
                for name in ('P', 'C', 'T', 'L')
            })
            self._init_edge_modules(self.embed_size, continuous_encoder_dim, cfg)
            late_in = 4 * self.embed_size + self.embed_size  # P,C,T,L + user
            self.late_mlp = nn.Sequential(
                nn.Linear(late_in, late_in),
                nn.ReLU(),
                nn.Linear(late_in, self.num_poi),
            )
            self.linear = None  # unused on late path
        else:
            # Paper / early fusion: concat modalities in CheckinEmbedding, then one hypergraph stack.
            self.checkin_embed_size = self.checkin_embedding_layer.output_embed_size
            continuous_encoder_dim = (
                self.checkin_embed_size
                if cfg.conv_args.time_fusion_mode == 'add'
                else cfg.model_args.st_embed_size
            )
            early_stack = self._build_hg_stack(self.checkin_embed_size, continuous_encoder_dim, cfg)
            self.conv_for_time_filter = early_stack['conv_for_time_filter']
            self.norms_for_time_filter = early_stack['norms_for_time_filter']
            self.dropout_for_time_filter = early_stack['dropout_for_time_filter']
            self.conv_list = early_stack['conv_list']
            self.norms_list = early_stack['norms_list']
            self.dropout_list = early_stack['dropout_list']
            self._init_edge_modules(self.checkin_embed_size, continuous_encoder_dim, cfg)
            self.linear = nn.Linear(self.checkin_embed_size, self.num_poi)

        self.loss_func = nn.CrossEntropyLoss()

    def _init_edge_modules(self, channel_size, continuous_encoder_dim, cfg):
        self.edge_type_embedding_layer = EdgeEmbedding(
            embed_size=channel_size,
            fusion_type=self.embed_fusion_type,
            num_edge_type=cfg.model_args.num_edge_type
        )
        if self.generate_edge_attr:
            self.edge_attr_embedding_layer = EdgeEmbedding(
                embed_size=channel_size,
                fusion_type=self.embed_fusion_type,
                num_edge_type=cfg.model_args.num_edge_type
            )
        else:
            if cfg.conv_args.edge_fusion_mode == 'add':
                self.edge_attr_embedding_layer = nn.Linear(3, channel_size)
            else:
                self.edge_attr_embedding_layer = None

        self.continuous_time_encoder = TimeEncoder(cfg.model_args, continuous_encoder_dim)
        if self.distance_encoder_type == 'stan':
            self.continuous_distance_encoder = DistanceEncoderSTAN(
                cfg.model_args, continuous_encoder_dim, cfg.dataset_args.spatial_slots
            )
        elif self.distance_encoder_type == 'time':
            self.continuous_distance_encoder = TimeEncoder(cfg.model_args, continuous_encoder_dim)
        elif self.distance_encoder_type == 'hstlstm':
            self.continuous_distance_encoder = DistanceEncoderHSTLSTM(
                cfg.model_args, continuous_encoder_dim, cfg.dataset_args.spatial_slots
            )
        elif self.distance_encoder_type == 'simple':
            self.continuous_distance_encoder = DistanceEncoderSimple(
                cfg.model_args, continuous_encoder_dim, cfg.dataset_args.spatial_slots
            )
        else:
            raise ValueError(
                f"Get wrong distance_encoder_type argument: {cfg.model_args.distance_encoder_type}!"
            )

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

    def _batch_size(self, label, mode):
        if label is not None:
            return int(label.size(0))
        if mode in ('test', 'validate'):
            return self.eval_batch_size
        return self.batch_size

    def _hypergraph_encode(self, x, data, label, mode, stack, channel_size):
        """Run one modality (or early fused) feature tensor through a hypergraph stack."""
        if self.hypergraph_edges == 'none':
            batch_size = self._batch_size(label, mode)
            # Late none-path passes check-in-only features [N_ci, D].
            if x.size(0) > 0:
                return x.mean(dim=0, keepdim=True).expand(batch_size, -1)
            return torch.zeros(batch_size, channel_size, device=x.device, dtype=x.dtype)

        edge_time_embed = self.continuous_time_encoder(data['delta_ts'][0] / (60 * 60))
        if self.distance_encoder_type == 'stan':
            edge_distance_embed = self.continuous_distance_encoder(data['delta_ss'][0], dist_type='ch2tj')
        else:
            edge_distance_embed = self.continuous_distance_encoder(data['delta_ss'][0])

        edge_attr_embed, edge_type_embed = None, None
        if data['edge_type'][0] is not None:
            if self.generate_edge_attr:
                edge_attr_embed = self.edge_attr_embedding_layer(data['edge_type'][0])
            edge_type_embed = self.edge_type_embedding_layer(data['edge_type'][0])

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
                edge_time_embed = self.continuous_time_encoder(delta_ts / (60 * 60))
                if self.distance_encoder_type == 'stan':
                    edge_distance_embed = self.continuous_distance_encoder(delta_dis, dist_type='tj2tj')
                else:
                    edge_distance_embed = self.continuous_distance_encoder(delta_dis)

                edge_attr_embed, edge_type_embed = None, None
                if edge_type is not None:
                    edge_type_embed = self.edge_type_embedding_layer(edge_type)
                    if self.generate_edge_attr:
                        edge_attr_embed = self.edge_attr_embedding_layer(edge_type)
                    else:
                        if self.edge_attr_embedding_layer:
                            edge_attr_embed = self.edge_attr_embedding_layer(edge_attr.to(torch.float32))
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

    def _aggregate_user_to_queries(self, user_ci, data, n_query, split_idx):
        """Mean-pool check-in user embeds onto query trajs via time-filtered ci2traj edges."""
        device = user_ci.device
        dtype = user_ci.dtype
        if data.get('edge_index') is None or data['edge_index'][0] is None:
            if user_ci.size(0) == 0:
                return torch.zeros(n_query, user_ci.size(-1), device=device, dtype=dtype)
            return user_ci.mean(dim=0, keepdim=True).expand(n_query, -1)

        st = data['edge_index'][0]
        row = st.storage.row()
        col = st.storage.col()
        # col indexes into full node feature space (traj | checkin); map to check-in rows.
        checkin_idx = col - (split_idx + 1)
        valid = (checkin_idx >= 0) & (checkin_idx < user_ci.size(0)) & (row < n_query)
        if not valid.any():
            if user_ci.size(0) == 0:
                return torch.zeros(n_query, user_ci.size(-1), device=device, dtype=dtype)
            return user_ci.mean(dim=0, keepdim=True).expand(n_query, -1)

        row_v = row[valid]
        src = user_ci[checkin_idx[valid]]
        out = torch.zeros(n_query, user_ci.size(-1), device=device, dtype=dtype)
        counts = torch.zeros(n_query, 1, device=device, dtype=dtype)
        out.index_add_(0, row_v, src)
        counts.index_add_(0, row_v, torch.ones(row_v.size(0), 1, device=device, dtype=dtype))
        out = out / counts.clamp(min=1.0)
        empty = (counts.squeeze(-1) == 0)
        if empty.any() and user_ci.size(0) > 0:
            out[empty] = user_ci.mean(dim=0)
        return out

    def forward(self, data, label=None, mode='train'):
        if self.modality_fusion == 'late':
            return self._forward_late(data, label=label, mode=mode)
        return self._forward_early(data, label=label, mode=mode)

    def _forward_early(self, data, label=None, mode='train'):
        input_x = data['x']  # [?, 8]
        split_idx = data['split_index']

        check_in_x = input_x[split_idx+1:]
        checkin_feature = self.checkin_embedding_layer(check_in_x)
        trajectory_feature = torch.zeros(
            split_idx+1,
            self.checkin_embed_size,
            device=checkin_feature.device
        )
        x = torch.cat([trajectory_feature, checkin_feature], dim=0)

        if self.hypergraph_edges == 'none':
            batch_size = self._batch_size(label, mode)
            if checkin_feature.size(0) > 0:
                pooled = checkin_feature.mean(dim=0, keepdim=True).expand(
                    min(batch_size, x.size(0)), -1
                )
                x = pooled
            else:
                x = x[:batch_size]
            logits = self.linear(x)
            loss = self.loss_func(logits, label.long())
            return logits, loss

        stack = {
            'conv_for_time_filter': self.conv_for_time_filter,
            'norms_for_time_filter': self.norms_for_time_filter,
            'dropout_for_time_filter': self.dropout_for_time_filter,
            'conv_list': self.conv_list,
            'norms_list': self.norms_list,
            'dropout_list': self.dropout_list,
        }
        x = self._hypergraph_encode(
            x, data, label, mode, stack, self.checkin_embed_size
        )
        logits = self.linear(x)
        loss = self.loss_func(logits, label.long())
        return logits, loss

    def _forward_late(self, data, label=None, mode='train'):
        """P/C/T/L each through own hypergraph stack; concat (+user) → MLP → POI logits."""
        input_x = data['x']
        split_idx = data['split_index']
        check_in_x = input_x[split_idx+1:]
        mods = self.checkin_embedding_layer.forward_modalities(check_in_x)
        loc_emb = self.location_proj(mods['L_raw'])

        modality_feats = {
            'P': mods['P'],
            'C': mods['C'],
            'T': mods['T'],
            'L': loc_emb,
        }

        encoded = {}
        for name, feat in modality_feats.items():
            if self.hypergraph_edges == 'none':
                encoded[name] = self._hypergraph_encode(
                    feat, data, label, mode, self.late_streams[name], self.embed_size
                )
            else:
                traj_zeros = torch.zeros(
                    split_idx + 1, self.embed_size, device=feat.device, dtype=feat.dtype
                )
                x = torch.cat([traj_zeros, feat], dim=0)
                encoded[name] = self._hypergraph_encode(
                    x, data, label, mode, self.late_streams[name], self.embed_size
                )

        # Align batch rows across streams (should already match).
        n_query = encoded['P'].size(0)
        user_q = self._aggregate_user_to_queries(
            mods['user'], data, n_query, split_idx
        )
        fused = torch.cat(
            [encoded['P'], encoded['C'], encoded['T'], encoded['L'], user_q], dim=-1
        )
        logits = self.late_mlp(fused)
        loss = self.loss_func(logits, label.long())
        return logits, loss
