import torch
from torch import nn


class CheckinEmbedding(nn.Module):
    def __init__(
        self,
        embed_size,
        fusion_type,
        dataset_args,
        use_user_embed=True,
    ):
        super(CheckinEmbedding, self).__init__()
        self.embed_size = embed_size
        self.fusion_type = fusion_type
        self.use_user_embed = use_user_embed
        self.user_embedding = nn.Embedding(
            dataset_args.num_user + 1,
            self.embed_size,
            padding_idx=dataset_args.padding_user_id
        )
        self.poi_embedding = nn.Embedding(
            dataset_args.num_poi + 1,
            self.embed_size,
            padding_idx=dataset_args.padding_poi_id
        )
        self.category_embedding = nn.Embedding(
            dataset_args.num_category + 1,
            self.embed_size,
            padding_idx=dataset_args.padding_poi_category
        )
        self.dayofweek_embedding = nn.Embedding(8, self.embed_size, padding_idx=dataset_args.padding_weekday_id)
        self.hourofday_embedding = nn.Embedding(25, self.embed_size, padding_idx=dataset_args.padding_hour_id)
        if self.fusion_type == 'concat':
            self.output_embed_size = 5 * self.embed_size
        elif self.fusion_type == 'add':
            self.output_embed_size = embed_size
        else:
            raise ValueError(f"Get wrong fusion type {self.fusion_type}")

    def forward_modalities(self, data):
        """Return per-modality embeddings (each ``embed_size``) for late fusion.

        P=POI, C=category, T=weekday+hour (sum), L=raw lon/lat (caller projects),
        user=user lookup (may be zeroed when ``use_user_embed`` is False).
        """
        user_emb = self.user_embedding(data[..., 0].long())
        if not self.use_user_embed:
            user_emb = torch.zeros_like(user_emb)
        poi_emb = self.poi_embedding(data[..., 1].long())
        cat_emb = self.category_embedding(data[..., 2].long())
        time_emb = self.dayofweek_embedding(data[..., 6].long()) + self.hourofday_embedding(
            data[..., 7].long()
        )
        loc_raw = data[..., 4:6].to(dtype=poi_emb.dtype)
        return {
            "user": user_emb,
            "P": poi_emb,
            "C": cat_emb,
            "T": time_emb,
            "L_raw": loc_raw,
        }

    def forward(self, data):
        user_emb = self.user_embedding(data[..., 0].long())
        if not self.use_user_embed:
            user_emb = torch.zeros_like(user_emb)
        embedding_list = [
            user_emb,
            self.poi_embedding(data[..., 1].long()),
            self.category_embedding(data[..., 2].long()),
            self.dayofweek_embedding(data[..., 6].long()),
            self.hourofday_embedding(data[..., 7].long())
        ]
        if self.fusion_type == 'concat':
            self.output_embed_size = len(embedding_list) * self.embed_size
            return torch.cat(embedding_list, -1)
        elif self.fusion_type == 'add':
            return torch.squeeze(sum(embedding_list))
        else:
            raise ValueError(f"Get wrong fusion type {self.fusion_type}")


class EdgeEmbedding(torch.nn.Module):
    def __init__(self, embed_size, fusion_type, num_edge_type):
        super(EdgeEmbedding, self).__init__()
        self.embed_size = embed_size
        self.fusion_type = fusion_type
        self.edge_type_embedding = nn.Embedding(num_edge_type, self.embed_size)
        self.output_embed_size = self.embed_size

    def forward(self, data):
        embedding_list = [self.edge_type_embedding(data.long())]

        if self.fusion_type == 'concat':
            self.output_embed_size = len(embedding_list) * self.embed_size
            return torch.cat(embedding_list, -1)
        elif self.fusion_type == 'add':
            return sum(embedding_list)
        else:
            raise ValueError(f"Get wrong fusion type {self.fusion_type}")
