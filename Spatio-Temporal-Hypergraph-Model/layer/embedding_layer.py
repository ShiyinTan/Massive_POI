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

    def sequence_feature(self, data, sequence):
        """Single-sequence embedding for late pull-out.

        ``sequence`` in ``{poi, category, time, location}``.
        time = weekday+hour (sum); location returns raw lon/lat ``[..., 2]``.
        """
        if sequence == 'poi':
            return self.poi_embedding(data[..., 1].long())
        if sequence == 'category':
            return self.category_embedding(data[..., 2].long())
        if sequence == 'time':
            return self.dayofweek_embedding(data[..., 6].long()) + self.hourofday_embedding(
                data[..., 7].long()
            )
        if sequence == 'location':
            poi_ref = self.poi_embedding(data[..., 1].long())
            return data[..., 4:6].to(dtype=poi_ref.dtype)
        raise ValueError(f"Unknown late sequence {sequence}")

    def forward(self, data, exclude=None):
        """Paper concat of user+poi+category+weekday+hour.

        When ``exclude`` is ``poi`` / ``category`` / ``time``, that sequence's
        slot(s) are zeroed so the remaining modalities keep the same concat
        layout and encoder width as the paper. ``location`` is not in the
        paper concat (caller handles it on a separate path).
        """
        user_emb = self.user_embedding(data[..., 0].long())
        if not self.use_user_embed:
            user_emb = torch.zeros_like(user_emb)
        poi_emb = self.poi_embedding(data[..., 1].long())
        cat_emb = self.category_embedding(data[..., 2].long())
        dow_emb = self.dayofweek_embedding(data[..., 6].long())
        hour_emb = self.hourofday_embedding(data[..., 7].long())

        if exclude == 'poi':
            poi_emb = torch.zeros_like(poi_emb)
        elif exclude == 'category':
            cat_emb = torch.zeros_like(cat_emb)
        elif exclude == 'time':
            dow_emb = torch.zeros_like(dow_emb)
            hour_emb = torch.zeros_like(hour_emb)
        elif exclude not in (None, 'none', 'location'):
            raise ValueError(f"Unsupported exclude={exclude}")

        embedding_list = [user_emb, poi_emb, cat_emb, dow_emb, hour_emb]
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
