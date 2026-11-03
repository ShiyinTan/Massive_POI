from dataclasses import dataclass
import torch.nn as nn
from collections import OrderedDict
import torch
from torch.nn import functional as F
import numpy as np
import inspect
from location_tower_model import Location_Tower
from torch.nn import MultiheadAttention, Transformer


def prob(x, vocab_embedding):
    x = x.unsqueeze(2)  # [B, T, 1, 512]
    vocab_embedding = vocab_embedding.unsqueeze(0).unsqueeze(0)  # [1, 1, loc_num, 512]
    cosine_similarity = torch.matmul(x, vocab_embedding.transpose(-1, -2))  # [B, T, 1, loc_num]
    cosine_similarity = cosine_similarity.squeeze(2)  # [B, T, loc_num]
    return cosine_similarity


class Expert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class NoisyTopkRouter(nn.Module):
    # def __init__(self, n_embed, num_experts, top_k):
    def __init__(self, config):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = config.top_k
        self.topkroute_linear = nn.Linear(config.n_embd, config.num_experts)
        # add noise
        self.noise_linear = nn.Linear(config.n_embd, config.num_experts)

    def forward(self, mh_output):
        # mh_ouput is the output tensor from multihead self attention block
        logits = self.topkroute_linear(mh_output)

        # Noise logits
        noise_logits = self.noise_linear(mh_output)

        # Adding scaled unit gaussian noise to the logits
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise
        gate1 = F.softmax(noisy_logits, dim=-1)
        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float("-inf"))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices, gate1


class SparseMoE(nn.Module):
    def __init__(self, config):
        super(SparseMoE, self).__init__()
        self.router = NoisyTopkRouter(config)
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.num_experts)])
        self.top_k = config.top_k

    def forward(self, x):
        gating_output, indices, gate1 = self.router(x)
        final_output = torch.zeros_like(x)
        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)
            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores
                final_output[expert_mask] += weighted_output.squeeze(1)

        return final_output, gate1


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiheadAttention(num_heads=config.n_head, embed_dim=config.n_embd, batch_first=True)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.smoe = SparseMoE(config)

    def forward(self, x, attn_mask, pad_mask):
        y = self.ln_1(x)
        attn_output, attn_weights = self.attn(y, y, y, pad_mask, True, attn_mask, True, True)
        x = x + attn_output
        x_, gate_out = self.smoe(self.ln_2(x))
        x = x + x_

        return x


@dataclass
class Traj_Config:
    block_size: int = 48 * 3  # max seq_len
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512  # embedding dim
    num_experts: int = 8
    top_k: int = 2


class Traj_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                time_embedding=nn.Embedding(48, config.n_embd),
                lon_lat_embedding=nn.Linear(2, config.n_embd // 2),
                poi_feature_embedding=nn.Linear(324, config.n_embd // 4),
                flow_rank_embedding=nn.Embedding(9, config.n_embd // 4),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.vocab_embd = Location_Tower(config)
        self.lm_head = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # init params
        self.apply(self._init_weights)  # iterate all submodule and apply init_modules

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, his, ts, targets, vocab, device):
        # idx is of shape (B, T), T is time dimension
        # poi (B, T,25)
        loc_feature = np.take(vocab, his, axis=0)
        his, targets, loc_feature, ts = his.to(device), targets.to(device), loc_feature.to(device), ts.to(device)
        B, T = his.size()
        padding_mask = (his == 0).to(torch.bool)
        ts = ts.to(torch.long)
        poi_feature = loc_feature[:, :, :324]
        lon_lat = loc_feature[:, :, 324 : 324 + 2]
        rank = loc_feature[:, :, -1].to(torch.long)
        vocab = vocab.to(device)
        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=device)  # shape (T)
        pos_emb = self.transformer.wpe(pos)
        poi_feature_emb = self.transformer.poi_feature_embedding(poi_feature)
        lon_lat_emb = self.transformer.lon_lat_embedding(lon_lat)
        rank_emb = self.transformer.flow_rank_embedding(rank)
        token_emb = torch.cat((lon_lat_emb, rank_emb, poi_feature_emb), dim=-1)
        ts_emb = self.transformer.time_embedding(ts)  # B T 16*3
        x = token_emb + ts_emb + pos_emb

        mask = Transformer.generate_square_subsequent_mask(T, device=device).to(torch.bool)
        for block in self.transformer.h:
            x = block(x, mask, padding_mask)

        x = self.transformer.ln_f(x)
        x = self.lm_head(x)
        loc_embedding = self.vocab_embd(vocab)

        logits = prob(x, loc_embedding)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)

        output = OrderedDict()

        output["logits"] = logits
        output["loss"] = loss

        return output

    def configure_optimizers(self, weight_decay, learning_rate):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} params")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} params")

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        if torch.cuda.is_available():
            device = "cuda"
        use_fused = fused_available and device == "cuda"  ## 8. fuse the adamw
        print(f"using fused AdamW: {use_fused}")

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

        return optimizer
