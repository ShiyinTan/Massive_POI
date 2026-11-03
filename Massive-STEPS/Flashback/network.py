import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from enum import Enum


class Rnn(Enum):
    """The available RNN units"""

    RNN = 0
    GRU = 1
    LSTM = 2

    @staticmethod
    def from_string(name):
        if name == "rnn":
            return Rnn.RNN
        if name == "gru":
            return Rnn.GRU
        if name == "lstm":
            return Rnn.LSTM
        raise ValueError("{} not supported in --rnn".format(name))


class RnnFactory:
    """Creates the desired RNN unit."""

    def __init__(self, rnn_type_str):
        self.rnn_type = Rnn.from_string(rnn_type_str)

    def __str__(self):
        if self.rnn_type == Rnn.RNN:
            return "Use pytorch RNN implementation."
        if self.rnn_type == Rnn.GRU:
            return "Use pytorch GRU implementation."
        if self.rnn_type == Rnn.LSTM:
            return "Use pytorch LSTM implementation."

    def is_lstm(self):
        return self.rnn_type in [Rnn.LSTM]

    def create(self, hidden_size):
        if self.rnn_type == Rnn.RNN:
            return nn.RNN(hidden_size, hidden_size)
        if self.rnn_type == Rnn.GRU:
            return nn.GRU(hidden_size, hidden_size)
        if self.rnn_type == Rnn.LSTM:
            return nn.LSTM(hidden_size, hidden_size)


class Flashback(nn.Module):
    """Flashback RNN: Applies weighted average using spatial and tempoarl data in combination
    of user embeddings to the output of a generic RNN unit (RNN, GRU, LSTM).
    """

    def __init__(self, input_size, user_count, hidden_size, f_t, f_s, rnn_factory, padding_idx):
        super().__init__()
        self.input_size = input_size
        self.user_count = user_count
        self.hidden_size = hidden_size
        self.f_t = f_t  # function for computing temporal weight
        self.f_s = f_s  # function for computing spatial weight

        self.encoder = nn.Embedding(input_size + 1, hidden_size, padding_idx=padding_idx)  # location embedding
        self.user_encoder = nn.Embedding(user_count, hidden_size)  # user embedding
        self.rnn = rnn_factory.create(hidden_size)
        self.fc = nn.Linear(2 * hidden_size, input_size)  # create outputs in lenght of locations

    def forward(self, x, t, s, lengths, h, active_user):
        # 1. Ensure active_user is 1D: [user_len]
        if active_user.dim() > 1:
            active_user = active_user.view(-1)

        seq_len, user_len = x.size()
        x_emb = self.encoder(x)

        packed_x = pack_padded_sequence(x_emb, lengths.cpu(), enforce_sorted=False)
        packed_out, h = self.rnn(packed_x, h)
        out, _ = pad_packed_sequence(packed_out)  # out: [max_seq_len, user_len, hidden_size]

        # 2. Vectorized Pairwise Differences
        # t is [seq_len, user_len], we want [seq_len, seq_len, user_len]
        # t.unsqueeze(0) is [1, seq_len, user_len]
        # t.unsqueeze(1) is [seq_len, 1, user_len]
        dist_t = t.unsqueeze(0) - t.unsqueeze(1)
        dist_t = torch.clamp(dist_t, min=0)  # Flashback only cares about past to current

        # s is [seq_len, user_len, 2]
        s_diff = s.unsqueeze(0) - s.unsqueeze(1)  # [seq_len, seq_len, user_len, 2]
        dist_s = torch.norm(s_diff, dim=-1)  # [seq_len, seq_len, user_len]

        # 3. Compute Weights
        a_j = self.f_t(dist_t, user_len)
        b_j = self.f_s(dist_s, user_len)
        weights = a_j * b_j + 1e-10

        # Mask future: tril for [seq_len, seq_len], then unsqueeze for user_len
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(-1)
        len_range = torch.arange(seq_len, device=x.device).view(-1, 1, 1)
        len_mask = (len_range < lengths.view(1, 1, -1)).float()
        weights = weights * mask * len_mask

        # 4. Weighted Average using Batch Matrix Multiplication (BMM)
        # BMM expects [B, N, M] x [B, M, P]
        # We treat user_len as the Batch dimension
        weights = weights.permute(2, 0, 1)  # [user_len, seq_len, seq_len]
        out_trans = out.transpose(0, 1)  # [user_len, seq_len, hidden_size]

        sum_w = torch.sum(weights, dim=-1, keepdim=True)
        out_w = torch.bmm(weights, out_trans) / (sum_w + 1e-10)  # [user_len, seq_len, hidden_size]
        out_w = out_w.transpose(0, 1)  # [seq_len, user_len, hidden_size]

        # 5. User Embedding (Fixing the .expand error)
        p_u = self.user_encoder(active_user)  # Result is [user_len, hidden_size]
        p_u = p_u.unsqueeze(0).expand(seq_len, -1, -1)  # [seq_len, user_len, hidden_size]

        # Combine and Linear
        out_pu = torch.cat([out_w, p_u], dim=-1)
        y_linear = self.fc(out_pu)

        return y_linear, h


"""
~~~ h_0 strategies ~~~
Initialize RNNs hidden states
"""


def create_h0_strategy(hidden_size, is_lstm):
    if is_lstm:
        return LstmStrategy(hidden_size, FixNoiseStrategy(hidden_size), FixNoiseStrategy(hidden_size))
    else:
        return FixNoiseStrategy(hidden_size)


class H0Strategy:

    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def on_init(self, user_len, device):
        pass

    def on_reset(self, user):
        pass

    def on_reset_test(self, user, device):
        return self.on_reset(user)


class FixNoiseStrategy(H0Strategy):
    """use fixed normal noise as initialization"""

    def __init__(self, hidden_size):
        super().__init__(hidden_size)
        mu = 0
        sd = 1 / self.hidden_size
        self.h0 = torch.randn(self.hidden_size, requires_grad=False) * sd + mu

    def on_init(self, user_len, device):
        hs = []
        for i in range(user_len):
            hs.append(self.h0)
        return torch.stack(hs, dim=0).view(1, user_len, self.hidden_size).to(device)

    def on_reset(self, user):
        return self.h0


class LstmStrategy(H0Strategy):
    """creates h0 and c0 using the inner strategy"""

    def __init__(self, hidden_size, h_strategy, c_strategy):
        super(LstmStrategy, self).__init__(hidden_size)
        self.h_strategy = h_strategy
        self.c_strategy = c_strategy

    def on_init(self, user_len, device):
        h = self.h_strategy.on_init(user_len, device)
        c = self.c_strategy.on_init(user_len, device)
        return (h, c)

    def on_reset(self, user):
        h = self.h_strategy.on_reset(user)
        c = self.c_strategy.on_reset(user)
        return (h, c)
