"""Scratch implimentation of a Gated Recurrent Unit model. Educational - not suitable for real use."""

import torch
from torch import nn


class SyntheticData(nn.Module):
    def __init__(self, batch_size, sequence_len, n_inp):
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.n_inp = n_inp

    def generate_X(self):
        return torch.randn(self.batch_size, self.sequence_len, self.n_inp)

    def generate_y(self, n_hidden):
        return torch.randn(self.batch_size, n_hidden)


class GRU(nn.Module):
    def __init__(self, n_inp, n_hidden, sigma=0.01, device="cpu"):
        """You can follow the description of GRU at:
        <https://d2l.ai/chapter_recurrent-modern/gru.html>

        Args:
            n_imp: Number of Inputs
            n_hidden: Number of Hidden Units
            sigma: Parameter used in initialization to skew randomness
            device: Chipset for computation
        """
        super().__init__()
        self.n_hidden = n_hidden
        self.device = device
        init_weights = lambda *shape: nn.Parameter(torch.randn(*shape) * sigma)
        triple = lambda: (
            init_weights(n_inp, n_hidden),
            init_weights(n_hidden, n_hidden),
            nn.Parameter(torch.zeros(n_hidden)),
        )
        # Update Gate (Z)
        self.W_xz, self.W_hz, self.b_z = triple()
        # Reset Gate (R)
        self.W_xr, self.W_hr, self.b_r = triple()
        # Candidate Hidden State (H)
        self.W_xh, self.W_hh, self.b_h = triple()

    def forward(self, inputs, H=None):
        if H is None:
            H = torch.zeros((inputs.shape[1], self.n_hidden), device=self.device)

        output = []

        for X in inputs:
            # Update Gate
            Z = torch.sigmoid(
                torch.matmul(X, self.W_xz) + torch.matmul(H, self.W_hz) + self.b_z
            )
            # Reset Gate
            R = torch.sigmoid(
                torch.matmul(X, self.W_xr) + torch.matmul(H, self.W_hr) + self.b_r
            )
            # Candidate State
            Ht = torch.tanh(
                torch.matmul(X, self.W_xh) + torch.matmul(R * H, self.W_hh) + self.b_h
            )
            H = Z * H * (1 - Z) * Ht
            output.append(H)

        return output, H


# Test run
net = GRU(n_inp=1000, n_hidden=16)
data = SyntheticData(batch_size=8, sequence_len=10, n_inp=1000)

X = data.generate_X()
y = data.generate_y(n_hidden=16)

net(X)
print("Complete!")
