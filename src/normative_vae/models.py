"""Native PyTorch SET-VAE, original additive GAT-VAE, and loop-free GCN-VAE."""
import torch
from torch import nn
from .attention import AttentionBlock
from .graphs import normalized_adjacency, safe_attention


def initialize(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class GATLayer(nn.Module):
    def __init__(self, input_dim, hidden, heads, slope, bias=True):
        super().__init__()
        self.heads, self.width, self.slope = heads, hidden // heads, slope
        self.transform = nn.Linear(input_dim, hidden, bias=False)
        self.source = nn.Parameter(torch.empty(heads, self.width))
        self.target = nn.Parameter(torch.empty(heads, self.width))
        self.bias = nn.Parameter(torch.zeros(hidden)) if bias else None
        nn.init.xavier_uniform_(self.source)
        nn.init.xavier_uniform_(self.target)

    def message(self, x, a):
        b, n, _ = x.shape
        h = self.transform(x).reshape(b, n, self.heads, self.width).transpose(1, 2)
        scores = torch.nn.functional.leaky_relu((h * self.source[None, :, None]).sum(-1).unsqueeze(-1) + (h * self.target[None, :, None]).sum(-1).unsqueeze(-2), self.slope)
        weights = safe_attention(scores, a[:, None])
        return (weights @ h).transpose(1, 2).reshape(b, n, -1)

    def forward(self, x, a):
        m = self.message(x, a)
        return m if self.bias is None else m + self.bias


class GCNLayer(nn.Module):
    def __init__(self, input_dim, hidden, bias=True):
        super().__init__()
        self.transform = nn.Linear(input_dim, hidden, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden)) if bias else None

    def forward(self, x, a):
        message = normalized_adjacency(a) @ self.transform(x)
        return message if self.bias is None else message + self.bias


class VAE(nn.Module):
    def __init__(self, architecture, n=379, hidden=256, latent=64, heads=2, layers=2, dropout=0., norm_eps=1e-5, gat_slope=0.2, bias=True):
        super().__init__()
        if architecture not in {"set", "gat", "gcn"}:
            raise ValueError(architecture)
        self.architecture = architecture
        self.spec = dict(architecture=architecture, n=n, hidden=hidden, latent=latent, heads=heads, layers=layers, dropout=dropout, norm_eps=norm_eps, gat_slope=gat_slope, bias=bias)
        if architecture == "set":
            self.input = nn.Linear(n, hidden, bias=bias)
            self.encoder = nn.ModuleList([AttentionBlock(hidden, heads, dropout, norm_eps, bias) for _ in range(2)])
            self.latent_input = nn.Linear(latent, hidden, bias=bias)
            self.decoder = nn.ModuleList([AttentionBlock(hidden, heads, dropout, norm_eps, bias) for _ in range(2)])
            self.output = nn.Sequential(nn.Linear(hidden, hidden, bias=bias), nn.Mish(), nn.Linear(hidden, n, bias=bias))
        else:
            self.encoder = nn.ModuleList([GATLayer(n if i == 0 else hidden, hidden, heads, gat_slope, bias) if architecture == "gat" else GCNLayer(n if i == 0 else hidden, hidden, bias) for i in range(layers)])
            self.post = nn.ModuleList([nn.Sequential(nn.LayerNorm(hidden, eps=norm_eps), nn.Mish(), nn.Dropout(dropout)) for _ in range(layers)])
            decoder = []
            for i in range(3):
                decoder.extend([nn.Linear(latent if i == 0 else hidden, hidden, bias=bias), nn.LayerNorm(hidden, eps=norm_eps), nn.Mish(), nn.Dropout(dropout)])
            decoder.append(nn.Linear(hidden, n, bias=bias))
            self.decoder = nn.Sequential(*decoder)
        self.mu = nn.Linear(hidden, latent, bias=bias)
        self.logvar = nn.Linear(hidden, latent, bias=bias)
        self.apply(initialize)

    def encode(self, x, a):
        if self.architecture == "set":
            h = self.encoder[1](self.encoder[0](self.input(x)), a)
        else:
            h = x
            for layer, post in zip(self.encoder, self.post):
                h = post(layer(h, a))
        return self.mu(h), self.logvar(h)

    def decode(self, z, a=None):
        if self.architecture == "set":
            return self.output(self.decoder[1](self.decoder[0](self.latent_input(z)), a))
        return self.decoder(z)

    def forward(self, x, a):
        mu, logvar = self.encode(x, a)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if self.training else mu
        return self.decode(z, a), mu, logvar


def make_model(cfg, architecture, candidate):
    return VAE(architecture, n=cfg["n_regions"], **cfg["model"], heads=candidate.get("heads", 1), layers=candidate.get("layers", 2), dropout=candidate["dropout"])

