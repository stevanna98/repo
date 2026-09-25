import numpy as np
import pytest
import torch
from normative_vae.models import VAE, GATLayer, GCNLayer
from normative_vae.attention import AttentionBlock
from normative_vae.losses import Objective


@pytest.mark.parametrize("architecture", ["set", "gat", "gcn"])
def test_models_379_forward_backward_eval(architecture):
    torch.manual_seed(4)
    model = VAE(architecture, heads=2, layers=1, dropout=.1)
    x = torch.randn(2, 379, 379)
    a = torch.zeros(2, 379, 379, dtype=torch.bool)
    a[:, 0, 1] = a[:, 1, 0] = True
    prediction, mu, logvar = model(x, a)
    assert prediction.shape == x.shape and mu.shape == logvar.shape == (2, 379, 64)
    objective = Objective()
    loss, terms = objective(x, prediction, mu, logvar)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert objective.alpha_logit.grad != 0 and objective.beta_logit.grad != 0
    model.eval()
    first = model(x, a)[0]
    assert torch.equal(first, model(x, a)[0])
    assert torch.equal(first, model.decode(model.encode(x, a)[0], a))
    if architecture != "set":
        assert torch.equal(model.decode(mu, a), model.decode(mu, ~a))


def test_loss_reductions_and_initialization():
    x = torch.tensor([[[1., .2], [.2, 1.]]])
    mu, logvar = torch.ones(1, 2, 3), torch.zeros(1, 2, 3)
    obj = Objective()
    _, terms = obj(x, x, mu, logvar)
    assert terms["mse"] == 0
    assert terms["pearson"].item() == pytest.approx(0, abs=1e-6)
    assert terms["kl"].item() == pytest.approx(1.5)  # Not divided by latent width.
    assert terms["alpha"].item() == pytest.approx(.8)
    assert terms["beta"].item() == pytest.approx(.1)
    x2 = x.clone(); x2[:, 0, 0] += 1
    _, terms = obj(x, x2, mu, logvar)
    assert terms["mse"].item() == pytest.approx(.25)  # Diagonal included.


def test_empty_attention_output_bias_is_zeroed():
    block = AttentionBlock(8, 2)
    block.output.bias.data.fill_(7)
    h, a = torch.randn(1, 3, 8), torch.zeros(1, 3, 3, dtype=torch.bool)
    assert block.contribution(h, a).eq(0).all()
    assert torch.isfinite(block(h, a)).all()


def test_original_gat_hand_computation_and_gcn_no_loops():
    gat = GATLayer(1, 1, 1, .2, bias=False)
    gat.transform.weight.data.fill_(1)
    gat.source.data.fill_(1); gat.target.data.fill_(1)
    x = torch.tensor([[[1.], [2.], [4.]]])
    a = torch.tensor([[[0, 1, 1], [1, 0, 0], [1, 0, 0]]], dtype=torch.bool)
    message = gat.message(x, a)
    expected = (2 * np.exp(3) + 4 * np.exp(5)) / (np.exp(3) + np.exp(5))
    assert message[0, 0, 0].item() == pytest.approx(expected)
    assert message[0, 1, 0] == 1
    gcn = GCNLayer(1, 1, bias=False)
    gcn.transform.weight.data.fill_(1)
    assert gcn(x, torch.zeros_like(a)).eq(0).all()
