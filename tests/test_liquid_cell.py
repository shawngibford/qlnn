import pytest
import torch

from quantum_liquid_neuralode.models import LiquidCell


def test_liquid_cell_output_shape_and_finite():
    torch.manual_seed(0)

    cell = LiquidCell(input_size=5, hidden_size=7, tau_min=0.1)

    h = torch.randn(3, 7)
    x = torch.randn(3, 5)

    dh = cell(h, x)

    assert dh.shape == (3, 7)
    assert torch.isfinite(dh).all()


def test_liquid_cell_tau_is_positive():
    torch.manual_seed(0)

    cell = LiquidCell(input_size=2, hidden_size=4, tau_min=0.123)
    tau = cell.tau()

    assert tau.shape == (4,)
    assert (tau > 0).all()
    assert tau.min().detach().item() >= 0.123


def test_liquid_cell_gradients_flow():
    torch.manual_seed(0)

    cell = LiquidCell(input_size=3, hidden_size=3)

    h = torch.randn(2, 3, requires_grad=True)
    x = torch.randn(2, 3)

    loss = cell(h, x).sum()
    loss.backward()

    assert h.grad is not None
    assert cell.tau_unconstrained.grad is not None
    assert cell.W_h.weight.grad is not None
    assert cell.W_x.weight.grad is not None


def test_liquid_cell_raises_on_bad_shapes():
    cell = LiquidCell(input_size=2, hidden_size=3)

    with pytest.raises(ValueError):
        cell(torch.randn(3), torch.randn(1, 2))

    with pytest.raises(ValueError):
        cell(torch.randn(1, 3), torch.randn(2, 2))

    with pytest.raises(ValueError):
        cell(torch.randn(1, 4), torch.randn(1, 2))

    with pytest.raises(ValueError):
        cell(torch.randn(1, 3), torch.randn(1, 5))
