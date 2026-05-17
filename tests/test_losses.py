import pytest
import torch

from quantum_liquid_neuralode.training import logistic_growth_residual_loss, smoothness_loss


def test_logistic_growth_residual_loss_is_zero_for_all_zero_od():
    time_points = torch.linspace(0.0, 1.0, steps=6)
    od = torch.zeros(2, 6)

    loss = logistic_growth_residual_loss(od, time_points, mu=0.3, K=3.8)

    assert loss.item() == 0.0


def test_smoothness_loss_is_zero_for_linear_sequence():
    seq = torch.arange(0.0, 10.0)

    loss = smoothness_loss(seq)

    assert loss.item() == 0.0


def test_logistic_growth_residual_loss_raises_for_non_increasing_time():
    time_points = torch.tensor([0.0, 1.0, 1.0, 2.0])
    od = torch.zeros(4)

    with pytest.raises(ValueError):
        logistic_growth_residual_loss(od, time_points, mu=0.3, K=3.8)
