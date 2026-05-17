import pytest
import torch

from quantum_liquid_neuralode.models import LiquidODForecaster


def _toy_inputs(batch=2, T=8, F=4):
    torch.manual_seed(0)
    x = torch.randn(batch, T, F)
    t = torch.linspace(0.0, 1.4, T).unsqueeze(0).expand(batch, -1).contiguous()
    return x, t


@pytest.mark.parametrize("method", ["euler", "rk4", "dopri5"])
def test_forecaster_forward_shape_and_finite(method):
    x, t = _toy_inputs()
    m = LiquidODForecaster(
        input_size=x.shape[-1], hidden_size=8,
        horizon_hours=1.0, forecast_steps=2,
        od_index=0, delta_scale=0.1, ode_method=method,
    )
    y = m(x, t)
    assert y.shape == (x.shape[0],)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("method", ["euler", "rk4", "dopri5"])
def test_forecaster_gradients_flow(method):
    x, t = _toy_inputs()
    m = LiquidODForecaster(
        input_size=x.shape[-1], hidden_size=8,
        horizon_hours=1.0, forecast_steps=2,
        od_index=0, delta_scale=0.1, ode_method=method,
    )
    y = m(x, t)
    loss = (y ** 2).mean()
    loss.backward()
    # At least one parameter should have a non-zero gradient.
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads, "no gradients flowed"
    total_norm = sum(float(g.abs().sum()) for g in grads)
    assert total_norm > 0.0


def test_forecaster_residual_anchors_to_od_last():
    # With weights initialized roughly zero, the prediction should sit close to
    # od_last (delta is bounded by tanh * delta_scale).
    x, t = _toy_inputs()
    m = LiquidODForecaster(
        input_size=x.shape[-1], hidden_size=8,
        horizon_hours=1.0, forecast_steps=1,
        od_index=0, delta_scale=0.01, ode_method="euler",
    )
    od_last = x[:, -1, 0]
    y = m(x, t).detach()
    assert torch.all((y - od_last).abs() <= 0.011)


def test_forecaster_rejects_bad_shapes():
    m = LiquidODForecaster(
        input_size=3, hidden_size=4, horizon_hours=1.0, forecast_steps=1,
        od_index=0, delta_scale=0.1, ode_method="euler",
    )
    with pytest.raises(ValueError):
        m(torch.randn(2, 3), torch.randn(2, 4))  # x not 3D
    with pytest.raises(ValueError):
        m(torch.randn(2, 1, 3), torch.randn(2, 1))  # T < 2
