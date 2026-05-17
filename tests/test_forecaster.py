import math

import pytest
import torch
import torch.nn.functional as F

from quantum_liquid_neuralode.models import LiquidODForecaster


def _toy_inputs(batch=2, T=8, F_dim=4):
    torch.manual_seed(0)
    x = torch.randn(batch, T, F_dim)
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


def test_forecaster_residual_anchored_init_matches_request():
    """Learnable delta_scale: at init, softplus(unconstrained) + min must equal
    the requested init value, and the predicted residual must stay bounded by
    that scale.
    """
    x, t = _toy_inputs()
    delta_scale_init = 0.02
    delta_scale_min = 0.001
    m = LiquidODForecaster(
        input_size=x.shape[-1], hidden_size=8,
        horizon_hours=1.0, forecast_steps=1,
        od_index=0,
        delta_scale_init=delta_scale_init,
        delta_scale_min=delta_scale_min,
        ode_method="euler",
    )
    # 1. The learnable scale, evaluated at init, must round-trip the request.
    assert float(m.delta_scale().item()) == pytest.approx(delta_scale_init, rel=1e-5, abs=1e-7)

    # 2. The bias-only correction at init must lie within delta_scale + slop.
    od_last = x[:, -1, 0]
    y = m(x, t).detach()
    assert torch.all((y - od_last).abs() <= delta_scale_init * 1.1)


def test_forecaster_delta_scale_is_learnable_parameter():
    """The unconstrained delta-scale param must show up in m.parameters() and
    receive gradients during a backward pass.
    """
    x, t = _toy_inputs()
    m = LiquidODForecaster(
        input_size=x.shape[-1], hidden_size=8,
        horizon_hours=1.0, forecast_steps=1,
        od_index=0, delta_scale_init=1.0, ode_method="euler",
    )
    # Discover by identity rather than by name.
    assert any(p is m.delta_scale_unconstrained for p in m.parameters())
    y = m(x, t)
    loss = (y ** 2).mean()
    loss.backward()
    g = m.delta_scale_unconstrained.grad
    assert g is not None
    # tanh-bounded delta means grad is non-trivial in general; smoke-check.
    assert torch.isfinite(g)


def test_forecaster_legacy_delta_scale_kwarg_still_works():
    """Configs and scripts still pass `delta_scale=0.1`; the legacy alias must
    be treated as the init of the now-learnable parameter without breaking.
    """
    m = LiquidODForecaster(
        input_size=3, hidden_size=4, horizon_hours=1.0, forecast_steps=1,
        od_index=0, delta_scale=0.1, ode_method="euler",
    )
    assert float(m.delta_scale().item()) == pytest.approx(0.1, rel=1e-5, abs=1e-7)


def test_forecaster_rejects_both_init_kwargs():
    with pytest.raises(ValueError):
        LiquidODForecaster(
            input_size=3, hidden_size=4, horizon_hours=1.0, forecast_steps=1,
            od_index=0, delta_scale=0.1, delta_scale_init=0.2,
            ode_method="euler",
        )


def test_forecaster_rejects_init_below_min():
    """The unconstrained pre-image is only well defined for init > min."""
    with pytest.raises(ValueError):
        LiquidODForecaster(
            input_size=3, hidden_size=4, horizon_hours=1.0, forecast_steps=1,
            od_index=0, delta_scale_init=0.005, delta_scale_min=0.01,
            ode_method="euler",
        )


def test_forecaster_rejects_bad_shapes():
    m = LiquidODForecaster(
        input_size=3, hidden_size=4, horizon_hours=1.0, forecast_steps=1,
        od_index=0, delta_scale=0.1, ode_method="euler",
    )
    with pytest.raises(ValueError):
        m(torch.randn(2, 3), torch.randn(2, 4))  # x not 3D
    with pytest.raises(ValueError):
        m(torch.randn(2, 1, 3), torch.randn(2, 1))  # T < 2
