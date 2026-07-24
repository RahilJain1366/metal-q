"""
Error metrics for the TGV solver.

All metrics work on flat (N²,) numpy arrays.
"""

import numpy as np


def l2_relative(u_num: np.ndarray, u_ex: np.ndarray) -> float:
    """Relative L2 error: ‖u_num − u_ex‖ / ‖u_ex‖."""
    denom = np.linalg.norm(u_ex)
    if denom < 1e-30:
        return float(np.linalg.norm(u_num - u_ex))
    return float(np.linalg.norm(u_num - u_ex) / denom)


def velocity_l2(u_num, v_num, u_ex, v_ex) -> float:
    """Combined relative L2 error on velocity field."""
    num = np.sqrt(np.mean((u_num - u_ex)**2 + (v_num - v_ex)**2))
    den = np.sqrt(np.mean(u_ex**2 + v_ex**2))
    return float(num / (den + 1e-30))


def max_divergence(div_op, u: np.ndarray, v: np.ndarray) -> float:
    """max |∇·(u,v)| — should be ~machine-ε after projection."""
    uv = np.concatenate([u, v])
    return float(np.max(np.abs(div_op @ uv)))


def convergence_rate(h_vals, err_vals) -> float:
    """Estimate convergence rate p from log-log slope of err vs h."""
    log_h = np.log(h_vals)
    log_e = np.log(err_vals)
    if len(log_h) < 2:
        return float('nan')
    slope = np.polyfit(log_h, log_e, 1)[0]
    return float(slope)
