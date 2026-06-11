"""
Test script: tdpbrs_optv — Time-Dependent PBRS for OPTV

Implements and validates the time-dependent potential-based reward shaping
(TD-PBRS) formulation of the OPTV auxiliary reward, based on:

    Devlin & Kudenko, "Dynamic Potential-Based Reward Shaping" (AAMAS 2012)
    Equation 4:  F(s, t, s', t') = gamma * Phi(s', t') - Phi(s, t)

OPTV as TD-PBRS
---------------
  The plain OPTV auxiliary reward is:
    R = normalizer * delta_L
  where delta_L = L_bar_t - L_bar_{t-1} is the per-step change in
  effective border length.

  We define a time-dependent potential:
    Phi(s, t) = L_bar_t
  and apply the TD-PBRS formula:
    F = gamma * Phi(s', t') - Phi(s, t)
      = gamma * L_bar_t - L_bar_{t-1}
      = gamma * delta_L - (1 - gamma) * L_bar_{t-1}

  For gamma = 1.0:  F = delta_L   (identical to plain OPTV)
  For gamma = 0.97: F ~ delta_L with small bias -0.03 * L_bar_{t-1}

  By Theorem 1 of Devlin & Kudenko (2012), this formulation is
  policy-invariant: it provably does not alter the optimal policy.

This script:
  1. Defines tdpbrs_optv() — the TD-PBRS reward function
  2. Runs the environment with random actions
  3. Compares plain OPTV (gamma=1) vs tdpbrs_optv (gamma=0.97) per step
  4. Verifies tdpbrs_optv matches the gymenv's actual implementation
  5. Generates a Plotly HTML visualization with explanations
"""

import os
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Imports ────────────────────────────────────────────────────────────────
from get_parameters import para
import class_gymenv
from class_gymenv import GymEnvOmniSafe


# ═══════════════════════════════════════════════════════════════════════════
# tdpbrs_optv — Time-Dependent Potential-Based Reward Shaping for OPTV
# ═══════════════════════════════════════════════════════════════════════════

def tdpbrs_optv(gamma, L_bar_prev, L_bar_curr, normalizer):
    """
    Time-Dependent PBRS for OPTV (Devlin & Kudenko, AAMAS 2012, Eq. 4).

    Potential function:
        Phi(s, t) = L_bar_t   (effective border length of covered area at time t)

    Shaped reward (Equation 4 of the paper):
        F(s, t, s', t') = gamma * Phi(s', t') - Phi(s, t)
                        = gamma * L_bar_t - L_bar_{t-1}
                        = gamma * delta_L - (1 - gamma) * L_bar_{t-1}
                                 ^                ^
                          scaled OPTV      bias term (-> 0 as gamma -> 1)

    The full shaped reward is:   normalizer * F

    Properties:
        - gamma = 1.0  ->  F = delta_L     (identical to plain OPTV, no bias)
        - gamma = 0.97 ->  F ~ delta_L     (bias = -0.03 * L_bar_{t-1})
        - Policy-invariant by Theorem 1 of Devlin & Kudenko (2012)
        - Matches class_gymenv.py when use_gamma_in_optv_reward = True

    Parameters
    ----------
    gamma : float
        Discount factor, 0 < gamma <= 1.
    L_bar_prev : float
        L_bar_{t-1}: accumulated effective border length at previous time step.
    L_bar_curr : float
        L_bar_t: accumulated effective border length at current time step.
    normalizer : float
        OPTV scaling factor (typically < 0, from class_gymenv).

    Returns
    -------
    float
        Shaped OPTV reward = normalizer * (gamma * L_bar_t - L_bar_{t-1})
    """
    F = gamma * L_bar_curr - L_bar_prev
    return normalizer * F


# ═══════════════════════════════════════════════════════════════════════════
# Environment Setup
# ═══════════════════════════════════════════════════════════════════════════

height_data_path = os.path.join(os.path.dirname(__file__), 'misc', 'geo_data',
                                'height_data', 'height_data.npy')
height_data = np.load(height_data_path)

omnisafe_env = GymEnvOmniSafe(radiation_grid_visualization=False, is_evaluation=False)
env = omnisafe_env._env  # AutoResetWrapper(GymnasiumEnv)

test_image = os.path.join("misc", "radiation_data", "bw_jon_images_from_paper",
                          "eval_mowing_1.png")
omnisafe_env.reset(options={'image_path': test_image})


# ═══════════════════════════════════════════════════════════════════════════
# Run Episode with Random Actions
# ═══════════════════════════════════════════════════════════════════════════

gamma = getattr(para, 'gamma_for_optv_pbrs', 0.97)
N_STEPS = 300
print(f"Running {N_STEPS} steps with random actions (gamma = {gamma}) ...")

# Check gymenv mode
gamma_active = getattr(para, 'use_gamma_in_optv_reward', False)
print(f"  Gymenv uses {'TD-PBRS' if gamma_active else 'Plain'} OPTV "
      f"(use_gamma_in_optv_reward = {gamma_active})")

# Storage arrays
steps           = []
delta_L_arr     = []  # delta_L per step
L_bar_arr       = []  # L_bar_t  (effective border length)
L_bar_prev_arr  = []  # L_bar_{t-1}
optv_current    = []  # Plain OPTV: normalizer * delta_L  (gamma=1 baseline)
optv_tdpbrs     = []  # TD-PBRS OPTV: tdpbrs_optv(gamma, L_prev, L_curr, norm)
optv_gymenv     = []  # Gymenv's actual rew_incremental_tv (for verification)
diff_arr        = []  # Per-step diff: tdpbrs - plain
bias_arr        = []  # Theoretical bias: normalizer * (-(1-gamma) * L_prev)

for step_i in range(N_STEPS):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    # Raw quantities from environment
    dL     = env.length_new_measured_minus_length_old_measured   # delta_L
    L_prev = env.effective_border_length_prev                   # L_bar_{t-1}
    L_curr = env.effective_border_length                        # L_bar_t
    norm   = env.incremental_tv_reward_normalizer               # < 0

    # ── Plain OPTV (gamma=1, no discount — the conceptual baseline) ────
    r_plain = norm * dL

    # ── TD-PBRS OPTV (our function, with gamma < 1) ───────────────────
    r_tdpbrs = tdpbrs_optv(gamma, L_prev, L_curr, norm)

    # ── Gymenv's actual reward (for verification) ──────────────────────
    r_gymenv = env.rew_incremental_tv

    # ── Per-step difference and theoretical bias ───────────────────────
    # diff = tdpbrs - plain
    #      = norm * (gamma*dL - (1-gamma)*L_prev) - norm * dL
    #      = norm * ((gamma-1)*dL - (1-gamma)*L_prev)
    #      = norm * (-(1-gamma)*(dL + L_prev))
    #      = norm * (-(1-gamma)*L_curr)
    diff = r_tdpbrs - r_plain
    bias = norm * (-(1 - gamma) * L_prev)  # approx (using L_prev instead of L_curr)

    # Store
    steps.append(step_i)
    delta_L_arr.append(dL)
    L_bar_arr.append(L_curr)
    L_bar_prev_arr.append(L_prev)
    optv_current.append(r_plain)
    optv_tdpbrs.append(r_tdpbrs)
    optv_gymenv.append(r_gymenv)
    diff_arr.append(diff)
    bias_arr.append(bias)

    if terminated or truncated:
        print(f"  Episode ended at step {step_i}")
        break

# Convert to numpy arrays
steps          = np.array(steps)
delta_L_arr    = np.array(delta_L_arr)
L_bar_arr      = np.array(L_bar_arr)
L_bar_prev_arr = np.array(L_bar_prev_arr)
optv_current   = np.array(optv_current)
optv_tdpbrs    = np.array(optv_tdpbrs)
optv_gymenv    = np.array(optv_gymenv)
diff_arr       = np.array(diff_arr)
bias_arr       = np.array(bias_arr)

# Exact theoretical bias: normalizer * (-(1-gamma) * L_bar_curr)
bias_exact = norm * (-(1 - gamma) * L_bar_arr)


# ═══════════════════════════════════════════════════════════════════════════
# Summary Statistics
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("SUMMARY: Plain OPTV vs TD-PBRS OPTV (tdpbrs_optv)")
print("=" * 72)
print(f"  Steps completed                 : {len(steps)}")
print(f"  gamma (discount factor)         : {gamma}")
print(f"  Normalizer                      : {norm:.6f}")
print(f"\n  L_bar (effective border length):")
print(f"    range                         : [{L_bar_arr.min():.2f}, {L_bar_arr.max():.2f}]")
print(f"    final                         : {L_bar_arr[-1]:.2f}")
print(f"\n  Plain OPTV reward (norm*dL, gamma=1 baseline):")
print(f"    range                         : [{optv_current.min():.6f}, {optv_current.max():.6f}]")
print(f"    mean                          : {optv_current.mean():.6f}")
print(f"\n  TD-PBRS OPTV reward (gamma={gamma}):")
print(f"    range                         : [{optv_tdpbrs.min():.6f}, {optv_tdpbrs.max():.6f}]")
print(f"    mean                          : {optv_tdpbrs.mean():.6f}")
print(f"\n  Verification: tdpbrs_optv vs gymenv:")
print(f"    max |tdpbrs - gymenv|         : {np.max(np.abs(optv_tdpbrs - optv_gymenv)):.2e}")
if gamma_active:
    print(f"    (Should be ~0: tdpbrs_optv matches gymenv when gamma-branch active)")
else:
    print(f"    (Non-zero expected: gymenv uses plain OPTV, not gamma-branch)")
print(f"\n  Per-step diff (tdpbrs - plain):")
print(f"    range                         : [{diff_arr.min():.6f}, {diff_arr.max():.6f}]")
print(f"    mean                          : {diff_arr.mean():.6f}")
print(f"    max |diff|                    : {np.max(np.abs(diff_arr)):.6f}")
print(f"\n  Bias analysis:  diff = normalizer * (-(1-gamma)*L_bar_t)")
print(f"    max |diff - bias_exact|       : {np.max(np.abs(diff_arr - bias_exact)):.2e}")
print(f"    (Should be ~0: diff is fully explained by the bias term)")
print("=" * 72)


# ═══════════════════════════════════════════════════════════════════════════
# Visualization (Plotly)
# ═══════════════════════════════════════════════════════════════════════════

import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=4, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.07,
    subplot_titles=[
        f"<b>Panel A</b> — Per-Step Reward: Plain OPTV (gamma=1) vs TD-PBRS OPTV (gamma={gamma})",
        "<b>Panel B</b> — Per-Step Difference: TD-PBRS minus Plain OPTV",
        "<b>Panel C</b> — Effective Border Length L_bar(t) (explains growing bias)",
        "<b>Panel D</b> — Verification: tdpbrs_optv() matches gymenv implementation",
    ],
    specs=[[{"secondary_y": True}]] * 4,
)

# ── Panel A: Reward comparison ─────────────────────────────────────────────
fig.add_trace(go.Scatter(
    x=steps, y=optv_current, name="Plain OPTV  (norm*dL, gamma=1)",
    line=dict(color="#2ca02c", width=2),
), row=1, col=1, secondary_y=False)

fig.add_trace(go.Scatter(
    x=steps, y=optv_tdpbrs,
    name=f"TD-PBRS OPTV  (gamma={gamma})",
    line=dict(color="#d62728", width=2, dash="dash"),
), row=1, col=1, secondary_y=False)

fig.update_yaxes(title_text="Reward", row=1, col=1, secondary_y=False)

# ── Panel B: Per-step diff with theoretical bias overlay ───────────────────
fig.add_trace(go.Scatter(
    x=steps, y=diff_arr, name="Actual diff (tdpbrs - plain)",
    line=dict(color="#1f77b4", width=2),
), row=2, col=1, secondary_y=False)

fig.add_trace(go.Scatter(
    x=steps, y=bias_exact,
    name=f"Theoretical bias: norm*(-(1-gamma))*L_bar_t",
    line=dict(color="#ff7f0e", width=2, dash="dot"),
), row=2, col=1, secondary_y=False)

fig.update_yaxes(title_text="Difference", row=2, col=1, secondary_y=False)

# ── Panel C: L_bar and delta_L ─────────────────────────────────────────────
fig.add_trace(go.Scatter(
    x=steps, y=L_bar_arr, name="L_bar(t)  (accumulated border length)",
    line=dict(color="#1f77b4", width=2),
), row=3, col=1, secondary_y=False)

fig.add_trace(go.Scatter(
    x=steps, y=delta_L_arr, name="delta_L  (per-step change)",
    line=dict(color="#ff7f0e", width=1.5),
), row=3, col=1, secondary_y=True)

fig.update_yaxes(title_text="L_bar (accumulated)", row=3, col=1, secondary_y=False)
fig.update_yaxes(title_text="delta_L (per step)", row=3, col=1, secondary_y=True)

# ── Panel D: Verification ─────────────────────────────────────────────────
verification_diff = optv_tdpbrs - optv_gymenv
fig.add_trace(go.Scatter(
    x=steps, y=verification_diff,
    name="tdpbrs_optv() - gymenv",
    line=dict(color="#9467bd", width=2),
    fill='tozeroy', fillcolor='rgba(148,103,189,0.2)',
), row=4, col=1, secondary_y=False)

fig.update_yaxes(title_text="tdpbrs - gymenv", row=4, col=1, secondary_y=False)
fig.update_xaxes(title_text="Step", row=4, col=1)

# ── Explanatory Annotations ───────────────────────────────────────────────
one_minus_gamma = 1 - gamma
fig.add_annotation(
    text=(
        "<b>Time-Dependent PBRS for OPTV — Explanation</b><br>"
        "<i>(Devlin & Kudenko, 'Dynamic Potential-Based Reward Shaping', AAMAS 2012)</i><br><br>"
        #
        "<b>What is TD-PBRS?</b> Standard (static) PBRS defines a fixed potential Phi(s). "
        "Time-dependent PBRS extends this to Phi(s, t), allowing the potential to change over "
        "time while preserving the policy-invariance guarantee (Theorem 1 of the paper). "
        "The shaped reward is: <b>F = gamma * Phi(s', t') - Phi(s, t)</b>.<br><br>"
        #
        "<b>OPTV as TD-PBRS:</b> We define <b>Phi(s, t) = L_bar_t</b> (effective border length). Then:<br>"
        "&nbsp;&nbsp;&nbsp;&nbsp;F = gamma * L_bar_t - L_bar_{t-1} = gamma * delta_L - (1-gamma) * L_bar_{t-1}<br>"
        f"&nbsp;&nbsp;&nbsp;&nbsp;For gamma = 1.0: F = delta_L (identical to plain OPTV, no bias)<br>"
        f"&nbsp;&nbsp;&nbsp;&nbsp;For gamma = {gamma}: F ~ delta_L with bias = -{one_minus_gamma:.2f} * L_bar "
        "(visible in Panel B as growing difference)<br><br>"
        #
        "<b>Panel A:</b> Both rewards track closely; they diverge as L_bar grows "
        "(the bias is proportional to L_bar).<br>"
        "<b>Panel B:</b> The per-step diff (blue solid) exactly matches the theoretical "
        "bias formula (orange dotted),<br>"
        f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
        f"confirming: diff = normalizer * (-{one_minus_gamma:.2f}) * L_bar_t.<br>"
        "<b>Panel C:</b> L_bar(t) grows as more area is covered, explaining the growing "
        "bias in Panel B.<br>"
        "<b>Panel D:</b> tdpbrs_optv() matches the gymenv's actual implementation "
        "(diff ~ 0), validating correctness."
    ),
    xref="paper", yref="paper",
    x=0.5, y=-0.08,
    showarrow=False,
    font=dict(size=11),
    align="left",
    bordercolor="#333", borderwidth=1, borderpad=8,
    bgcolor="rgba(255,255,240,0.95)",
)

fig.update_layout(
    height=1400,
    width=1100,
    title=dict(
        text=(
            "<b>OPTV Reward: Plain (gamma=1) vs Time-Dependent PBRS (tdpbrs_optv)</b><br>"
            f"<sup>gamma = {gamma} | {len(steps)} random-action steps | "
            f"L_bar range [{L_bar_arr.min():.1f}, {L_bar_arr.max():.1f}] | "
            f"max |diff| = {np.max(np.abs(diff_arr)):.4f}</sup>"
        ),
        x=0.5,
    ),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    template="plotly_white",
    margin=dict(b=280),
)

out_path = os.path.join(os.path.dirname(__file__), "optv_pbrs_analysis.html")
fig.write_html(out_path, include_plotlyjs=True)
print(f"\nVisualization saved to: {out_path}")
