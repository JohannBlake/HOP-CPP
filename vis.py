# Visualize a trained HOP-CPP agent.
#
# Loads a checkpoint produced by train.py (misc/logs/runs/.../torch_save/epoch-*.pt),
# rolls out deterministic episodes in the coverage environment, prints episode
# metrics, and saves one trajectory image per episode.
#
# Usage:
#   python vis.py --model path/to/epoch-N.pt [--episodes 3] [--out vis_output]
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np
import torch
import yaml


def load_vendored_omnisafe():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "omnisafe")
    spec = importlib.util.spec_from_file_location("omnisafe", os.path.join(path, "__init__.py"))
    omnisafe = importlib.util.module_from_spec(spec)
    sys.modules["omnisafe"] = omnisafe
    spec.loader.exec_module(omnisafe)
    return omnisafe


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize a trained HOP-CPP agent.")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to a checkpoint, e.g. misc/logs/runs/.../torch_save/epoch-10.pt")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to roll out.")
    parser.add_argument("--out", type=str, default="vis_output", help="Output directory for trajectory PNGs.")
    parser.add_argument("--seed", type=int, default=0, help="Environment seed for the first episode.")
    parser.add_argument("--max-steps", type=int, default=8000, help="Step cap per episode.")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device (default: cuda if available, else cpu).")
    return parser.parse_args()


def build_policy(model_path, observation_space, action_space, device):
    from omnisafe.models.actor.actor_builder import ActorBuilder

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "omnisafe", "configs", "off-policy", "TD3Lag.yaml")
    with open(config_path, "r") as f:
        alg_config = yaml.safe_load(f)
    actor_type = alg_config["defaults"]["model_cfgs"]["actor_type"]
    hidden_sizes = alg_config["defaults"]["model_cfgs"]["actor"]["hidden_sizes"]

    actor_builder = ActorBuilder(observation_space, action_space, hidden_sizes=hidden_sizes)
    policy = actor_builder.build_actor(actor_type=actor_type)

    state = torch.load(model_path, map_location="cpu")
    state_dict = state.get("pi", state)
    model_state_dict = policy.state_dict()
    filtered_state_dict = {}
    for k, v in state_dict.items():
        if k in model_state_dict and v.shape == model_state_dict[k].shape:
            filtered_state_dict[k] = v
        elif k in model_state_dict:
            print(f"Warning: shape mismatch for {k}: checkpoint {v.shape} vs model {model_state_dict[k].shape}")
    policy.load_state_dict(filtered_state_dict, strict=False)
    policy.eval()
    return policy.to(device)


def to_scalar(x):
    return float(x.item()) if hasattr(x, "item") else float(x)


def main():
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.out, exist_ok=True)

    load_vendored_omnisafe()
    import class_gymenv  # registers GymEnvOmniSafe-v0; must follow omnisafe load

    env = class_gymenv.GymEnvOmniSafe(radiation_grid_visualization=False, is_evaluation=True)
    gymenv = env._env.env  # unwrapped GymnasiumEnv (for render + metrics)

    policy = build_policy(args.model, env.observation_space, env.action_space, device)

    obs, _ = env.reset(seed=args.seed)
    for episode in range(args.episodes):
        terminated = truncated = False
        ep_reward = 0.0
        ep_cost = 0.0
        steps = 0
        while not (terminated or truncated) and steps < args.max_steps:
            with torch.no_grad():
                if isinstance(obs, np.ndarray):
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                else:
                    obs_tensor = obs.to(device)
                action = policy.predict(obs_tensor, deterministic=True)
                if isinstance(action, torch.Tensor):
                    action = action.cpu()
            obs, reward, cost, terminated, truncated, _info = env.step(action)
            terminated = bool(to_scalar(terminated))
            truncated = bool(to_scalar(truncated))
            ep_reward += to_scalar(reward)
            ep_cost += to_scalar(cost)
            steps += 1

        collisions = getattr(gymenv, "num_episode_collisions", float("nan"))
        coverage = 1.0 - float(getattr(gymenv, "percentage_of_target_area_left", float("nan")))
        print(f"Episode {episode}: steps={steps} return={ep_reward:.2f} cost={ep_cost:.2f} "
              f"collisions={collisions} coverage={coverage:.3f}")

        # Resetting flushes the finished episode into previous_episode_data,
        # which render() draws from; it also starts the next episode.
        obs, _ = env.reset(seed=args.seed + episode + 1)
        save_path = os.path.join(args.out, f"episode_{episode}.png")
        gymenv.render(save_path=save_path)
        print(f"Saved trajectory image to {save_path}")


if __name__ == "__main__":
    main()
