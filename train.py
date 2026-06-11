# Imports
from __future__ import annotations
import yaml
import wandb
# Imports
import importlib
import get_parameters
from get_parameters import para
from stable_baselines3 import PPO, A2C, TD3, DDPG, SAC, DQN
from sb3_contrib import TRPO
from stable_baselines3.common.callbacks import EvalCallback
import os
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, SubprocVecEnv
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
import numpy as np
import json
import os

policy_type = 'MultiInputPolicy'

# 1: Define objective/training function
def objective():
    if para.training_library == 'sb3':
        import class_gymenv
        # Create training environments
        height_data_path = os.path.join('.', 'misc', 'geo_data','height_data', 'height_data.npy')    
        height_data = np.load(height_data_path) 
        envs = [lambda: Monitor(class_gymenv.GymnasiumEnv(height_data)) for _ in range(para.num_envs_for_training)]
        if para.dummy_or_subproc == 'dummy':
            gymenv = DummyVecEnv(envs) 
        elif para.dummy_or_subproc == 'subproc':
            gymenv = SubprocVecEnv(envs)
            
        gymenv = VecTransposeImage(gymenv)

        # Create evaluation environments
        envs_eval = [lambda: Monitor(class_gymenv.GymnasiumEnv(height_data, logging_enabled = True)) for _ in range(para.num_envs_for_training)]
        if para.dummy_or_subproc == 'dummy':
            gymenv_eval = DummyVecEnv(envs_eval) 
        elif para.dummy_or_subproc == 'subproc':
            gymenv_eval = SubprocVecEnv(envs_eval)
        gymenv_eval = VecTransposeImage(gymenv_eval) 
        directory_path = f'./logs/{wandb.run.name}' 
        os.makedirs(directory_path, exist_ok=True)
        n_actions = gymenv.action_space.shape[-1]
        mean = np.zeros(n_actions)
        sigma = para.sigma_coef_for_noise * np.ones(n_actions)

        action_noise = OrnsteinUhlenbeckActionNoise(mean=mean, sigma=sigma)

        if para.sb3_model_type == 'PPO':
            model = PPO(policy_type, gymenv, verbose=0,
                        learning_rate=para.learning_rate,
                        n_steps=para.n_steps,
                        batch_size=para.batch_size,
                        n_epochs=para.n_epochs,
                        gamma=para.gamma,
                        clip_range=para.clip_range,
                        max_grad_norm = 0.3,
                        ent_coef=para.ent_coef)
        elif para.sb3_model_type == 'A2C':
            model = A2C(policy_type, gymenv, verbose=0,
                        learning_rate=para.learning_rate,
                        gamma=para.gamma,
                        n_steps=para.n_steps,
                        ent_coef=para.ent_coef)
        elif para.sb3_model_type == 'TD3':
            model = TD3(policy_type, gymenv, verbose=0,
                        learning_rate=para.learning_rate_td3,
                        buffer_size=para.buffer_size,
                        learning_starts=para.learning_starts,
                        batch_size=para.batch_size,
                        gamma=para.gamma,
                        train_freq=(para.train_freq_td3, "step"),
                        gradient_steps=para.gradient_steps_td3,
                        action_noise=action_noise if para.include_action_noise else None
                        )
        elif para.sb3_model_type == 'DDPG':
            model = DDPG(policy_type, gymenv, verbose=0,
                        learning_rate=para.learning_rate,
                        learning_starts=para.learning_starts,
                        buffer_size=para.buffer_size,
                        gamma=para.gamma,
                        action_noise=action_noise if para.include_action_noise else None
                        )
        elif para.sb3_model_type == 'DQN': # test
            model = DQN(policy_type, gymenv, verbose=0,
                        learning_rate=para.learning_rate,
                        buffer_size=para.buffer_size,
                        learning_starts=para.learning_starts,
                        gamma=para.gamma,
                        train_freq=para.train_freq)
        elif para.sb3_model_type == 'TRPO':
            model = TRPO(policy_type, gymenv, verbose=0,
                        learning_rate=para.learning_rate,
                        gamma=para.gamma,
                        )
        elif para.sb3_model_type == 'SAC':
            model = SAC(policy_type, gymenv, verbose=0,
                        learning_rate=para.learning_rate_sac,
                        buffer_size=para.buffer_size,
                        batch_size=para.batch_size,
                        learning_starts=para.learning_starts,
                        gamma=para.gamma,
                        train_freq=para.train_freq_sac,
                        gradient_steps=para.gradient_steps_sac,
                        action_noise=action_noise if para.include_action_noise else None
                        )
        else:
            raise ValueError("Unsupported model type")
        class EvalAndWandbCallback(EvalCallback): # logs only for the first of n envs
            def __init__(self, eval_env, log_freq, *args, **kwargs):
                super().__init__(eval_env, *args, **kwargs)
                self.log_freq = log_freq
                self._last_logged_step = 0
                self.last_log_dict = {}
            def _on_step(self) -> bool:
                result = super()._on_step()
                # Log every log_freq steps using the most recent eval infos
                if (
                    self.num_timesteps > 0
                    and self.num_timesteps % self.log_freq == 0
                    and self._last_logged_step != self.num_timesteps
                ):
                    self._last_logged_step = self.num_timesteps
                    log_dir = os.path.join(os.path.dirname(__file__), "misc", "logs", "gymenv_logs")
                    # Find all .json files in the directory (sorted for determinism)
                    log_files = sorted([f for f in os.listdir(log_dir) if f.endswith(".json")])
                    log_dicts = []
                    for log_file in log_files:
                        log_file_path = os.path.join(log_dir, log_file)
                        try:
                            with open(log_file_path, "r") as f:
                                # Read the last non-empty line (latest log)
                                for line in reversed(f.readlines()):
                                    if line.strip():
                                        log_dicts.append(json.loads(line))
                                        break
                        except Exception as e:
                            print(f"Failed to read log from {log_file_path}: {e}")
                    if log_dicts:
                        # Compute average for each key
                        avg_log_dict = {}
                        keys = set().union(*log_dicts)
                        for key in keys:
                            values = [d[key] for d in log_dicts if key in d and isinstance(d[key], (int, float))]
                            if values:
                                avg_log_dict[key] = sum(values) / len(values)
                            else:
                                # If not numeric or missing, just take the first present value
                                for d in log_dicts:
                                    if key in d:
                                        avg_log_dict[key] = d[key]
                                        break
                        if self.last_log_dict != avg_log_dict or self.last_log_dict == {}:
                            wandb.log(avg_log_dict, step=self.num_timesteps)
                        self.last_log_dict = avg_log_dict
                return result
        eval_callback = EvalAndWandbCallback(
            eval_env=gymenv_eval,
            best_model_save_path=directory_path,
            eval_freq=para.logging_frequency,
            n_eval_episodes=para.n_eval_episodes,
            deterministic=True,
            render=False,
            log_freq=para.logging_frequency,
        )
        try:
            print("Environment observation space", gymenv.observation_space)
            model.learn(
                total_timesteps=para.total_timesteps_for_training,
                callback=eval_callback
            )
        except Exception as e:
            import traceback
            print("Exception during model.learn:", e, flush=True)
            print(traceback.format_exc(), flush=True)
            try:
                wandb.log({"error": str(e)})
            except Exception:
                pass
            raise
    if para.training_library == 'omnisafe':
        ### instead of #########################
        #from omnisafe import Agent
        #from omnisafe.envs.core import CMDP, env_register
        # the logic below is used
        import sys
        import importlib.util
        path_to_modified_omnisafe = os.path.join(os.path.dirname(__file__), "omnisafe")

        # Modul-Spec für 'omnisafe' erstellen
        spec = importlib.util.spec_from_file_location("omnisafe", os.path.join(path_to_modified_omnisafe, "__init__.py"))

        # Modul laden
        omnisafe = importlib.util.module_from_spec(spec)
        sys.modules["omnisafe"] = omnisafe
        spec.loader.exec_module(omnisafe)


        import class_gymenv
        Agent = omnisafe.Agent
        CMDP = omnisafe.envs.core.CMDP
        env_register = omnisafe.envs.core.env_register

        ### instead of #########################
        #from omnisafe import Agent
        #from omnisafe.envs.core import CMDP, env_register
        # the logic above is usedtest

        # Pass seed from parameters_default.yaml to ensure reproducibility
        custom_cfgs = {'seed': para.seed} if hasattr(para, 'seed') else {}
        agent = Agent(algo=para.omnisafe_alg, env_id="GymEnvOmniSafe-v0", custom_cfgs=custom_cfgs)
        env_to_discard = class_gymenv.GymEnvOmniSafe(radiation_grid_visualization=False)
        # Load parameters_default.yaml and update wandb config if wandb logging is enabled
        with open("parameters_default.yaml", "r") as file:
            yaml_parameter_defaults_config = yaml.safe_load(file)
        
        # Update wandb config with parameters_default.yaml content if wandb is being used
        if agent.cfgs.logger_cfgs.use_wandb:
            try:
                wandb.config.update(yaml_parameter_defaults_config)
            except Exception as e:
                print(f"Warning: Could not update wandb config with parameters_default.yaml: {e}")
        
        try:
            agent.learn()
        except Exception as e:
            import traceback
            print("Exception during model.learn:", e, flush=True)
            print(traceback.format_exc(), flush=True)
            try:
                wandb.log({"error": str(e)})
            except Exception:
                pass
            raise
def main():
    if para.surface_grid_creation_type == "no_height_map":
        minimal_height_data = np.array([
            [7.18375, 48.60152778, 0.0]
        ])
        file_path = os.path.join(os.getcwd(), "misc", 'geo_data','height_data' ,"height_data.npy")
        np.save(file_path, minimal_height_data)
    #else:
    #    import misc.aid.create_height_map
    with open("parameters_default.yaml", "r") as file:
        yaml_parameter_defaults_config = yaml.safe_load(file)
    if para.training_library == "classic":
        wandb.init(project="Heli-Logs", config=yaml_parameter_defaults_config)
    elif para.training_library == "constrained":
        # For constrained RL, wandb.init is handled by omnisafe, 
        # but we'll update the config after agent creation in objective()
        pass
    objective()
if __name__ == "__main__":
    main()