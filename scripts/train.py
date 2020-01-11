import ray
from exp import models
from exp.envs import CartPole, LunarLanderContinuous
from ray import tune

BASE_PATH = "/home/ziyadedher/exp/results/ray"


ray.init()


tune.run(
    "PPO",

    name="CartPole_conf_both",

    stop={
        "episode_reward_mean": 495,
    },

    config={
        "num_workers": 7,
        "num_envs_per_worker": 1,
        "sample_batch_size": 200,
        "batch_mode": "complete_episodes",

        "env": CartPole,
        "env_config": {
            "time_limit": 500,
            "with_confounder": True,
            "action_map": ['lambda x: x', 'lambda x: int(not x)'],
        },
        "gamma": 0.99,
        "horizon": None,
        "soft_horizon": False,
        "no_done_at_end": False,
        "clip_rewards": None,
        "clip_actions": True,
        "preprocessor_pref": "deepmind",
        "lr": 0.0001,

        "num_gpus": 0,
        "train_batch_size": 200,
        "model": {
            "conv_filters": None,
            "conv_activation": "relu",
            "fcnet_activation": "tanh",
            "fcnet_hiddens": [128, 256, 32],
            "free_log_std": False,
            "no_final_linear": False,
            "vf_share_layers": True,

            "use_lstm": False,
            "max_seq_len": 20,
            "lstm_cell_size": 256,
            "lstm_use_prev_action_reward": False,
            "state_shape": None,

            "framestack": True,
            "dim": 84,

            "custom_preprocessor": None,
            "custom_model": "ConfounderLatent",
            "custom_action_dist": None,
            "custom_options": {},
        },
        "optimizer": {},

        "monitor": False,
        "log_level": "WARN",
        "callbacks": {
            "on_episode_start": None,     # arg: {"env": .., "episode": ...}
            "on_episode_step": None,      # arg: {"env": .., "episode": ...}
            "on_episode_end": None,       # arg: {"env": .., "episode": ...}
            "on_sample_end": None,        # arg: {"samples": .., "worker": ...}
            "on_train_result": None,      # arg: {"trainer": ..., "result": ...}
            "on_postprocess_traj": None,  # arg: {
                                          #   "agent_id": ..., "episode": ...,
                                          #   "pre_batch": (before processing),
                                          #   "post_batch": (after processing),
                                          #   "all_pre_batches": (other agent ids),
                                          # }
        },
        "ignore_worker_failures": False,
        "log_sys_usage": True,
        "eager": False,
        "eager_tracing": False,
        "no_eager_on_workers": False,

        "evaluation_interval": None,
        "evaluation_num_episodes": 10,
        "evaluation_config": {},

        "sample_async": False,
        "observation_filter": "NoFilter",
        "synchronize_filters": True,
        "tf_session_args": {
            # note: overriden by `local_tf_session_args`
            "intra_op_parallelism_threads": 2,
            "inter_op_parallelism_threads": 2,
            "gpu_options": {
                "allow_growth": True,
            },
            "log_device_placement": False,
            "device_count": {
                "CPU": 1
            },
            "allow_soft_placement": True,  # required by PPO multi-gpu
        },
        "local_tf_session_args": {
            "intra_op_parallelism_threads": 8,
            "inter_op_parallelism_threads": 8,
        },
        "compress_observations": False,
        "collect_metrics_timeout": 180,
        "metrics_smoothing_episodes": 100,
        "remote_worker_envs": False,
        "remote_env_batch_wait_ms": 0,
        "min_iter_time_s": 0,
        "timesteps_per_iteration": 0,
        "seed": None,

        "num_cpus_per_worker": 1,
        "num_gpus_per_worker": 0,
        "custom_resources_per_worker": {},
        "num_cpus_for_driver": 1,
        "memory": 0,
        "object_store_memory": 0,
        "memory_per_worker": 0,
        "object_store_memory_per_worker": 0,

        "input": "sampler",
        "input_evaluation": ["is", "wis"],
        "postprocess_inputs": False,
        "shuffle_buffer_size": 0,
        "output": None,
        "output_compress_columns": ["obs", "new_obs"],
        "output_max_file_size": 64 * 1024 * 1024,

        "multiagent": {
            "policies": {},
            "policy_mapping_fn": None,
            "policies_to_train": None,
        },
    },
    checkpoint_at_end=True,
    checkpoint_freq=10,
    verbose=1,
    reuse_actors=True,
    local_dir=BASE_PATH,

)
