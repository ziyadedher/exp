import glob
import json
import os
import pickle
import time

import ray
import ray.rllib.agents.ppo as ppo
from exp.envs import CartPole, LunarLanderContinuous
from exp.rollout import rollout

BASE_PATH = "/home/ziyadedher/exp/results/ray"
EXPERIMENT_STATE_GLOB = "experiment_state*.json"
RESULTS_PATH = "/home/ziyadedher/results/test"

ENV = CartPole
STEPS = 2500


ray.init()


def recent_checkpoint(checkpoint):
    num = max(int(x.split("_")[-1]) for x in glob.glob(os.path.join(checkpoint["logdir"], "checkpoint_*")))
    return os.path.join(checkpoint["logdir"], f"checkpoint_{num}", f"checkpoint-{num}")


def run(agent, config):
    env = ENV(config)
    results = rollout(
        agent, env, STEPS,
        force_use_env=True,
        no_render=False,
        monitor=False,
        print_results=False
    )
    env.close()

    return {
        "config": config,
        "rewards": results,
    }


@ray.remote
def test(checkpoint, configs):
    checkpoint["config"]["env"] = pickle.loads(bytes.fromhex(checkpoint["config"]["env"]["value"]))
    agent = ppo.PPOTrainer(config=checkpoint["config"])
    agent.restore(recent_checkpoint(checkpoint))

    return {
        "config": checkpoint["config"]["env_config"],
        "results": [run(agent, config) for config in configs],
    }


def evaluate(experiment, configs):
    path = os.path.join(BASE_PATH, experiment)
    if not os.path.isdir(path):
        raise ValueError(f"{path} is not a valid directory")

    try:
        state_path = glob.glob(os.path.join(path, EXPERIMENT_STATE_GLOB))[0]
    except IndexError:
        raise ValueError(f"no experiment state JSON found in {path}")

    with open(state_path, "r") as file:
        state = json.load(file)

    return ray.get([test.remote(checkpoint, configs) for checkpoint in state["checkpoints"]])


if __name__ == '__main__':
    experiment = "CartPole_conf_both"
    test_configs = [
        {"action_map": ['lambda action: action', 'lambda action: int(not action)']},
    ]

    start_time = int(time.time())
    if not os.path.isdir(RESULTS_PATH):
        try:
            os.makedirs(RESULTS_PATH)
        except IOError:
            raise ValueError(f"results path {RESULTS_PATH} does not exist and could not be created")

    results = evaluate(experiment, test_configs)

    with open(os.path.join(RESULTS_PATH, f"{experiment}_{start_time}.json"), "w") as file:
        json.dump(results, file, indent="\t")
