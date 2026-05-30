"""Run the UR10 + suction-gripper cube stacking task.

Stacks three cubes bottom-to-top: blue -> red -> green using a UR10 arm with
either a long or short surface (suction) gripper.

Uses the Isaac Lab registered tasks:
    Isaac-Stack-Cube-UR10-Long-Suction-IK-Rel-v0
    Isaac-Stack-Cube-UR10-Short-Suction-IK-Rel-v0

Usage:
    python run_ur10_stack.py --gripper long
    python run_ur10_stack.py --gripper short --num_envs 1
    python run_ur10_stack.py --gripper long --agent zero      # robot stays still
    python run_ur10_stack.py --gripper long --agent random    # random actions
    python run_ur10_stack.py --gripper long --headless --device cpu

Notes:
    * Surface grippers require IsaacSim 5.0+ and CPU simulation (`--device cpu`).
    * Accept the Omniverse EULA the first time by setting OMNI_KIT_ACCEPT_EULA=YES.
"""

import argparse
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="UR10 suction-gripper cube stacking.")
parser.add_argument(
    "--gripper",
    type=str,
    default="long",
    choices=["long", "short"],
    help="Which surface gripper variant to use.",
)
parser.add_argument(
    "--agent",
    type=str,
    default="zero",
    choices=["zero", "random"],
    help="Action source: 'zero' (stand still) or 'random'.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument("--max_steps", type=int, default=0, help="Stop after N steps (0 = run forever).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Suction grippers require CPU simulation.
if getattr(args_cli, "device", "cuda:0") != "cpu":
    print("[INFO] Forcing --device cpu (surface gripper requires CPU sim).", flush=True)
    args_cli.device = "cpu"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401  # registers all Isaac-* gym envs


TASK_ID = {
    "long": "Isaac-Stack-Cube-UR10-Long-Suction-IK-Rel-v0",
    "short": "Isaac-Stack-Cube-UR10-Short-Suction-IK-Rel-v0",
}


def main() -> int:
    task = TASK_ID[args_cli.gripper]
    print(f"[INFO] Creating gym env: {task} (num_envs={args_cli.num_envs}, agent={args_cli.agent})", flush=True)

    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(
        task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    try:
        env = gym.make(task, cfg=env_cfg)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        return 1

    print(f"[INFO] Env created. Action space: {env.action_space}", flush=True)
    print(f"[INFO] Observation space keys: {list(env.observation_space.spaces.keys()) if hasattr(env.observation_space, 'spaces') else env.observation_space}", flush=True)
    print(
        "[INFO] Cubes: cube_1=blue (bottom)  cube_2=red (middle)  cube_3=green (top)."
        " Goal: stack cube_2 on cube_1, then cube_3 on cube_2.",
        flush=True,
    )

    env.reset()
    print("[INFO] env.reset() done. Entering simulation loop.", flush=True)

    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            if args_cli.agent == "zero":
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            else:
                actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1

            env.step(actions)

            step += 1
            if step == 1 or step % 50 == 0:
                print(f"[INFO] step {step}", flush=True)
            if args_cli.max_steps and step >= args_cli.max_steps:
                print(f"[INFO] Reached max_steps={args_cli.max_steps}, exiting.", flush=True)
                break

    env.close()
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        rc = 1
    finally:
        simulation_app.close()
    sys.exit(rc)
