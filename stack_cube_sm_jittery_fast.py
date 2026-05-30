#!/usr/bin/env python3
"""
Franka Cube Stacking Demo — Multi-env with state machine, fast motions, vibration, and missed grasps.

Usage:
    conda activate env_isaaclab
    cd /home/truongnq/Desktop/3cube
    python stack_cube_sm_jittery_fast.py --num_envs 4 --vibration_pos 0.06 --vibration_rot 0.10 --miss_rate 0.4
"""

"""Launch Omniverse first."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Multi-env Franka cube stacking demo with vibration.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--vibration_pos", type=float, default=0.06, help="Standard deviation of position vibration noise.")
parser.add_argument("--vibration_rot", type=float, default=0.10, help="Standard deviation of rotation vibration noise.")
parser.add_argument("--miss_rate", type=float, default=0.4, help="Probability of missing a grasp (0.0 to 1.0).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import math
import gymnasium as gym
import torch
import isaaclab.utils.math as math_utils
from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


# ── State Machine ──────────────────────────────────────────────────────────

class S:
    REST = 0
    ABOVE_PICK = 1
    DOWN_PICK = 2
    GRASP = 3
    LIFT = 4
    ABOVE_PLACE = 5
    DOWN_PLACE = 6
    RELEASE = 7
    RETREAT = 8
    NEXT = 9
    DONE = 10  # wait then let env timeout/reset


WAIT = {
    S.REST: 0.05,     # Faster transition (down from 0.1)
    S.GRASP: 0.15,    # Faster grasp (down from 0.35)
    S.RELEASE: 0.1,   # Faster release (down from 0.2)
    S.DONE: 0.5,      # Faster done hold (down from 1.0)
}

STATE_NAMES = [
    "REST", "ABOVE_PICK", "DOWN_PICK", "GRASP", "LIFT",
    "ABOVE_PLACE", "DOWN_PLACE", "RELEASE", "RETREAT", "NEXT", "DONE",
]

# Base gripper-down orientation: 180° rotation about x-axis = (w=0, x=1, y=0, z=0).
FLIP_QUAT = torch.tensor([0.0, 1.0, 0.0, 0.0])


def yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Extract yaw (rotation about world z) from a (N, 4) wxyz quaternion."""
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_quarter(yaw: torch.Tensor) -> torch.Tensor:
    """Wrap yaw to (-pi/4, pi/4]."""
    return ((yaw + math.pi / 4.0) % (math.pi / 2.0)) - math.pi / 4.0


def _ee_quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    half = yaw * 0.5
    c, s = torch.cos(half), torch.sin(half)
    zero = torch.zeros_like(c)
    return torch.stack((zero, c, s, zero), dim=-1)


def _angular_dist_mod_pi(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Shortest distance between two angles modulo π."""
    d = (a - b + math.pi / 2.0) % math.pi - math.pi / 2.0
    return torch.abs(d)


def grasp_yaw_avoiding(
    pick_pos: torch.Tensor,
    pick_quat: torch.Tensor,
    obstacles_pos: torch.Tensor,
) -> torch.Tensor:
    """Pick the best gripper yaw to grasp a cube while keeping jaws away from obstacles."""
    cube_yaw = wrap_quarter(yaw_from_quat(pick_quat))

    # Pick the NEAREST obstacle (xy distance).
    dxy = obstacles_pos[..., :2] - pick_pos[:, None, :2]  # (N, K, 2)
    d2 = (dxy * dxy).sum(dim=-1)                          # (N, K)
    nearest = d2.argmin(dim=-1)                           # (N,)
    sel = nearest[:, None, None].expand(-1, 1, 2)         # (N, 1, 2)
    nearest_xy = obstacles_pos[..., :2].gather(1, sel).squeeze(1)  # (N, 2)
    min_d = d2.gather(1, nearest[:, None]).squeeze(-1)    # (N,)

    diff = pick_pos[:, :2] - nearest_xy
    pref_yaw = torch.atan2(diff[..., 1], diff[..., 0])

    cand1 = cube_yaw
    cand2 = cube_yaw + math.pi / 2.0

    use1 = _angular_dist_mod_pi(cand1, pref_yaw) <= _angular_dist_mod_pi(cand2, pref_yaw)
    chosen = torch.where(use1, cand1, cand2)

    too_close = min_d < 1.0e-6
    return torch.where(too_close, cube_yaw, chosen)


def desired_ee_quat_aligned_with_cube(
    pick_pos: torch.Tensor,
    pick_quat: torch.Tensor,
    obstacles_pos: torch.Tensor,
) -> torch.Tensor:
    """End-effector quaternion: gripper down, yaw aligned with cube faces, jaws clear of obstacles."""
    yaw = grasp_yaw_avoiding(pick_pos, pick_quat, obstacles_pos)
    return _ee_quat_from_yaw(yaw)


class StackSM:
    """P-control state machine with orientation correction, vibration, and missed grasps.

    Phase 0: pick cube_2 (red) → stack on cube_1 (blue)
    Phase 1: pick cube_3 (green) → stack on top of cube_2
    Then reset.
    """

    def __init__(self, dt, num_envs, device, vibration_pos=0.06, vibration_rot=0.10, miss_rate=0.4):
        self.dt = dt
        self.N = num_envs
        self.dev = device
        self.vibration_pos = vibration_pos
        self.vibration_rot = vibration_rot
        self.miss_rate = miss_rate

        self.state = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.timer = torch.zeros(num_envs, device=device)
        self.phase = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.saved_pick_pos = torch.zeros(num_envs, 3, device=device)
        self.saved_pick_quat = (
            FLIP_QUAT.to(device).unsqueeze(0).expand(num_envs, -1).clone()
        )

        # Track whether each environment will miss its grasp on the current attempt
        self.will_miss = torch.zeros(num_envs, dtype=torch.bool, device=device)
        # Store random offsets for missing attempts so they are consistent during DOWN_PICK
        self.miss_offsets = torch.zeros(num_envs, 3, device=device)

        # Tuning — FASTER trajectories (higher gains + clamps + alpha)
        self.kp = 6.0          # position P-gain (up from 3.0)
        self.kr = 8.0          # orientation P-gain (up from 4.0)
        self.dpos_clamp = 1.20 # max per-step position delta in metres (up from 0.60)
        self.drot_clamp = 2.40 # max per-step axis-angle delta in radians (up from 1.20)
        self.action_alpha = 0.50 # less smoothing (up from 0.35)

        # Per-state arrival thresholds:
        self.dist_th_tight = 0.015  # slightly relaxed to trigger faster transitions
        self.dist_th_loose = 0.06
        self.rot_th = 0.20     # orientation arrival threshold
        self.hover = 0.14      # hover height above target (m)
        self.cube_h = 0.045    # cube side length
        self.place_margin = 0.006
        self.slow_speed_scale = 0.7  # faster descent during pick/place (up from 0.4)

        # Previous action for EMA smoothing.
        self.prev_action = torch.zeros(num_envs, 7, device=device)
        self.prev_action[:, 6] = 1.0

        # Initialize miss flags
        self.reset_idx(torch.arange(num_envs, device=device))

    def reset_idx(self, ids):
        self.state[ids] = S.REST
        self.timer[ids] = 0.0
        self.phase[ids] = 0
        self.saved_pick_pos[ids] = 0.0
        self.saved_pick_quat[ids] = FLIP_QUAT.to(self.dev)
        self.prev_action[ids] = 0.0
        self.prev_action[ids, 6] = 1.0  # gripper open

        # Decide if this new cycle will result in a missed grasp
        rands = torch.rand(len(ids), device=self.dev)
        self.will_miss[ids] = rands < self.miss_rate

        # Generate a random XY offset (e.g. 3 to 4.5 cm in a random direction)
        angles = torch.rand(len(ids), device=self.dev) * 2.0 * math.pi
        dists = 0.03 + 0.015 * torch.rand(len(ids), device=self.dev)
        self.miss_offsets[ids, 0] = dists * torch.cos(angles)
        self.miss_offsets[ids, 1] = dists * torch.sin(angles)
        self.miss_offsets[ids, 2] = -0.005

    def compute(self, ee_pos, ee_quat, c1, c2, c3, c1_q, c2_q, c3_q):
        is_phase0 = (self.phase == 0).unsqueeze(-1)  # (N,1)
        pick = torch.where(is_phase0, c2, c3)
        place = torch.where(is_phase0, c1, c2)
        pick_q = torch.where(is_phase0, c2_q, c3_q)
        obstacles = torch.stack(
            (
                c1,
                torch.where(is_phase0.expand_as(c2), c3, c2),
            ),
            dim=1,
        )  # (N, 2, 3)
        _ = c1_q

        pick_des_quat = desired_ee_quat_aligned_with_cube(pick, pick_q, obstacles)
        in_place_phase = ((self.state >= S.GRASP) & (self.state <= S.RETREAT)).unsqueeze(-1)
        des_quat = torch.where(in_place_phase, self.saved_pick_quat, pick_des_quat)

        target = ee_pos.clone()
        gripper = torch.ones(self.N, device=self.dev)

        for i in range(self.N):
            s = self.state[i].item()
            ph = self.phase[i].item()

            if ph >= 2 and s != S.DONE:
                self.state[i] = S.DONE
                self.timer[i] = 0.0
                s = S.DONE

            p = pick[i]
            t = place[i]
            ch = self.cube_h

            if s == S.REST:
                target[i] = ee_pos[i]
            elif s == S.ABOVE_PICK:
                target[i] = torch.tensor([p[0], p[1], p[2] + self.hover], device=self.dev)
                if self.will_miss[i]:
                    target[i] += self.miss_offsets[i]
            elif s == S.DOWN_PICK:
                target[i] = p.clone()
                if self.will_miss[i]:
                    target[i] += self.miss_offsets[i]
            elif s == S.GRASP:
                target[i] = self.saved_pick_pos[i]
                gripper[i] = -1.0
            elif s == S.LIFT:
                target[i] = self.saved_pick_pos[i].clone()
                target[i, 2] += self.hover
                gripper[i] = -1.0
            elif s == S.ABOVE_PLACE:
                target[i] = torch.tensor([t[0], t[1], t[2] + self.hover + ch], device=self.dev)
                gripper[i] = -1.0
            elif s == S.DOWN_PLACE:
                target[i] = torch.tensor(
                    [t[0], t[1], t[2] + ch + self.place_margin], device=self.dev
                )
                gripper[i] = -1.0
            elif s == S.RELEASE:
                target[i] = torch.tensor(
                    [t[0], t[1], t[2] + ch + self.place_margin], device=self.dev
                )
            elif s == S.RETREAT:
                target[i] = torch.tensor([t[0], t[1], t[2] + self.hover + ch], device=self.dev)
            elif s == S.DONE:
                target[i] = ee_pos[i]

        # === Position: P-control ===
        delta_pos = self.kp * (target - ee_pos)
        delta_pos = torch.clamp(delta_pos, -self.dpos_clamp, self.dpos_clamp)

        slow_mask = (
            (self.state == S.DOWN_PICK)
            | (self.state == S.DOWN_PLACE)
            | (self.state == S.RELEASE)
        )
        speed_scale = (
            1.0 - slow_mask.float() * (1.0 - self.slow_speed_scale)
        ).unsqueeze(-1)
        delta_pos = delta_pos * speed_scale

        # === Orientation: track per-env desired quat ===
        quat_err = math_utils.quat_mul(des_quat, math_utils.quat_conjugate(ee_quat))
        quat_err = torch.where(quat_err[:, 0:1] < 0, -quat_err, quat_err)
        axis_angle_err = math_utils.axis_angle_from_quat(quat_err)
        delta_rot = self.kr * axis_angle_err
        delta_rot = torch.clamp(delta_rot, -self.drot_clamp, self.drot_clamp)
        delta_rot = delta_rot * speed_scale

        # === State transitions ===
        dist = torch.norm(target - ee_pos, dim=-1)
        rot_err_norm = torch.norm(axis_angle_err, dim=-1)
        self.timer += self.dt

        LOOSE = {S.ABOVE_PICK, S.LIFT, S.ABOVE_PLACE, S.RETREAT}
        TIGHT_BOTH = {S.DOWN_PICK, S.DOWN_PLACE}

        for i in range(self.N):
            s = self.state[i].item()
            d = dist[i].item()
            r = rot_err_norm[i].item()
            t = self.timer[i].item()

            advance = False
            if s == S.REST:
                advance = (t >= WAIT[S.REST])
            elif s in LOOSE:
                advance = d < self.dist_th_loose
            elif s in TIGHT_BOTH:
                advance = (d < self.dist_th_tight) and (r < self.rot_th)
            elif s == S.GRASP:
                advance = (t >= WAIT[S.GRASP])
            elif s == S.RELEASE:
                advance = (t >= WAIT[S.RELEASE])
            elif s == S.NEXT:
                advance = True
            elif s == S.DONE:
                advance = False

            if advance:
                if s == S.DOWN_PICK:
                    pick_cube = c2[i] if self.phase[i] == 0 else c3[i]
                    self.saved_pick_pos[i] = pick_cube.clone()
                    if self.will_miss[i]:
                        # Apply the offset to the saved pose so the gripper closes here
                        self.saved_pick_pos[i] += self.miss_offsets[i]
                        print(f"[{i}] ❌ Missed grasp attempt! Offsetting grasp position by {self.miss_offsets[i].cpu().numpy()}.", flush=True)
                    else:
                        print(f"[{i}] ✅ Clean grasp attempt.", flush=True)
                    self.saved_pick_quat[i] = des_quat[i].clone()
                    self.state[i] = S.GRASP
                elif s == S.NEXT:
                    if self.phase[i] == 0:
                        self.phase[i] = 1
                        self.state[i] = S.REST
                        # Roll a new miss decision for phase 1
                        self.will_miss[i] = torch.rand((), device=self.dev) < self.miss_rate
                        angle = torch.rand((), device=self.dev) * 2.0 * math.pi
                        dist_val = 0.03 + 0.015 * torch.rand((), device=self.dev)
                        self.miss_offsets[i, 0] = dist_val * torch.cos(angle)
                        self.miss_offsets[i, 1] = dist_val * torch.sin(angle)
                        self.miss_offsets[i, 2] = -0.005
                    else:
                        self.phase[i] = 2
                        self.state[i] = S.DONE
                elif s == S.RETREAT:
                    self.state[i] = S.NEXT
                else:
                    self.state[i] = s + 1
                self.timer[i] = 0.0

        actions = torch.cat([delta_pos, delta_rot, gripper.unsqueeze(-1)], dim=-1)

        # EMA smoothing
        a = self.action_alpha
        actions[:, :6] = a * actions[:, :6] + (1.0 - a) * self.prev_action[:, :6]
        self.prev_action = actions.clone()

        # Add vibration noise (high-frequency jitter)
        if self.vibration_pos > 0 or self.vibration_rot > 0:
            vibration = torch.randn(self.N, 6, device=self.dev)
            vibration[:, :3] *= self.vibration_pos
            vibration[:, 3:6] *= self.vibration_rot
            actions[:, :6] += vibration

        return actions

    def force_reset_mask(self) -> torch.Tensor:
        return (self.state == S.DONE) & (self.timer >= WAIT[S.DONE])


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    task = "Isaac-Stack-Cube-Franka-IK-Rel-v0"
    env_cfg = parse_env_cfg(
        task, device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    # Shorter episodes so it resets faster
    env_cfg.episode_length_s = 18.0

    print(f"[INFO] Task: {task} | num_envs: {args_cli.num_envs}", flush=True)
    print(f"[INFO] Settings: Vibration Pos={args_cli.vibration_pos:.3f}, Rot={args_cli.vibration_rot:.3f} | Miss Rate={args_cli.miss_rate * 100:.1f}%", flush=True)
    
    env = gym.make(task, cfg=env_cfg)
    print(f"[INFO] Action space: {env.action_space.shape}", flush=True)
    obs, info = env.reset()
    print(f"[INFO] Running! Robot will pick & stack 3 cubes (with jitter & misses), then reset.", flush=True)

    sm = StackSM(
        dt=env_cfg.sim.dt * env_cfg.decimation,
        num_envs=env.unwrapped.num_envs,
        device=env.unwrapped.device,
        vibration_pos=args_cli.vibration_pos,
        vibration_rot=args_cli.vibration_rot,
        miss_rate=args_cli.miss_rate,
    )

    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            ee_frame = env.unwrapped.scene["ee_frame"]
            ee_pos = ee_frame.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
            ee_quat = ee_frame.data.target_quat_w[..., 0, :].clone()

            c1 = env.unwrapped.scene["cube_1"]
            c2 = env.unwrapped.scene["cube_2"]
            c3 = env.unwrapped.scene["cube_3"]
            origins = env.unwrapped.scene.env_origins
            c1_pos = c1.data.root_pos_w - origins
            c2_pos = c2.data.root_pos_w - origins
            c3_pos = c3.data.root_pos_w - origins
            c1_q = c1.data.root_quat_w
            c2_q = c2.data.root_quat_w
            c3_q = c3.data.root_quat_w

            actions = sm.compute(ee_pos, ee_quat, c1_pos, c2_pos, c3_pos, c1_q, c2_q, c3_q)

            obs, rew, terminated, truncated, info = env.step(actions)

            dones = terminated | truncated
            if dones.any():
                done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                sm.reset_idx(done_ids)
                print(f"[{step:5d}] ♻ env-reset: {done_ids.tolist()}", flush=True)

            force = sm.force_reset_mask()
            if force.any():
                force_ids = force.nonzero(as_tuple=False).squeeze(-1)
                env.unwrapped._reset_idx(force_ids)
                sm.reset_idx(force_ids)
                print(f"[{step:5d}] ⏱ SM-forced reset: {force_ids.tolist()}", flush=True)

            step += 1
            if step == 1 or step % 100 == 0:
                sn = [STATE_NAMES[s] for s in sm.state.tolist()]
                ph = sm.phase.tolist()
                print(f"[{step:5d}] {sn} ph={ph}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
