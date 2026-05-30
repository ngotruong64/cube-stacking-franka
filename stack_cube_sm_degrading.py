#!/usr/bin/env python3
"""
Franka Cube Stacking — degrading demo.

Plays out a scripted scenario for video recording, with progressive failure:

  Phase 0 CLEAN  (~25s):  All robots stack cubes accurately, in sync.
  Phase 1 JITTER (~8s):   Motion gradually starts to jitter (action noise ramps up).
  Phase 2 FREEZE (~3s):   Robots briefly freeze; gripper opens → held cubes drop.
  Phase 3 CHAOS  (∞):     Random actions — flailing arms, missed grasps, dropped cubes.

Usage:
    conda activate env_isaaclab
    cd /home/truongnq/Desktop/3cube
    OMNI_KIT_ACCEPT_EULA=YES python stack_cube_sm_degrading.py --num_envs 4
"""

"""Launch Omniverse first."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Franka cube stacking with progressive failure.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--t_clean", type=float, default=25.0, help="seconds of clean stacking before degradation")
parser.add_argument("--t_jitter", type=float, default=8.0, help="seconds of jittering motion")
parser.add_argument("--t_freeze", type=float, default=3.0, help="seconds of freeze + drop")
parser.add_argument("--jitter_max", type=float, default=0.7,
                    help="max axis-angle noise magnitude at the end of the JITTER phase (rad/m)")
parser.add_argument("--chaos_mag", type=float, default=1.0,
                    help="random action magnitude during CHAOS (1.0 = full clamp)")
parser.add_argument("--sync_scene", action=argparse.BooleanOptionalAction, default=True,
                    help="spawn cubes at fixed identical positions across envs so motions look in sync"
                         " (pass --no-sync_scene to disable)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import math
import gymnasium as gym
import torch
import isaaclab.utils.math as math_utils
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


# ── State Machine (copied from stack_cube_sm.py — kept self-contained) ─────

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
    DONE = 10


WAIT = {
    S.REST: 0.1,
    S.GRASP: 0.35,
    S.RELEASE: 0.2,
    S.DONE: 1.0,
}

STATE_NAMES = [
    "REST", "ABOVE_PICK", "DOWN_PICK", "GRASP", "LIFT",
    "ABOVE_PLACE", "DOWN_PLACE", "RELEASE", "RETREAT", "NEXT", "DONE",
]

FLIP_QUAT = torch.tensor([0.0, 1.0, 0.0, 0.0])


def yaw_from_quat(q):
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_quarter(yaw):
    return ((yaw + math.pi / 4.0) % (math.pi / 2.0)) - math.pi / 4.0


def _ee_quat_from_yaw(yaw):
    half = yaw * 0.5
    c, s = torch.cos(half), torch.sin(half)
    zero = torch.zeros_like(c)
    return torch.stack((zero, c, s, zero), dim=-1)


def _angular_dist_mod_pi(a, b):
    d = (a - b + math.pi / 2.0) % math.pi - math.pi / 2.0
    return torch.abs(d)


def grasp_yaw_avoiding(pick_pos, pick_quat, obstacles_pos):
    cube_yaw = wrap_quarter(yaw_from_quat(pick_quat))
    dxy = obstacles_pos[..., :2] - pick_pos[:, None, :2]
    d2 = (dxy * dxy).sum(dim=-1)
    nearest = d2.argmin(dim=-1)
    sel = nearest[:, None, None].expand(-1, 1, 2)
    nearest_xy = obstacles_pos[..., :2].gather(1, sel).squeeze(1)
    min_d = d2.gather(1, nearest[:, None]).squeeze(-1)
    diff = pick_pos[:, :2] - nearest_xy
    pref_yaw = torch.atan2(diff[..., 1], diff[..., 0])
    cand1 = cube_yaw
    cand2 = cube_yaw + math.pi / 2.0
    use1 = _angular_dist_mod_pi(cand1, pref_yaw) <= _angular_dist_mod_pi(cand2, pref_yaw)
    chosen = torch.where(use1, cand1, cand2)
    too_close = min_d < 1.0e-6
    return torch.where(too_close, cube_yaw, chosen)


def desired_ee_quat_aligned_with_cube(pick_pos, pick_quat, obstacles_pos):
    yaw = grasp_yaw_avoiding(pick_pos, pick_quat, obstacles_pos)
    return _ee_quat_from_yaw(yaw)


class StackSM:
    def __init__(self, dt, num_envs, device):
        self.dt = dt
        self.N = num_envs
        self.dev = device
        self.state = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.timer = torch.zeros(num_envs, device=device)
        self.phase = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.saved_pick_pos = torch.zeros(num_envs, 3, device=device)
        self.saved_pick_quat = (
            FLIP_QUAT.to(device).unsqueeze(0).expand(num_envs, -1).clone()
        )
        # Tuning (same as stack_cube_sm.py)
        self.kp = 3.0
        self.kr = 4.0
        self.dpos_clamp = 0.60
        self.drot_clamp = 1.20
        self.action_alpha = 0.35
        self.dist_th_tight = 0.012
        self.dist_th_loose = 0.05
        self.rot_th = 0.15
        self.hover = 0.14
        self.cube_h = 0.045
        self.place_margin = 0.006
        self.slow_speed_scale = 0.4
        self.prev_action = torch.zeros(num_envs, 7, device=device)
        self.prev_action[:, 6] = 1.0

    def reset_idx(self, ids):
        self.state[ids] = S.REST
        self.timer[ids] = 0.0
        self.phase[ids] = 0
        self.saved_pick_pos[ids] = 0.0
        self.saved_pick_quat[ids] = FLIP_QUAT.to(self.dev)
        self.prev_action[ids] = 0.0
        self.prev_action[ids, 6] = 1.0

    def compute(self, ee_pos, ee_quat, c1, c2, c3, c1_q, c2_q, c3_q):
        is_phase0 = (self.phase == 0).unsqueeze(-1)
        pick = torch.where(is_phase0, c2, c3)
        place = torch.where(is_phase0, c1, c2)
        pick_q = torch.where(is_phase0, c2_q, c3_q)
        obstacles = torch.stack(
            (c1, torch.where(is_phase0.expand_as(c2), c3, c2)), dim=1
        )
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
            elif s == S.DOWN_PICK:
                target[i] = p.clone()
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
                target[i] = torch.tensor([t[0], t[1], t[2] + ch + self.place_margin], device=self.dev)
                gripper[i] = -1.0
            elif s == S.RELEASE:
                target[i] = torch.tensor([t[0], t[1], t[2] + ch + self.place_margin], device=self.dev)
            elif s == S.RETREAT:
                target[i] = torch.tensor([t[0], t[1], t[2] + self.hover + ch], device=self.dev)
            elif s == S.DONE:
                target[i] = ee_pos[i]

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

        quat_err = math_utils.quat_mul(des_quat, math_utils.quat_conjugate(ee_quat))
        quat_err = torch.where(quat_err[:, 0:1] < 0, -quat_err, quat_err)
        axis_angle_err = math_utils.axis_angle_from_quat(quat_err)
        delta_rot = self.kr * axis_angle_err
        delta_rot = torch.clamp(delta_rot, -self.drot_clamp, self.drot_clamp)
        delta_rot = delta_rot * speed_scale

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
                    self.saved_pick_quat[i] = des_quat[i].clone()
                    self.state[i] = S.GRASP
                elif s == S.NEXT:
                    if self.phase[i] == 0:
                        self.phase[i] = 1
                        self.state[i] = S.REST
                    else:
                        self.phase[i] = 2
                        self.state[i] = S.DONE
                elif s == S.RETREAT:
                    self.state[i] = S.NEXT
                else:
                    self.state[i] = s + 1
                self.timer[i] = 0.0

        actions = torch.cat([delta_pos, delta_rot, gripper.unsqueeze(-1)], dim=-1)
        a = self.action_alpha
        actions[:, :6] = a * actions[:, :6] + (1.0 - a) * self.prev_action[:, :6]
        self.prev_action = actions.clone()
        return actions

    def force_reset_mask(self) -> torch.Tensor:
        return (self.state == S.DONE) & (self.timer >= WAIT[S.DONE])


# ── Action degrader ────────────────────────────────────────────────────────

class Phase:
    CLEAN = "CLEAN"
    JITTER = "JITTER"
    FREEZE = "FREEZE"
    CHAOS = "CHAOS"


class ActionDegrader:
    """Transforms the clean SM actions to reproduce the scripted failure arc.

    Phase boundaries are wall-clock (simulated). The transformations are:

      CLEAN:   pass-through.
      JITTER:  add Gaussian noise that ramps from 0 → jitter_max over the phase.
      FREEZE:  zero out arm deltas + force gripper open (any held cube falls).
      CHAOS:   replace actions with sampled random deltas (independent per env).
    """

    def __init__(self, t_clean, t_jitter, t_freeze, jitter_max, chaos_mag, num_envs, device):
        self.t_clean = t_clean
        self.t_jitter = t_jitter
        self.t_freeze = t_freeze
        self.jitter_max = jitter_max
        self.chaos_mag = chaos_mag
        self.N = num_envs
        self.dev = device
        self.t = 0.0

    def step(self, dt):
        self.t += dt

    def phase(self) -> str:
        if self.t < self.t_clean:
            return Phase.CLEAN
        if self.t < self.t_clean + self.t_jitter:
            return Phase.JITTER
        if self.t < self.t_clean + self.t_jitter + self.t_freeze:
            return Phase.FREEZE
        return Phase.CHAOS

    def phase_progress(self) -> float:
        """0 → 1 progress through the current phase. Useful for ramps."""
        p = self.phase()
        if p == Phase.CLEAN:
            return min(1.0, self.t / max(self.t_clean, 1e-6))
        if p == Phase.JITTER:
            return min(1.0, (self.t - self.t_clean) / max(self.t_jitter, 1e-6))
        if p == Phase.FREEZE:
            return min(1.0, (self.t - self.t_clean - self.t_jitter) / max(self.t_freeze, 1e-6))
        # CHAOS: no upper bound
        return 1.0

    def apply(self, actions: torch.Tensor) -> torch.Tensor:
        ph = self.phase()
        if ph == Phase.CLEAN:
            return actions

        if ph == Phase.JITTER:
            # Gaussian noise on the 6 arm channels, ramping from 0 to jitter_max.
            # The gripper channel is kept clean here so the cube is still gripped
            # during the "shaky" approach — failures look organic.
            sev = self.phase_progress()
            sigma = self.jitter_max * sev
            noise = torch.randn(self.N, 6, device=self.dev) * sigma
            out = actions.clone()
            out[:, :6] = out[:, :6] + noise
            return out

        if ph == Phase.FREEZE:
            # Zero arm motion, force gripper open → cubes fall from any held hand.
            out = torch.zeros_like(actions)
            out[:, 6] = 1.0  # open gripper
            return out

        # CHAOS: random arm motion, random gripper toggles. Magnitude tuned so it
        # looks like "flailing" rather than tiny twitches.
        out = (2 * torch.rand_like(actions) - 1) * self.chaos_mag
        # Snap gripper to ±1 with a 30% close probability per step.
        grip_rand = torch.rand(self.N, device=self.dev)
        out[:, 6] = torch.where(grip_rand < 0.3, -1.0, 1.0)
        return out


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    task = "Isaac-Stack-Cube-Franka-IK-Rel-v0"
    env_cfg = parse_env_cfg(
        task, device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    # Long episode — we control the timing ourselves via the degrader. Without
    # this, the env would auto-reset every 18 s during the CHAOS phase and erase
    # the visual mess we just made.
    env_cfg.episode_length_s = 10_000.0

    # Disable the env's automatic terminations so they don't reset envs while
    # we're showing failures (a cube falling off the table during CHAOS would
    # normally trigger cube_dropping → reset). We drive resets ourselves during
    # CLEAN via sm.force_reset_mask().
    for name in ("cube_1_dropping", "cube_2_dropping", "cube_3_dropping", "success"):
        if hasattr(env_cfg.terminations, name):
            setattr(env_cfg.terminations, name, None)

    # Force identical scenes across envs so the robots look perfectly in sync
    # during the CLEAN phase. We do this by overriding the cube-randomization
    # event's pose_range to zero variance and disabling the joint randomization.
    if args_cli.sync_scene:
        try:
            env_cfg.events.randomize_cube_positions.params["pose_range"] = {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "yaw": (0.0, 0.0),
            }
            env_cfg.events.randomize_cube_positions.params["min_separation"] = 0.0
        except AttributeError:
            pass
        try:
            # Tiny joint noise so robots start from the same default pose.
            env_cfg.events.randomize_franka_joint_state.params["std"] = 0.0
        except AttributeError:
            pass

    print(f"[INFO] Task: {task} | num_envs: {args_cli.num_envs}", flush=True)
    print(
        f"[INFO] Scenario: CLEAN {args_cli.t_clean:.0f}s → JITTER {args_cli.t_jitter:.0f}s"
        f" → FREEZE {args_cli.t_freeze:.0f}s → CHAOS (until quit)",
        flush=True,
    )
    env = gym.make(task, cfg=env_cfg)
    print(f"[INFO] Action space: {env.action_space.shape}", flush=True)
    env.reset()

    dt = env_cfg.sim.dt * env_cfg.decimation
    sm = StackSM(dt=dt, num_envs=env.unwrapped.num_envs, device=env.unwrapped.device)
    deg = ActionDegrader(
        t_clean=args_cli.t_clean,
        t_jitter=args_cli.t_jitter,
        t_freeze=args_cli.t_freeze,
        jitter_max=args_cli.jitter_max,
        chaos_mag=args_cli.chaos_mag,
        num_envs=env.unwrapped.num_envs,
        device=env.unwrapped.device,
    )

    step = 0
    last_phase = None
    while simulation_app.is_running():
        with torch.inference_mode():
            ee_frame = env.unwrapped.scene["ee_frame"]
            origins = env.unwrapped.scene.env_origins
            ee_pos = ee_frame.data.target_pos_w[..., 0, :].clone() - origins
            ee_quat = ee_frame.data.target_quat_w[..., 0, :].clone()

            c1 = env.unwrapped.scene["cube_1"]
            c2 = env.unwrapped.scene["cube_2"]
            c3 = env.unwrapped.scene["cube_3"]
            c1_pos = c1.data.root_pos_w - origins
            c2_pos = c2.data.root_pos_w - origins
            c3_pos = c3.data.root_pos_w - origins
            c1_q = c1.data.root_quat_w
            c2_q = c2.data.root_quat_w
            c3_q = c3.data.root_quat_w

            clean_actions = sm.compute(ee_pos, ee_quat, c1_pos, c2_pos, c3_pos, c1_q, c2_q, c3_q)
            actions = deg.apply(clean_actions)

            obs, rew, terminated, truncated, info = env.step(actions)
            deg.step(dt)

            # Print phase transitions.
            ph = deg.phase()
            if ph != last_phase:
                banner = {
                    Phase.CLEAN:  "▶ CLEAN — robots stack cubes in sync",
                    Phase.JITTER: "▶ JITTER — motion starts to shake",
                    Phase.FREEZE: "▶ FREEZE — arms stop, grippers release",
                    Phase.CHAOS:  "▶ CHAOS — random flailing",
                }[ph]
                print(f"[t={deg.t:6.2f}s, step={step:5d}] {banner}", flush=True)
                last_phase = ph

            # During CLEAN: keep the demo loop alive by force-resetting envs that
            # finished a 3-cube stack. During the degrading phases we DO NOT
            # reset — the failures are the show.
            if ph == Phase.CLEAN:
                dones = terminated | truncated
                if dones.any():
                    done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                    sm.reset_idx(done_ids)
                force = sm.force_reset_mask()
                if force.any():
                    force_ids = force.nonzero(as_tuple=False).squeeze(-1)
                    env.unwrapped._reset_idx(force_ids)
                    sm.reset_idx(force_ids)

            step += 1
            if step % 200 == 0:
                sn = [STATE_NAMES[s] for s in sm.state.tolist()]
                print(f"[t={deg.t:6.2f}s] phase={ph} sm={sn}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
