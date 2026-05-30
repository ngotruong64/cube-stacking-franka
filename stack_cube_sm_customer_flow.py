#!/usr/bin/env python3
"""
Franka Cube Stacking Customer Flow — expert motion and failure samples.

Choreographed Scenario:
- Stage 1 (0s to 30s): Expert stack/unstack motion.
- Stage 2 (30s to 80s): Unstable stacking with wrong placement, then collapse.
- Stage 3 (80s to 110s): Repeated cube drops from different heights.
- Stage 4 (110s to 140s): Abnormal arm behavior: turn away, scrape table, random gripper.
- At 140s, the entire scenario resets back to Stage 1.

Usage:
    conda activate env_isaaclab
    cd /home/truongnq/Desktop/3cube
    python stack_cube_sm_customer_flow.py --num_envs 4
"""

"""Launch Omniverse first."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Multi-env Franka choreographed demo.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--debug_env_logs", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import math
import gymnasium as gym
import torch
import isaaclab.sim as sim_utils
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

CARRYING_STATES = {S.GRASP, S.LIFT, S.ABOVE_PLACE, S.DOWN_PLACE}

SCENARIO_STAGE_NAMES = {
    1: "Stage 1 EXPERT | clean stack/unstack motion",
    2: "Stage 2 UNSTABLE STACK | wrong placement, wobble, collapse",
    3: "Stage 3 REPEATED DROPS | varied drop heights",
    4: "Stage 4 ABNORMAL ARM | turn away + table scrape",
}

FLIP_QUAT = torch.tensor([0.0, 1.0, 0.0, 0.0])


def scenario_stage(global_time: float) -> tuple[int, str]:
    if global_time < 30.0:
        stage = 1
    elif global_time < 80.0:
        stage = 2
    elif global_time < 110.0:
        stage = 3
    else:
        stage = 4
    return stage, SCENARIO_STAGE_NAMES[stage]


def format_env_ids(env_ids, max_count: int = 12) -> str:
    values = env_ids.detach().cpu().tolist() if isinstance(env_ids, torch.Tensor) else list(env_ids)
    shown = values[:max_count]
    suffix = "" if len(values) <= max_count else f"...(+{len(values) - max_count})"
    return f"{len(values)} envs {shown}{suffix}"


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


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of quaternion q (N, 4) in wxyz format."""
    w, x, y, z = q.unbind(-1)
    return torch.stack((w, -x, -y, -z), dim=-1)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v (N, 3) by quaternion q (N, 4) in wxyz format."""
    w, x, y, z = q.unbind(-1)
    q_xyz = torch.stack((x, y, z), dim=-1)
    uv = torch.cross(q_xyz, v, dim=-1)
    uuv = torch.cross(q_xyz, uv, dim=-1)
    return v + 2.0 * (uv * w.unsqueeze(-1) + uuv)


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
    """P-control state machine with orientation correction and alignment."""

    def __init__(self, dt, num_envs, device, debug_env_logs=False):
        self.dt = dt
        self.N = num_envs
        self.dev = device
        self.debug_env_logs = debug_env_logs

        # Global scenario timer
        self.global_time = 0.0

        self.state = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.timer = torch.zeros(num_envs, device=device)
        self.phase = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.saved_pick_pos = torch.zeros(num_envs, 3, device=device)
        self.saved_pick_quat = (
            FLIP_QUAT.to(device).unsqueeze(0).expand(num_envs, -1).clone()
        )

        # Track original positions and orientations of cube 2 and cube 3 on the table
        self.c2_init = torch.zeros(num_envs, 3, device=device)
        self.c3_init = torch.zeros(num_envs, 3, device=device)
        self.c2_init_q = torch.zeros(num_envs, 4, device=device)
        self.c3_init_q = torch.zeros(num_envs, 4, device=device)
        self.c2_init_q[:, 0] = 1.0
        self.c3_init_q[:, 0] = 1.0
        self.init_saved = torch.zeros(num_envs, dtype=torch.bool, device=device)

        # Stage 3 repeated-drop timer per env
        self.freeze_timer = torch.zeros(num_envs, device=device)
        self.freeze_duration = 1.2
        self.freeze_trigger_prob = 0.045
        self.freeze_lift_duration = 0.45
        self.freeze_lift_z = 0.06
        self.drop_height_by_env = torch.zeros(num_envs, device=device)
        self.drop_height_options = torch.tensor([0.055, 0.095, 0.145], device=device)
        self.freeze_gripper_action = 1.0

        # Grasp offsets at grasp time:
        self.grasp_offset_yaw = torch.zeros(num_envs, device=device)
        self.grasp_offset_pos = torch.zeros(num_envs, 3, device=device)

        # Tuning parameters (same as original stack_cube_sm)
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

        # Previous action for EMA smoothing.
        self.prev_action = torch.zeros(num_envs, 7, device=device)
        self.prev_action[:, 6] = 1.0

    def reset_idx(self, ids):
        self.state[ids] = S.REST
        self.timer[ids] = 0.0
        self.phase[ids] = 0
        self.saved_pick_pos[ids] = 0.0
        self.saved_pick_quat[ids] = FLIP_QUAT.to(self.dev)
        self.prev_action[ids] = 0.0
        self.prev_action[ids, 6] = 1.0  # gripper open
        self.init_saved[ids] = False
        self.grasp_offset_yaw[ids] = 0.0
        self.grasp_offset_pos[ids] = 0.0
        self.freeze_timer[ids] = 0.0
        self.drop_height_by_env[ids] = 0.0

    def compute(self, ee_pos, ee_quat, c1, c2, c3, c1_q, c2_q, c3_q):
        # Update global timer
        self.global_time += self.dt

        # Decay repeated-drop timer
        self.freeze_timer = torch.clamp(self.freeze_timer - self.dt, min=0.0)
        current_stage, _ = scenario_stage(self.global_time)

        # Save initial positions once per episode when env resets
        for i in range(self.N):
            if not self.init_saved[i]:
                self.c2_init[i] = c2[i].clone()
                self.c3_init[i] = c3[i].clone()
                self.c2_init_q[i] = c2_q[i].clone()
                self.c3_init_q[i] = c3_q[i].clone()
                self.init_saved[i] = True

        # Stage 3: repeatedly lift and drop cubes from varied heights while gripping.
        if current_stage == 3:
            for i in range(self.N):
                if self.freeze_timer[i] == 0.0:
                    s = self.state[i].item()
                    if s in CARRYING_STATES:
                        if torch.rand(1, device=self.dev).item() < self.freeze_trigger_prob:
                            height_idx = (i + int(self.global_time * 2.0)) % len(self.drop_height_options)
                            drop_height = self.drop_height_options[height_idx]
                            self.drop_height_by_env[i] = drop_height
                            self.freeze_timer[i] = self.freeze_duration
                            print(
                                f"[Stage 3 REPEATED DROPS] env={i} state={STATE_NAMES[s]} "
                                f"phase={self.phase[i].item()} lift_z={drop_height.item():.3f}m; "
                                "release after lift.",
                                flush=True,
                            )

        freeze_mask = self.freeze_timer > 0.0
        freeze_lift_mask = freeze_mask & (self.freeze_timer > self.freeze_duration - self.freeze_lift_duration)
        freeze_release_mask = freeze_mask & ~freeze_lift_mask
        drop_z = torch.where(
            self.drop_height_by_env > 0.0,
            self.drop_height_by_env,
            torch.full_like(self.drop_height_by_env, self.freeze_lift_z),
        )

        # Vectorized target calculation based on phase
        is_ph0 = (self.phase == 0).unsqueeze(-1)
        is_ph1 = (self.phase == 1).unsqueeze(-1)
        is_ph2 = (self.phase == 2).unsqueeze(-1)
        is_ph3 = (self.phase == 3).unsqueeze(-1)

        pick = (
            is_ph0.float() * c2 +
            is_ph1.float() * c3 +
            is_ph2.float() * c3 +
            is_ph3.float() * c2
        )
        place = (
            is_ph0.float() * c1 +
            is_ph1.float() * c2 +
            is_ph2.float() * self.c3_init +
            is_ph3.float() * self.c2_init
        )
        pick_q = (
            is_ph0.float() * c2_q +
            is_ph1.float() * c3_q +
            is_ph2.float() * c3_q +
            is_ph3.float() * c2_q
        )
        ch_add = (is_ph0.float() + is_ph1.float()).squeeze(-1) * self.cube_h

        # Obstacles for yaw selection:
        is_c2_pick = ((self.phase == 0) | (self.phase == 3)).unsqueeze(-1).unsqueeze(-1)
        obstacles = torch.where(
            is_c2_pick,
            torch.stack((c1, c3), dim=1),
            torch.stack((c1, c2), dim=1)
        )

        # Place target orientations depending on phase
        place_q = (
            is_ph0.float() * c1_q +
            is_ph1.float() * c2_q +
            is_ph2.float() * self.c3_init_q +
            is_ph3.float() * self.c2_init_q
        )

        pick_des_quat = desired_ee_quat_aligned_with_cube(pick, pick_q, obstacles)
        
        # Place desired orientation: target orientation + grasp offset yaw
        place_yaw = yaw_from_quat(place_q)
        place_des_yaw = place_yaw + self.grasp_offset_yaw
        place_des_quat = _ee_quat_from_yaw(place_des_yaw)

        # Select target orientation based on state
        is_pick_state = (self.state < S.GRASP).unsqueeze(-1)
        is_lift_state = ((self.state >= S.GRASP) & (self.state < S.ABOVE_PLACE)).unsqueeze(-1)
        des_quat = torch.where(
            is_pick_state,
            pick_des_quat,
            torch.where(
                is_lift_state,
                self.saved_pick_quat,
                place_des_quat
            )
        )

        # Compute translation offset in world frame
        offset_world_all = quat_rotate(des_quat, self.grasp_offset_pos)

        target = ee_pos.clone()
        gripper = torch.ones(self.N, device=self.dev)

        for i in range(self.N):
            s = self.state[i].item()
            ph = self.phase[i].item()

            if ph >= 4 and s != S.DONE:
                self.state[i] = S.DONE
                self.timer[i] = 0.0
                s = S.DONE

            p = pick[i]
            t = place[i]
            ch = ch_add[i].item()
            off_w = offset_world_all[i]

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
                target[i] = torch.tensor([t[0], t[1], t[2] + self.hover + ch], device=self.dev) + off_w
                gripper[i] = -1.0
            elif s == S.DOWN_PLACE:
                target[i] = torch.tensor(
                    [t[0], t[1], t[2] + ch + self.place_margin], device=self.dev
                ) + off_w
                gripper[i] = -1.0
            elif s == S.RELEASE:
                target[i] = torch.tensor(
                    [t[0], t[1], t[2] + ch + self.place_margin], device=self.dev
                ) + off_w
            elif s == S.RETREAT:
                target[i] = torch.tensor([t[0], t[1], t[2] + self.hover + ch], device=self.dev) + off_w
            elif s == S.DONE:
                target[i] = ee_pos[i]

        # ── Customer failure flow overrides ────────────────────────────────────────

        if current_stage == 2:
            for i in range(self.N):
                s = self.state[i].item()
                if s in {S.ABOVE_PLACE, S.DOWN_PLACE, S.RELEASE, S.RETREAT}:
                    wobble = self.global_time * 1.35 + i * 0.41 + self.phase[i].item() * 0.9
                    target[i, 0] += 0.045 * math.sin(wobble)
                    target[i, 1] += 0.035 * math.cos(wobble * 0.73)
                    if s in {S.DOWN_PLACE, S.RELEASE}:
                        target[i, 2] += 0.018
                if s in {S.DOWN_PLACE, S.RELEASE} and (int(self.global_time * 4.0) + i) % 17 == 0:
                    gripper[i] = 1.0

        for i in range(self.N):
            if freeze_mask[i]:
                target[i] = ee_pos[i].clone()
                if freeze_lift_mask[i]:
                    target[i, 2] += drop_z[i]
                    gripper[i] = -1.0
                else:
                    gripper[i] = self.freeze_gripper_action

        if current_stage == 4:
            t_wave = self.global_time
            for i in range(self.N):
                mode = (i + int((t_wave - 110.0) / 4.0)) % 3
                if mode == 0:
                    # Turn away from the table.
                    side = -1.0 if i % 2 == 0 else 1.0
                    target[i, 0] = 0.25 + 0.08 * math.sin(t_wave * 1.7 + i)
                    target[i, 1] = side * (0.32 + 0.06 * math.cos(t_wave * 1.3 + i))
                    target[i, 2] = 0.22 + 0.05 * math.sin(t_wave * 2.0 + i)
                elif mode == 1:
                    # Rub/scrape the table surface.
                    target[i, 0] = 0.50 + 0.16 * math.sin(t_wave * 2.2 + i * 0.3)
                    target[i, 1] = 0.00 + 0.14 * math.cos(t_wave * 2.0 + i * 0.5)
                    target[i, 2] = 0.035 + 0.010 * math.sin(t_wave * 6.0 + i)
                else:
                    # Erratic waving over the workspace.
                    target[i, 0] += 0.16 * math.sin(t_wave * 3.0 + i * 0.5)
                    target[i, 1] += 0.16 * math.cos(t_wave * 3.5 + i * 0.7)
                    target[i, 2] += 0.10 * math.sin(t_wave * 2.0 + i * 1.1)

                if (int(t_wave * 10) + i) % 13 == 0:
                    gripper[i] = 1.0
                elif (int(t_wave * 10) + i) % 13 == 6:
                    gripper[i] = -1.0

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

        # ── Action Jitter Overrides ───────────────────────────────────────────────

        if current_stage == 2:
            # Unstable stack: visible motion error plus wrong placement offsets above.
            pos_noise = torch.randn_like(delta_pos) * 0.08
            rot_noise = torch.randn_like(delta_rot) * 0.18
            delta_pos += pos_noise
            delta_rot += rot_noise
        elif current_stage == 3:
            # Repeated drops: light jitter, with drop events providing the main failure.
            pos_noise = torch.randn_like(delta_pos) * 0.04
            rot_noise = torch.randn_like(delta_rot) * 0.10
            delta_pos += pos_noise
            delta_rot += rot_noise
        elif current_stage == 4:
            # Abnormal arm behavior: high chaotic jitter on top of mode-specific target overrides.
            pos_noise = torch.randn_like(delta_pos) * 0.22
            rot_noise = torch.randn_like(delta_rot) * 0.50
            delta_pos += pos_noise
            delta_rot += rot_noise

        if freeze_mask.any():
            # Remove jitter and EMA carry-over from frozen envs.
            # The first slice lifts Z a bit; the rest sends zero pose action to hold joint posture.
            delta_pos[freeze_mask] = 0.0
            delta_rot[freeze_mask] = 0.0
            if freeze_lift_mask.any():
                delta_pos[freeze_lift_mask, 2] = self.kp * drop_z[freeze_lift_mask]
                gripper[freeze_lift_mask] = -1.0
            if freeze_release_mask.any():
                gripper[freeze_release_mask] = self.freeze_gripper_action

        # === State transitions ===
        dist = torch.norm(target - ee_pos, dim=-1)
        rot_err_norm = torch.norm(axis_angle_err, dim=-1)

        # Keep the stack/unstack state machine running throughout the customer flow.
        if self.global_time < 140.0:
            self.timer += self.dt

            LOOSE_POS_ONLY = {S.LIFT, S.RETREAT}
            LOOSE_WITH_ROT = {S.ABOVE_PICK, S.ABOVE_PLACE}
            TIGHT_BOTH = {S.DOWN_PICK, S.DOWN_PLACE}

            for i in range(self.N):
                # Do not transition if frozen
                if freeze_mask[i]:
                    continue

                s = self.state[i].item()
                d = dist[i].item()
                r = rot_err_norm[i].item()
                t = self.timer[i].item()

                advance = False
                if s == S.REST:
                    advance = (t >= WAIT[S.REST])
                elif s in LOOSE_POS_ONLY:
                    advance = d < self.dist_th_loose
                elif s in LOOSE_WITH_ROT:
                    advance = (d < self.dist_th_loose) and (r < self.rot_th)
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
                        ph = self.phase[i].item()
                        if ph == 0:
                            pick_cube = c2[i]
                            pick_cube_q = c2_q[i]
                        elif ph == 1:
                            pick_cube = c3[i]
                            pick_cube_q = c3_q[i]
                        elif ph == 2:
                            pick_cube = c3[i]
                            pick_cube_q = c3_q[i]
                        elif ph == 3:
                            pick_cube = c2[i]
                            pick_cube_q = c2_q[i]
                        
                        self.saved_pick_pos[i] = ee_pos[i].clone()
                        self.saved_pick_quat[i] = des_quat[i].clone()
                        
                        gripper_yaw = yaw_from_quat(des_quat[i].unsqueeze(0)).item()
                        cube_yaw = yaw_from_quat(pick_cube_q.unsqueeze(0)).item()
                        self.grasp_offset_yaw[i] = gripper_yaw - cube_yaw
                        
                        offset_world = ee_pos[i] - pick_cube
                        q_conj = quat_conjugate(des_quat[i].unsqueeze(0))
                        offset_local = quat_rotate(q_conj, offset_world.unsqueeze(0)).squeeze(0)
                        self.grasp_offset_pos[i] = offset_local
                        
                        self.state[i] = S.GRASP
                        if self.debug_env_logs:
                            print(
                                f"[Env {i}] Grasping cube in phase {ph} at "
                                f"{self.saved_pick_pos[i].cpu().numpy()} | "
                                f"offset yaw={self.grasp_offset_yaw[i]:.3f} | "
                                f"offset pos={self.grasp_offset_pos[i].cpu().numpy()}",
                                flush=True,
                            )
                    elif s == S.NEXT:
                        ph = self.phase[i].item()
                        if ph < 3:
                            self.phase[i] = ph + 1
                            self.state[i] = S.REST
                            if self.debug_env_logs:
                                print(f"[Env {i}] Transitioning to Phase {ph+1}", flush=True)
                        else:
                            self.phase[i] = 4
                            self.state[i] = S.DONE
                            if self.debug_env_logs:
                                print(f"[Env {i}] Stacking & Unstacking complete! State: DONE.", flush=True)
                        self.timer[i] = 0.0
                    elif s == S.RETREAT:
                        self.state[i] = S.NEXT
                        self.timer[i] = 0.0
                    else:
                        self.state[i] = s + 1
                        self.timer[i] = 0.0

        actions = torch.cat([delta_pos, delta_rot, gripper.unsqueeze(-1)], dim=-1)

        a = self.action_alpha
        actions[:, :6] = a * actions[:, :6] + (1.0 - a) * self.prev_action[:, :6]
        if freeze_mask.any():
            actions[freeze_mask, :6] = 0.0
            if freeze_lift_mask.any():
                actions[freeze_lift_mask, 2] = self.kp * drop_z[freeze_lift_mask]
        self.prev_action = actions.clone()
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

    # Keep the built-in timeout beyond the 140s customer flow; we reset manually.
    env_cfg.episode_length_s = 160.0

    # Colors in LINEAR color space (converted from sRGB).
    # sRGB → linear: ((x + 0.055) / 1.055) ^ 2.4  (for x > 0.04045)
    cube_colors = {
        "cube_1": (0.0, 0.233, 1.0),   # sRGB #0084ff → linear
        "cube_2": (1.0, 0.0, 0.0),     # #ff0000 (unchanged, 0 and 1 are same in both spaces)
        "cube_3": (1.0, 1.0, 0.0),     # #ffff00 (unchanged)
    }
    for cube_name, color in cube_colors.items():
        getattr(env_cfg.scene, cube_name).spawn.visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=color,
            roughness=0.45,
        )

    # Disable standard terminations completely (we will handle resets manually)
    if hasattr(env_cfg, "terminations"):
        if hasattr(env_cfg.terminations, "success"):
            env_cfg.terminations.success = None
        for term_name in ["cube_1_dropping", "cube_2_dropping", "cube_3_dropping"]:
            if hasattr(env_cfg.terminations, term_name):
                setattr(env_cfg.terminations, term_name, None)

    print(f"[INFO] Task: {task} | num_envs: {args_cli.num_envs}", flush=True)
    env = gym.make(task, cfg=env_cfg)
    print(f"[INFO] Action space: {env.action_space.shape}", flush=True)
    obs, info = env.reset()
    print(f"[INFO] Running! Choreographed demo starting.", flush=True)
    for stage_id in range(1, 5):
        print(f"[INFO] {SCENARIO_STAGE_NAMES[stage_id]}", flush=True)

    sm = StackSM(
        dt=env_cfg.sim.dt * env_cfg.decimation,
        num_envs=env.unwrapped.num_envs,
        device=env.unwrapped.device,
        debug_env_logs=args_cli.debug_env_logs,
    )

    step = 0
    last_stage_num = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # Check scenario reset
            if sm.global_time >= 140.0:
                sm.global_time = 0.0
                env.unwrapped._reset_idx(torch.arange(sm.N, device=sm.dev))
                sm.reset_idx(torch.arange(sm.N, device=sm.dev))
                last_stage_num = 0
                print(f"[{step:5d}] Customer flow complete at 140s. Resetting to Stage 1.", flush=True)

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
            stage_num, stage_name = scenario_stage(sm.global_time)
            if stage_num != last_stage_num:
                print(f"[{step:5d}] >>> {stage_name} (t={sm.global_time:.1f}s) <<<", flush=True)
                last_stage_num = stage_num

            obs, rew, terminated, truncated, info = env.step(actions)

            # Manual Reset detection when cube drops off the table
            # Since the table surface is at z = 0.0, any cube z center < 0.01 means it has fallen off the table
            cube_dropped = (c1_pos[:, 2] < 0.01) | (c2_pos[:, 2] < 0.01) | (c3_pos[:, 2] < 0.01)

            # Expert and repeated-drop sections reset cleanly; collapse/abnormal sections keep failures visible.
            if stage_num in (1, 3):
                reset_mask = cube_dropped | sm.force_reset_mask()
            else:
                reset_mask = torch.zeros(sm.N, dtype=torch.bool, device=sm.dev)

            if reset_mask.any():
                reset_ids = reset_mask.nonzero(as_tuple=False).squeeze(-1)
                env.unwrapped._reset_idx(reset_ids)
                sm.reset_idx(reset_ids)
                print(f"[{step:5d}] manual-reset: {format_env_ids(reset_ids)}", flush=True)

            dones = terminated | truncated
            if dones.any():
                done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                sm.reset_idx(done_ids)
                print(f"[{step:5d}] env-reset: {format_env_ids(done_ids)}", flush=True)

            step += 1
            if step == 1 or step % 100 == 0:
                sn = [STATE_NAMES[s] for s in sm.state.tolist()]
                ph = sm.phase.tolist()
                _, stage_name = scenario_stage(sm.global_time)
                frozen_envs = (sm.freeze_timer > 0.0).nonzero(as_tuple=False).squeeze(-1).tolist()
                print(
                    f"[{step:5d}] t={sm.global_time:.1f}s | {stage_name} | "
                    f"states={sn} ph={ph} frozen_envs={frozen_envs}",
                    flush=True,
                )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
