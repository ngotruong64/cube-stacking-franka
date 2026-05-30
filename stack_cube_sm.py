#!/usr/bin/env python3
"""
Franka Cube Stacking Demo — Multi-env with state machine.

Looks like RL training: multiple parallel envs, each robot picks & stacks
3 cubes, then resets and repeats. Purely scripted for demo recording.

Usage:
    conda activate env_isaaclab
    cd ~/IsaacLab
    python ~/Desktop/3cube/stack_cube_sm.py --num_envs 4
"""

"""Launch Omniverse first."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Multi-env Franka cube stacking demo.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

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
    S.REST: 0.1,
    S.GRASP: 0.35,   # just enough for fingers to close on the cube
    S.RELEASE: 0.2,  # let the cube settle on the stack
    S.DONE: 1.0,     # hold 1 s after the last release, then force-reset the env
}

STATE_NAMES = [
    "REST", "ABOVE_PICK", "DOWN_PICK", "GRASP", "LIFT",
    "ABOVE_PLACE", "DOWN_PLACE", "RELEASE", "RETREAT", "NEXT", "DONE",
]

# Base gripper-down orientation: 180° rotation about x-axis = (w=0, x=1, y=0, z=0).
# We compose this with a yaw rotation about world-Z so the parallel jaws clamp on
# the cube's faces instead of its corners.
FLIP_QUAT = torch.tensor([0.0, 1.0, 0.0, 0.0])

import math


def yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Extract yaw (rotation about world z) from a (N, 4) wxyz quaternion."""
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_quarter(yaw: torch.Tensor) -> torch.Tensor:
    """Wrap yaw to (-pi/4, pi/4]. A cube has 4-fold symmetry around its vertical axis,
    so we only need to rotate the gripper by at most ±45° to align with its faces."""
    return ((yaw + math.pi / 4.0) % (math.pi / 2.0)) - math.pi / 4.0


def _ee_quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    """q_des = q_yaw(yaw) * q_flip = (0, cos(yaw/2), sin(yaw/2), 0). See derivation in module docstring."""
    half = yaw * 0.5
    c, s = torch.cos(half), torch.sin(half)
    zero = torch.zeros_like(c)
    return torch.stack((zero, c, s, zero), dim=-1)


def _angular_dist_mod_pi(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Shortest distance between two angles modulo π (gripper has 2-fold symmetry)."""
    d = (a - b + math.pi / 2.0) % math.pi - math.pi / 2.0
    return torch.abs(d)


def grasp_yaw_avoiding(
    pick_pos: torch.Tensor,
    pick_quat: torch.Tensor,
    obstacles_pos: torch.Tensor,
) -> torch.Tensor:
    """Pick the best gripper yaw to grasp a cube while keeping jaws away from obstacles.

    Args:
        pick_pos:       (N, 3) cube to pick.
        pick_quat:      (N, 4) cube quaternion (wxyz).
        obstacles_pos:  (N, K, 3) up to K obstacle positions (typically other cubes).

    Returns:
        (N,) gripper yaw such that the jaws are roughly perpendicular to the
        direction from the nearest obstacle. Snapped to either cube_yaw or
        cube_yaw + π/2 so the jaws still align with cube faces.
    """
    # Yaw of cube wrapped to (-π/4, π/4]; due to 4-fold cube symmetry this is one
    # of the canonical grasp orientations.
    cube_yaw = wrap_quarter(yaw_from_quat(pick_quat))

    # Pick the NEAREST obstacle (xy distance).
    dxy = obstacles_pos[..., :2] - pick_pos[:, None, :2]  # (N, K, 2)
    d2 = (dxy * dxy).sum(dim=-1)                          # (N, K)
    nearest = d2.argmin(dim=-1)                           # (N,)
    sel = nearest[:, None, None].expand(-1, 1, 2)         # (N, 1, 2)
    nearest_xy = obstacles_pos[..., :2].gather(1, sel).squeeze(1)  # (N, 2)
    min_d = d2.gather(1, nearest[:, None]).squeeze(-1)    # (N,)

    # Preferred yaw: jaws perpendicular to the obstacle→pick direction. From the
    # derivation in the module docstring, jaw direction at yaw φ is at angle
    # (φ - π/2) from world +X, so setting jaws perpendicular to direction θ_d
    # gives φ ≡ θ_d (mod π).
    diff = pick_pos[:, :2] - nearest_xy
    pref_yaw = torch.atan2(diff[..., 1], diff[..., 0])

    # The two valid grasp orientations on the cube due to its 4-fold symmetry.
    cand1 = cube_yaw
    cand2 = cube_yaw + math.pi / 2.0

    # Pick the candidate whose jaw line is closest to perpendicular w.r.t. the
    # obstacle direction.
    use1 = _angular_dist_mod_pi(cand1, pref_yaw) <= _angular_dist_mod_pi(cand2, pref_yaw)
    chosen = torch.where(use1, cand1, cand2)

    # If the nearest "obstacle" is essentially the pick cube itself (e.g. all
    # obstacles coincide with the pick — shouldn't happen but be safe), fall
    # back to plain cube yaw.
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
    """P-control state machine with orientation correction.

    Phase 0: pick cube_2 (red) → stack on cube_1 (blue)
    Phase 1: pick cube_3 (green) → stack on top of cube_2
    Then reset.
    """

    def __init__(self, dt, num_envs, device):
        self.dt = dt
        self.N = num_envs
        self.dev = device

        self.state = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.timer = torch.zeros(num_envs, device=device)
        self.phase = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.saved_pick_pos = torch.zeros(num_envs, 3, device=device)
        # Yaw of the cube we are currently picking; latched at DOWN_PICK so the
        # grasp/lift/place phases keep the same orientation even if the cube
        # rotates slightly while being squeezed.
        self.saved_pick_quat = (
            FLIP_QUAT.to(device).unsqueeze(0).expand(num_envs, -1).clone()
        )

        # Tuning — fast, RL-looking trajectories (loose waypoints + high gain)
        self.kp = 3.0          # position P-gain
        self.kr = 4.0          # orientation P-gain — high so rotation leads position
        self.dpos_clamp = 0.60 # max per-step position delta in metres
        self.drot_clamp = 1.20 # max per-step axis-angle delta in radians
        # Action smoothing (EMA): blends current command with previous one to
        # kill the velocity step that happens at every state transition. Higher
        # alpha = more responsive but more "khựng"; lower = smoother but laggy.
        self.action_alpha = 0.35
        # Per-state arrival thresholds:
        #   - "loose" for hover/transition waypoints → state advances early so
        #     the next waypoint blends in and the trajectory curves naturally
        #     instead of stopping at every intermediate point.
        #   - "tight" for terminal waypoints where we actually grasp/release.
        self.dist_th_tight = 0.012
        self.dist_th_loose = 0.05
        self.rot_th = 0.15     # orientation arrival threshold (rad ≈ 8.6°)
        self.hover = 0.14      # hover height above target (m); lower since we're faster
        self.cube_h = 0.045    # cube side length
        # Small upward margin when releasing/setting down a cube. Without it,
        # IK overshoot at the high speeds below pushes TCP into the cube under
        # it. 6 mm is too small to tilt the cube when released.
        self.place_margin = 0.006
        # Speed multiplier for "careful" states (slow descent during grasp /
        # place). 1.0 for transit, ~0.4 for delicate motions.
        self.slow_speed_scale = 0.4

        # Previous action for EMA smoothing.
        self.prev_action = torch.zeros(num_envs, 7, device=device)
        # Gripper starts open (+1).
        self.prev_action[:, 6] = 1.0

    def reset_idx(self, ids):
        self.state[ids] = S.REST
        self.timer[ids] = 0.0
        self.phase[ids] = 0
        self.saved_pick_pos[ids] = 0.0
        # Reset the latched grasp orientation to the bare flip-down quat.
        self.saved_pick_quat[ids] = FLIP_QUAT.to(self.dev)
        # Reset action smoothing so the new episode starts cleanly.
        self.prev_action[ids] = 0.0
        self.prev_action[ids, 6] = 1.0  # gripper open

    def compute(self, ee_pos, ee_quat, c1, c2, c3, c1_q, c2_q, c3_q):
        """
        Args:
            ee_pos: (N,3), ee_quat: (N,4) w,x,y,z
            c1,c2,c3: (N,3) cube positions
            c1_q,c2_q,c3_q: (N,4) cube quaternions (wxyz)
        Returns:
            (N,7) [delta_pos(3), delta_rot(3), gripper(1)]
        """
        is_phase0 = (self.phase == 0).unsqueeze(-1)  # (N,1)
        pick = torch.where(is_phase0, c2, c3)
        # `place` is the cube directly below the picked one's destination.
        # Phase 0 → c1 (blue, on table). Phase 1 → c2 (red, already stacked on c1).
        # Either way, we just need to add ONE cube height to land on top of it.
        place = torch.where(is_phase0, c1, c2)
        pick_q = torch.where(is_phase0, c2_q, c3_q)
        # Obstacles for grasp-yaw selection:
        #   phase 0 (pick red):   c1 (blue) and c3 (green) are on the table
        #   phase 1 (pick green): c1 (blue) and c2 (red, stacked on c1)
        obstacles = torch.stack(
            (
                c1,
                torch.where(is_phase0.expand_as(c2), c3, c2),
            ),
            dim=1,
        )  # (N, 2, 3)
        # c1_q is unused: place orientation is latched from the pick yaw so the
        # grasped cube doesn't twist on its way to the stack. Kept in the
        # signature for clarity / future use.
        _ = c1_q
        # We always add ONE cube-height on top of the `place` cube, because in
        # phase 1 `place` is already the top of the existing stack (c2 on c1).

        # Per-env desired quat: aligned with the currently relevant cube's yaw
        # AND with jaws perpendicular to the nearest obstacle, so we don't bash
        # the already-stacked cubes when picking the next one. During place
        # phases (ABOVE_PLACE..RETREAT) we keep the latched pick quat so the
        # grasped cube doesn't suddenly twist after being grasped.
        pick_des_quat = desired_ee_quat_aligned_with_cube(pick, pick_q, obstacles)
        in_place_phase = ((self.state >= S.GRASP) & (self.state <= S.RETREAT)).unsqueeze(-1)
        # While placing, hold the latched grasp orientation; otherwise track the pick cube.
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
            ch = self.cube_h  # one cube-height above the `place` cube → land exactly on top

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
                # Stop slightly ABOVE the exact stacking height so the IK
                # overshoot at high speeds doesn't ram the cube below.
                target[i] = torch.tensor(
                    [t[0], t[1], t[2] + ch + self.place_margin], device=self.dev
                )
                gripper[i] = -1.0
            elif s == S.RELEASE:
                # Same target as DOWN_PLACE — opening the gripper drops the cube
                # the tiny margin and it settles flat on the one below.
                target[i] = torch.tensor(
                    [t[0], t[1], t[2] + ch + self.place_margin], device=self.dev
                )
            elif s == S.RETREAT:
                target[i] = torch.tensor([t[0], t[1], t[2] + self.hover + ch], device=self.dev)
            elif s == S.DONE:
                target[i] = ee_pos[i]  # hold position

        # === Position: P-control ===
        delta_pos = self.kp * (target - ee_pos)
        delta_pos = torch.clamp(delta_pos, -self.dpos_clamp, self.dpos_clamp)

        # Slow down for delicate states (descent to grasp + descent / release on
        # the stack). Transit states stay at full speed so the demo still looks
        # fast overall.
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
        # q_err = q_des * q_curr^{-1} is a world-frame rotation taking ee_quat → des_quat.
        quat_err = math_utils.quat_mul(des_quat, math_utils.quat_conjugate(ee_quat))
        # Ensure shortest-path rotation (q and -q represent the same orientation).
        quat_err = torch.where(quat_err[:, 0:1] < 0, -quat_err, quat_err)
        axis_angle_err = math_utils.axis_angle_from_quat(quat_err)
        delta_rot = self.kr * axis_angle_err
        delta_rot = torch.clamp(delta_rot, -self.drot_clamp, self.drot_clamp)
        # Apply the same slow-down to rotation so the gripper doesn't twitch
        # while we're trying to settle a cube.
        delta_rot = delta_rot * speed_scale

        # === State transitions ===
        dist = torch.norm(target - ee_pos, dim=-1)
        rot_err_norm = torch.norm(axis_angle_err, dim=-1)
        self.timer += self.dt

        # Per-state arrival policy:
        #   - LOOSE: hover / transition waypoints — advance early so paths blend.
        #     We don't require orientation alignment here either; the next state
        #     keeps tracking the same yaw anyway.
        #   - TIGHT_BOTH: terminal pick/place — wait for both pos and rot to
        #     converge before we close/open the gripper.
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
                # Hold the finished stack briefly; env will time out & reset.
                advance = False

            if advance:
                if s == S.DOWN_PICK:
                    # Latch the pick pose at the moment we finished descending,
                    # so GRASP/LIFT use a stable reference even if the cube
                    # moves slightly during the grasp.
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

        # EMA smoothing on the continuous channels (pos + rot). The gripper
        # channel is left binary so open/close stays crisp.
        a = self.action_alpha
        actions[:, :6] = a * actions[:, :6] + (1.0 - a) * self.prev_action[:, :6]
        self.prev_action = actions.clone()
        return actions

    def force_reset_mask(self) -> torch.Tensor:
        """Envs that have been in DONE for at least WAIT[DONE] seconds.

        Returns a bool tensor of shape (N,). The caller is expected to call
        `env.unwrapped._reset_idx(ids)` and `sm.reset_idx(ids)` for the True
        positions, so the demo restarts shortly after the last cube is placed
        regardless of whether the env's own success/timeout termination fired.
        """
        return (self.state == S.DONE) & (self.timer >= WAIT[S.DONE])


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    task = "Isaac-Stack-Cube-Franka-IK-Rel-v0"
    env_cfg = parse_env_cfg(
        task, device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    # Shorter episodes so it resets faster (looks like RL training)
    env_cfg.episode_length_s = 18.0

    print(f"[INFO] Task: {task} | num_envs: {args_cli.num_envs}", flush=True)
    env = gym.make(task, cfg=env_cfg)
    print(f"[INFO] Action space: {env.action_space.shape}", flush=True)
    obs, info = env.reset()
    print(f"[INFO] Running! Robot will pick & stack 3 cubes, then reset.", flush=True)

    sm = StackSM(
        dt=env_cfg.sim.dt * env_cfg.decimation,
        num_envs=env.unwrapped.num_envs,
        device=env.unwrapped.device,
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

            # Force a reset 1 s after we finished placing the last cube,
            # regardless of whether the env's own success/time-out fired.
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
