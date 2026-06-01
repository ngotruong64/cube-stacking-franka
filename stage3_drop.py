#!/usr/bin/env python3
"""
Stage 3 — LIGHT JITTER + RANDOM DROP: Light jitter; gripper randomly opens while carrying.

Usage:
    conda activate env_isaaclab
    cd /home/truongnq/Desktop/3cube
    python stage3_drop.py --num_envs 4
    python stage3_drop.py --num_envs 4 --no-video
"""

"""Launch Omniverse first."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Stage 3: Light jitter + random drop.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--debug_env_logs", action="store_true", default=False)
parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--video_length", type=int, default=3000)
parser.add_argument("--video_folder", type=str, default="videos/stage3_drop")
parser.add_argument("--video_fps", type=int, default=60)
parser.add_argument("--video_width", type=int, default=1998)
parser.add_argument("--video_height", type=int, default=1080)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import math, os, glob, shutil, subprocess
import gymnasium as gym
import torch
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


class S:
    REST = 0; ABOVE_PICK = 1; DOWN_PICK = 2; GRASP = 3; LIFT = 4
    ABOVE_PLACE = 5; DOWN_PLACE = 6; RELEASE = 7; RETREAT = 8; NEXT = 9; DONE = 10

WAIT = {S.REST: 0.1, S.GRASP: 0.35, S.RELEASE: 0.2, S.DONE: 1.0}
STATE_NAMES = ["REST", "ABOVE_PICK", "DOWN_PICK", "GRASP", "LIFT",
               "ABOVE_PLACE", "DOWN_PLACE", "RELEASE", "RETREAT", "NEXT", "DONE"]
CARRYING_STATES = {S.GRASP, S.LIFT, S.ABOVE_PLACE, S.DOWN_PLACE}
FLIP_QUAT = torch.tensor([0.0, 1.0, 0.0, 0.0])

def format_env_ids(env_ids, max_count=12):
    values = env_ids.detach().cpu().tolist() if isinstance(env_ids, torch.Tensor) else list(env_ids)
    shown = values[:max_count]
    suffix = "" if len(values) <= max_count else f"...(+{len(values) - max_count})"
    return f"{len(values)} envs {shown}{suffix}"

def natural_video_fps(env_cfg):
    return max(1, int(round(1.0 / (env_cfg.sim.dt * env_cfg.decimation))))

def ffmpeg_executable():
    exe = shutil.which("ffmpeg")
    if exe is not None: return exe
    try:
        import imageio_ffmpeg; return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception: return None

def convert_latest_video_fps(video_folder, target_fps, capture_fps):
    if target_fps <= 0 or target_fps == capture_fps: return
    ffmpeg = ffmpeg_executable()
    if ffmpeg is None: return
    candidates = [p for p in glob.glob(os.path.join(video_folder, "*.mp4")) if ".tmp-" not in os.path.basename(p)]
    if not candidates: return
    src = max(candidates, key=os.path.getmtime)
    tmp = f"{src[:-4]}.tmp-{target_fps}fps.mp4"
    cmd = [ffmpeg, "-y", "-i", src, "-vf", f"fps={target_fps}", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        if os.path.exists(tmp): os.remove(tmp)
        return
    os.replace(tmp, src)

def yaw_from_quat(q):
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))

def wrap_quarter(yaw):
    return ((yaw + math.pi/4.0) % (math.pi/2.0)) - math.pi/4.0

def _ee_quat_from_yaw(yaw):
    half = yaw*0.5; c, s = torch.cos(half), torch.sin(half); zero = torch.zeros_like(c)
    return torch.stack((zero, c, s, zero), dim=-1)

def quat_conjugate(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack((w, -x, -y, -z), dim=-1)

def quat_rotate(q, v):
    w, x, y, z = q.unbind(-1); q_xyz = torch.stack((x, y, z), dim=-1)
    uv = torch.cross(q_xyz, v, dim=-1); uuv = torch.cross(q_xyz, uv, dim=-1)
    return v + 2.0*(uv*w.unsqueeze(-1) + uuv)

def _angular_dist_mod_pi(a, b):
    return torch.abs((a - b + math.pi/2.0) % math.pi - math.pi/2.0)

def grasp_yaw_avoiding(pick_pos, pick_quat, obstacles_pos):
    cube_yaw = wrap_quarter(yaw_from_quat(pick_quat))
    dxy = obstacles_pos[..., :2] - pick_pos[:, None, :2]; d2 = (dxy*dxy).sum(dim=-1)
    nearest = d2.argmin(dim=-1); sel = nearest[:, None, None].expand(-1, 1, 2)
    nearest_xy = obstacles_pos[..., :2].gather(1, sel).squeeze(1)
    min_d = d2.gather(1, nearest[:, None]).squeeze(-1)
    diff = pick_pos[:, :2] - nearest_xy; pref_yaw = torch.atan2(diff[..., 1], diff[..., 0])
    cand1, cand2 = cube_yaw, cube_yaw + math.pi/2.0
    use1 = _angular_dist_mod_pi(cand1, pref_yaw) <= _angular_dist_mod_pi(cand2, pref_yaw)
    chosen = torch.where(use1, cand1, cand2)
    return torch.where(min_d < 1e-6, cube_yaw, chosen)

def desired_ee_quat_aligned_with_cube(pick_pos, pick_quat, obstacles_pos):
    return _ee_quat_from_yaw(grasp_yaw_avoiding(pick_pos, pick_quat, obstacles_pos))


class StackSM:
    """Stage 3: Light jitter + random gripper drops while carrying."""

    def __init__(self, dt, num_envs, device, debug_env_logs=False):
        self.dt, self.N, self.dev, self.debug_env_logs = dt, num_envs, device, debug_env_logs
        self.global_time = 0.0
        self.state = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.timer = torch.zeros(num_envs, device=device)
        self.phase = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.phase_timer = torch.zeros(num_envs, device=device)
        self.saved_pick_pos = torch.zeros(num_envs, 3, device=device)
        self.saved_pick_quat = FLIP_QUAT.to(device).unsqueeze(0).expand(num_envs, -1).clone()
        self.c2_init = torch.zeros(num_envs, 3, device=device)
        self.c3_init = torch.zeros(num_envs, 3, device=device)
        self.c2_init_q = torch.zeros(num_envs, 4, device=device); self.c2_init_q[:, 0] = 1.0
        self.c3_init_q = torch.zeros(num_envs, 4, device=device); self.c3_init_q[:, 0] = 1.0
        self.init_saved = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.grasp_offset_yaw = torch.zeros(num_envs, device=device)
        self.grasp_offset_pos = torch.zeros(num_envs, 3, device=device)

        # Stage 3 random drop parameters
        self.drop_timer = torch.zeros(num_envs, device=device)
        self.drop_duration = 0.5
        self.drop_trigger_prob = 0.04

        self.kp, self.kr = 3.0, 4.0
        self.dpos_clamp, self.drot_clamp = 0.60, 1.20
        self.action_alpha = 0.35
        self.dist_th_tight, self.dist_th_loose, self.rot_th = 0.012, 0.05, 0.15
        self.hover, self.cube_h, self.place_margin = 0.14, 0.045, 0.006
        self.slow_speed_scale = 0.4
        self.prev_action = torch.zeros(num_envs, 7, device=device); self.prev_action[:, 6] = 1.0

    def reset_idx(self, ids):
        self.state[ids] = S.REST; self.timer[ids] = 0.0; self.phase[ids] = 0; self.phase_timer[ids] = 0.0
        self.saved_pick_pos[ids] = 0.0; self.saved_pick_quat[ids] = FLIP_QUAT.to(self.dev)
        self.prev_action[ids] = 0.0; self.prev_action[ids, 6] = 1.0
        self.init_saved[ids] = False; self.grasp_offset_yaw[ids] = 0.0; self.grasp_offset_pos[ids] = 0.0
        self.drop_timer[ids] = 0.0

    def compute(self, ee_pos, ee_quat, c1, c2, c3, c1_q, c2_q, c3_q):
        self.global_time += self.dt; self.phase_timer += self.dt
        self.drop_timer = torch.clamp(self.drop_timer - self.dt, min=0.0)

        for i in range(self.N):
            if not self.init_saved[i]:
                self.c2_init[i] = c2[i].clone(); self.c3_init[i] = c3[i].clone()
                self.c2_init_q[i] = c2_q[i].clone(); self.c3_init_q[i] = c3_q[i].clone()
                self.init_saved[i] = True

        # Random gripper drop while carrying
        for i in range(self.N):
            if self.drop_timer[i] == 0.0:
                s = self.state[i].item()
                if s in CARRYING_STATES:
                    if torch.rand(1, device=self.dev).item() < self.drop_trigger_prob:
                        self.drop_timer[i] = self.drop_duration
                        print(f"[Stage 3 RANDOM DROP] env={i} state={STATE_NAMES[s]} phase={self.phase[i].item()} | gripper opening for {self.drop_duration:.1f}s", flush=True)
        drop_mask = self.drop_timer > 0.0

        is_ph0 = (self.phase == 0).unsqueeze(-1); is_ph1 = (self.phase == 1).unsqueeze(-1)
        is_ph2 = (self.phase == 2).unsqueeze(-1); is_ph3 = (self.phase == 3).unsqueeze(-1)
        pick = is_ph0.float()*c2 + is_ph1.float()*c3 + is_ph2.float()*c3 + is_ph3.float()*c2
        place = is_ph0.float()*c1 + is_ph1.float()*c2 + is_ph2.float()*self.c3_init + is_ph3.float()*self.c2_init
        pick_q = is_ph0.float()*c2_q + is_ph1.float()*c3_q + is_ph2.float()*c3_q + is_ph3.float()*c2_q
        ch_add = (is_ph0.float() + is_ph1.float()).squeeze(-1) * self.cube_h
        is_c2_pick = ((self.phase == 0)|(self.phase == 3)).unsqueeze(-1).unsqueeze(-1)
        obstacles = torch.where(is_c2_pick, torch.stack((c1, c3), dim=1), torch.stack((c1, c2), dim=1))
        place_q = is_ph0.float()*c1_q + is_ph1.float()*c2_q + is_ph2.float()*self.c3_init_q + is_ph3.float()*self.c2_init_q

        pick_des_quat = desired_ee_quat_aligned_with_cube(pick, pick_q, obstacles)
        place_des_quat = _ee_quat_from_yaw(yaw_from_quat(place_q) + self.grasp_offset_yaw)
        is_pick_state = (self.state < S.GRASP).unsqueeze(-1)
        is_lift_state = ((self.state >= S.GRASP) & (self.state < S.ABOVE_PLACE)).unsqueeze(-1)
        des_quat = torch.where(is_pick_state, pick_des_quat, torch.where(is_lift_state, self.saved_pick_quat, place_des_quat))
        offset_world_all = quat_rotate(des_quat, self.grasp_offset_pos)

        target = ee_pos.clone(); gripper = torch.ones(self.N, device=self.dev)
        for i in range(self.N):
            s = self.state[i].item(); ph = self.phase[i].item()
            if ph >= 4 and s != S.DONE: self.state[i] = S.DONE; self.timer[i] = 0.0; s = S.DONE
            p, t, ch, off_w = pick[i], place[i], ch_add[i].item(), offset_world_all[i]
            if s == S.REST: target[i] = ee_pos[i]
            elif s == S.ABOVE_PICK: target[i] = torch.tensor([p[0], p[1], p[2]+self.hover], device=self.dev)
            elif s == S.DOWN_PICK: target[i] = p.clone()
            elif s == S.GRASP: target[i] = self.saved_pick_pos[i]; gripper[i] = -1.0
            elif s == S.LIFT: target[i] = self.saved_pick_pos[i].clone(); target[i, 2] += self.hover; gripper[i] = -1.0
            elif s == S.ABOVE_PLACE: target[i] = torch.tensor([t[0], t[1], t[2]+self.hover+ch], device=self.dev)+off_w; gripper[i] = -1.0
            elif s == S.DOWN_PLACE: target[i] = torch.tensor([t[0], t[1], t[2]+ch+self.place_margin], device=self.dev)+off_w; gripper[i] = -1.0
            elif s == S.RELEASE: target[i] = torch.tensor([t[0], t[1], t[2]+ch+self.place_margin], device=self.dev)+off_w
            elif s == S.RETREAT: target[i] = torch.tensor([t[0], t[1], t[2]+self.hover+ch], device=self.dev)+off_w
            elif s == S.DONE: target[i] = ee_pos[i]

        # Force gripper open on envs with active drop timer
        if drop_mask.any():
            gripper[drop_mask] = 1.0

        delta_pos = self.kp * (target - ee_pos)
        delta_pos = torch.clamp(delta_pos, -self.dpos_clamp, self.dpos_clamp)
        slow_mask = (self.state == S.DOWN_PICK) | (self.state == S.DOWN_PLACE) | (self.state == S.RELEASE)
        speed_scale = (1.0 - slow_mask.float() * (1.0 - self.slow_speed_scale)).unsqueeze(-1)
        delta_pos = delta_pos * speed_scale

        quat_err = math_utils.quat_mul(des_quat, math_utils.quat_conjugate(ee_quat))
        quat_err = torch.where(quat_err[:, 0:1] < 0, -quat_err, quat_err)
        axis_angle_err = math_utils.axis_angle_from_quat(quat_err)
        delta_rot = self.kr * axis_angle_err
        delta_rot = torch.clamp(delta_rot, -self.drot_clamp, self.drot_clamp)
        delta_rot = delta_rot * speed_scale

        # ── Stage 3: LIGHT JITTER ──
        delta_pos += torch.randn_like(delta_pos) * 0.05
        delta_rot += torch.randn_like(delta_rot) * 0.05

        dist = torch.norm(target - ee_pos, dim=-1); rot_err_norm = torch.norm(axis_angle_err, dim=-1)
        self.timer += self.dt
        LOOSE_POS_ONLY = {S.LIFT, S.RETREAT}; LOOSE_WITH_ROT = {S.ABOVE_PICK, S.ABOVE_PLACE}; TIGHT_BOTH = {S.DOWN_PICK, S.DOWN_PLACE}
        for i in range(self.N):
            s, d, r, t = self.state[i].item(), dist[i].item(), rot_err_norm[i].item(), self.timer[i].item()
            advance = False
            if s == S.REST: advance = (t >= WAIT[S.REST])
            elif s in LOOSE_POS_ONLY: advance = d < self.dist_th_loose
            elif s in LOOSE_WITH_ROT: advance = (d < self.dist_th_loose) and (r < self.rot_th)
            elif s in TIGHT_BOTH: advance = (d < self.dist_th_tight) and (r < self.rot_th)
            elif s == S.GRASP: advance = (t >= WAIT[S.GRASP])
            elif s == S.RELEASE: advance = (t >= WAIT[S.RELEASE])
            elif s == S.NEXT: advance = True
            elif s == S.DONE: advance = False
            if advance:
                if s == S.DOWN_PICK:
                    ph = self.phase[i].item()
                    pick_cube = c2[i] if ph in (0, 3) else c3[i]
                    pick_cube_q = c2_q[i] if ph in (0, 3) else c3_q[i]
                    self.saved_pick_pos[i] = ee_pos[i].clone(); self.saved_pick_quat[i] = des_quat[i].clone()
                    self.grasp_offset_yaw[i] = yaw_from_quat(des_quat[i].unsqueeze(0)).item() - yaw_from_quat(pick_cube_q.unsqueeze(0)).item()
                    offset_world = ee_pos[i] - pick_cube
                    self.grasp_offset_pos[i] = quat_rotate(quat_conjugate(des_quat[i].unsqueeze(0)), offset_world.unsqueeze(0)).squeeze(0)
                    self.state[i] = S.GRASP
                elif s == S.NEXT:
                    ph = self.phase[i].item()
                    if ph < 3: self.phase[i] = ph+1; self.phase_timer[i] = 0.0; self.state[i] = S.REST
                    else: self.phase[i] = 4; self.phase_timer[i] = 0.0; self.state[i] = S.DONE
                    self.timer[i] = 0.0
                elif s == S.RETREAT: self.state[i] = S.NEXT; self.timer[i] = 0.0
                else: self.state[i] = s+1; self.timer[i] = 0.0

        actions = torch.cat([delta_pos, delta_rot, gripper.unsqueeze(-1)], dim=-1)
        a = self.action_alpha
        actions[:, :6] = a * actions[:, :6] + (1.0-a) * self.prev_action[:, :6]
        self.prev_action = actions.clone()
        return actions

    def force_reset_mask(self):
        return (self.state == S.DONE) & (self.timer >= WAIT[S.DONE])

    def check_phase_timeout(self):
        return (self.phase < 4) & (self.phase_timer > 9.0)


def main():
    task = "Isaac-Stack-Cube-Franka-IK-Rel-v0"
    env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    env_cfg.episode_length_s = 150.0
    if args_cli.video: env_cfg.viewer.resolution = (args_cli.video_width, args_cli.video_height)
    video_capture_fps = natural_video_fps(env_cfg)
    for name, color in {"cube_1": (0.0,0.233,1.0), "cube_2": (1.0,0.0,0.0), "cube_3": (1.0,1.0,0.0)}.items():
        getattr(env_cfg.scene, name).spawn.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.45)
    if hasattr(env_cfg, "terminations"):
        if hasattr(env_cfg.terminations, "success"): env_cfg.terminations.success = None
        for tn in ["cube_1_dropping","cube_2_dropping","cube_3_dropping"]:
            if hasattr(env_cfg.terminations, tn): setattr(env_cfg.terminations, tn, None)

    print(f"[INFO] Stage 3 LIGHT JITTER + RANDOM DROP | Task: {task} | num_envs: {args_cli.num_envs}", flush=True)
    if args_cli.video:
        env = gym.make(task, cfg=env_cfg, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, video_folder=args_cli.video_folder, step_trigger=lambda step: step==0,
                                        video_length=args_cli.video_length, fps=video_capture_fps, disable_logger=True)
    else:
        env = gym.make(task, cfg=env_cfg)
    obs, info = env.reset()
    sm = StackSM(dt=env_cfg.sim.dt*env_cfg.decimation, num_envs=env.unwrapped.num_envs, device=env.unwrapped.device, debug_env_logs=args_cli.debug_env_logs)

    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            ee_frame = env.unwrapped.scene["ee_frame"]
            ee_pos = ee_frame.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
            ee_quat = ee_frame.data.target_quat_w[..., 0, :].clone()
            c1 = env.unwrapped.scene["cube_1"]; c2 = env.unwrapped.scene["cube_2"]; c3 = env.unwrapped.scene["cube_3"]
            origins = env.unwrapped.scene.env_origins
            c1_pos, c2_pos, c3_pos = c1.data.root_pos_w-origins, c2.data.root_pos_w-origins, c3.data.root_pos_w-origins
            c1_q, c2_q, c3_q = c1.data.root_quat_w, c2.data.root_quat_w, c3.data.root_quat_w
            actions = sm.compute(ee_pos, ee_quat, c1_pos, c2_pos, c3_pos, c1_q, c2_q, c3_q)
            obs, rew, terminated, truncated, info = env.step(actions)

            cube_dropped = (c1_pos[:,2]<0.01)|(c2_pos[:,2]<0.01)|(c3_pos[:,2]<0.01)
            phase_timeout = sm.check_phase_timeout()
            reset_mask = cube_dropped | phase_timeout | sm.force_reset_mask()
            if reset_mask.any():
                reset_ids = reset_mask.nonzero(as_tuple=False).squeeze(-1)
                env.unwrapped._reset_idx(reset_ids); sm.reset_idx(reset_ids)
                print(f"[{step:5d}] manual-reset: {format_env_ids(reset_ids)}", flush=True)
            dones = terminated | truncated
            if dones.any(): sm.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))
            step += 1
            if step == 1 or step % 100 == 0:
                dropping = (sm.drop_timer > 0.0).nonzero(as_tuple=False).squeeze(-1).tolist()
                print(f"[{step:5d}] t={sm.global_time:.1f}s | Stage 3 DROP | states={[STATE_NAMES[s] for s in sm.state.tolist()]} ph={sm.phase.tolist()} dropping={dropping}", flush=True)
            if args_cli.video and step >= args_cli.video_length: break
    env.close()
    if args_cli.video: convert_latest_video_fps(args_cli.video_folder, args_cli.video_fps, video_capture_fps)

if __name__ == "__main__":
    main()
    simulation_app.close()
