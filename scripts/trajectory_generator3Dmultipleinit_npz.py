"""Generate trapezoidal trajectory datasets (3D: x, y, theta) and save to .npz.

This script samples trajectories at 25 steps per second (25 Hz) while keeping
total trajectory time variable based on sampled dynamics and displacement.
Theta is the heading angle (direction of travel). Robot always rotates first
(in-place) to align heading, then translates along a straight line.
Output is .npz with features (N,14), targets (N,50,3), and feature_names (14,).
"""

# Variable	Min	Max	Notes
# init_x	-5.0	5.0	
# init_y	-5.0	5.0	
# init_theta	-π	π	
# r_goal	0.0	5.0	Euclidean distance from init to goal
# theta_goal	-π	π	
# goal_x	-10.0	10.0	init_x + r_goal·cos(theta_goal)
# goal_y	-10.0	10.0	init_y + r_goal·sin(theta_goal)
# v_max	0.1	0.2	m/s
# a_max	0.02	0.04	m/s²
# omega_max	0.5	1.0	rad/s
# alpha_max	0.05	0.1	rad/s²

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class GeneratorConfig:
    num_samples: int = 5
    sample_hz: float = 25.0
    # linear limits
    v_max_range: tuple[float, float] = (0.1, 0.2)
    a_max_range: tuple[float, float] = (0.02, 0.04)
    # angular limits (rad, rad/s^2)
    omega_max_range: tuple[float, float] = (0.5, 1.0)
    alpha_max_range: tuple[float, float] = (0.05, 0.1)
    # 0 - 5 meters
    displacement_range: tuple[float, float] = (0.0, 5.0)
    max_via_points: int = 1
    end_hold_steps: int = 50
    seed: int = 30


def solve_profile_times(
    displacement: float,
    v_max: float,
    a_max: float,
) -> tuple[float, float, float, float]:
    """Solve trapezoid/triangle timing from displacement and dynamic limits.

    Returns (t_acc, t_cruise, total_time, v_peak).
    """
    critical_disp = (v_max**2) / a_max

    if displacement <= critical_disp:
        # Triangular profile (never reaches v_max).
        t_acc = np.sqrt(displacement / a_max)
        t_cruise = 0.0
        v_peak = a_max * t_acc
    else:
        # Trapezoidal profile (reaches v_max and cruises).
        t_acc = v_max / a_max
        v_peak = v_max
        t_cruise = (displacement - critical_disp) / v_max

    total_time = 2.0 * t_acc + t_cruise
    return t_acc, t_cruise, total_time, v_peak


def sample_time_grid(total_time: float, sample_hz: float) -> np.ndarray:
    """Build time points at fixed 25 Hz-style spacing with variable duration."""
    dt = 1.0 / sample_hz
    t = np.arange(0.0, total_time + 1e-12, dt, dtype=np.float64)
    if len(t) == 0:
        t = np.array([0.0], dtype=np.float64)
    if t[-1] < total_time:
        t = np.append(t, total_time)
    return t


def profile_at_time(
    t: float,
    a_max: float,
    t_acc: float,
    t_cruise: float,
    v_peak: float,
) -> tuple[float, float, float, str]:
    """Evaluate position, velocity, acceleration, and phase at time t."""
    eps = 1e-10
    t_dec_start = t_acc + t_cruise
    s_acc = 0.5 * a_max * (t_acc**2)
    s_cruise = v_peak * t_cruise

    if t <= t_acc + eps:
        pos = 0.5 * a_max * (t**2)
        vel = a_max * t
        acc = a_max
        phase = "accel"
    elif t <= t_dec_start + eps and t_cruise > eps:
        tau = t - t_acc
        pos = s_acc + v_peak * tau
        vel = v_peak
        acc = 0.0
        phase = "const"
    else:
        tau = min(max(0.0, t - t_dec_start), t_acc)
        pos = s_acc + s_cruise + v_peak * tau - 0.5 * a_max * (tau**2)
        vel = max(0.0, v_peak - a_max * tau)
        acc = -a_max
        phase = "deaccel"

    return float(pos), float(vel), float(acc), phase


def create_dataset(config: GeneratorConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate 3D (x, y, theta) dataset.

    Returns (features, targets, feature_names) where:
      features:      (N, 14) float16
      targets:       (N, 50, 3) float16  — [x, y, theta]
      feature_names: (14,) str
    N = num_samples * (num_windows per sample).
    """
    random.seed(config.seed)
    np.random.seed(config.seed)

    phase_map = {"accel": 0, "const": 1, "deaccel": 2, "end_hold": 3}
    T = 50  # target timesteps

    # feature_names = np.array([
    #     "s_goal_x", "s_goal_y", "s_goal_theta",
    #     "v_const", "accel",
    #     "q_init_x", "q_init_y", "q_init_theta",
    #     "qdot_init_x", "qdot_init_y", "qdot_init_theta",
    #     "t_init", "part_enum",
    # ])
    feature_names = np.array([
        "s_goal_x", "s_goal_y", "s_goal_theta",
        "v_const", "accel", "omega_const", "alpha_const",
        "q_init_x", "q_init_y", "q_init_theta",
        "qdot_init_x", "qdot_init_y", "qdot_init_theta",
        "t_init",
    ])

    all_features = []
    all_targets = []

    for _ in range(config.num_samples):
        # sample linear and angular dynamics
        v_max = random.uniform(*config.v_max_range)
        a_max = random.uniform(*config.a_max_range)
        omega_max = random.uniform(*config.omega_max_range)
        alpha_max = random.uniform(*config.alpha_max_range)

        # Robot initial pose: at origin but heading may differ -> must rotate first
        init_x = random.uniform(-5.0, 5.0)
        init_y = random.uniform(-5.0, 5.0)
        init_theta = float(random.uniform(-np.pi, np.pi))

        r_goal = random.uniform(*config.displacement_range)
        # theta_goal = random.uniform(0, 2 * np.pi)
        theta_goal = random.uniform(-np.pi, np.pi)
        goal_x = init_x + r_goal * np.cos(theta_goal)
        goal_y = init_y + r_goal * np.sin(theta_goal)

        # rotation: rotate in-place from init_theta -> goal_theta
        theta_diff = float(np.arctan2(np.sin(theta_goal - init_theta), np.cos(theta_goal - init_theta)))
        rot_disp = abs(theta_diff)
        rot_sign = np.sign(theta_diff) if rot_disp > 0.0 else 1.0

        t_acc_r, t_cruise_r, total_time_r, omega_peak = solve_profile_times(
            displacement=rot_disp, v_max=omega_max, a_max=alpha_max
        )
        t_rot = sample_time_grid(total_time=total_time_r, sample_hz=config.sample_hz)

        # translation: straight-line from origin to goal, heading fixed to goal_theta
        t_acc_t, t_cruise_t, total_time_t, v_peak = solve_profile_times(
            displacement=r_goal, v_max=v_max, a_max=a_max
        )
        t_trans = sample_time_grid(total_time=total_time_t, sample_hz=config.sample_hz)

        positions = []  # list of (x, y, theta)
        velocitys = []  # list of (vel_x, vel_y)
        phases = []
        times = []

        # build rotation phase (in-place)
        for time_s in t_rot:
            path_pos, path_vel, _, phase = profile_at_time(
                t=float(time_s), a_max=alpha_max, t_acc=t_acc_r,
                t_cruise=t_cruise_r, v_peak=omega_peak,
            )
            frac = path_pos / rot_disp if rot_disp > 0.0 else 0.0
            theta = init_theta + rot_sign * path_pos
            # keep position at origin during rotation
            positions.append((init_x, init_y, float(theta)))
            velocitys.append((0.0, 0.0))
            phases.append(phase)
            times.append(float(time_s))

        # translation phase: offset times by rotation duration
        t_offset = times[-1] if len(times) > 0 else 0.0
        for time_s in t_trans:
            time_global = float(time_s + t_offset)
            path_pos, path_vel, _, phase = profile_at_time(
                t=float(time_s), a_max=a_max, t_acc=t_acc_t,
                t_cruise=t_cruise_t, v_peak=v_peak,
            )
            progress_ratio = path_pos / r_goal if r_goal > 0.0 else 0.0
            pos_x = init_x + (goal_x - init_x) * progress_ratio
            pos_y = init_y + (goal_y - init_y) * progress_ratio
            vel_x = path_vel * (goal_x - init_x) / r_goal if r_goal > 0.0 else 0.0
            vel_y = path_vel * (goal_y - init_y) / r_goal if r_goal > 0.0 else 0.0
            positions.append((float(pos_x), float(pos_y), float(theta_goal)))
            velocitys.append((float(vel_x), float(vel_y)))
            phases.append(phase)
            times.append(time_global)

        # end hold
        for hold_idx in range(config.end_hold_steps):
            hold_time = float(times[-1] + ((hold_idx + 1) / config.sample_hz))
            positions.append((float(goal_x), float(goal_y), float(theta_goal)))
            velocitys.append((0.0, 0.0))
            phases.append("end_hold")
            times.append(hold_time)

        n_steps = len(positions)
        n_windows = n_steps // T

        if n_windows <= 0:
            continue

        start_positions_arr = np.linspace(0, n_steps - T, n_windows, dtype=int)
        start_positions_arr = np.unique(start_positions_arr)

        for start in start_positions_arr:
            feat = np.zeros(14, dtype=np.float16)
            feat[0] = goal_x                    # s_goal_x
            feat[1] = goal_y                    # s_goal_y
            feat[2] = theta_goal                # s_goal_theta
            feat[3] = v_max                     # v_const
            feat[4] = a_max                     # accel
            feat[5] = omega_max                 # omega_const
            feat[6] = alpha_max                 # alpha_const
            feat[7] = positions[start][0]       # q_init_x
            feat[8] = positions[start][1]       # q_init_y
            feat[9] = positions[start][2]       # q_init_theta
            feat[10] = velocitys[start][0]      # qdot_init_x
            feat[11] = velocitys[start][1]      # qdot_init_y

            # compute qdot_init_theta by forward finite-difference of wrapped angle
            if start + 1 < n_steps:
                dt_theta = times[start + 1] - times[start]
                if dt_theta == 0.0:
                    qdot_theta = 0.0
                else:
                    theta_curr = positions[start][2]
                    theta_next = positions[start + 1][2]
                    delta_theta = float(np.arctan2(np.sin(theta_next - theta_curr), np.cos(theta_next - theta_curr)))
                    qdot_theta = float(delta_theta) / float(dt_theta)
            else:
                qdot_theta = 0.0

            feat[12] = qdot_theta               # qdot_init_theta
            feat[13] = times[start]             # t_init
            all_features.append(feat)

            remaining = positions[start:]
            n_rem = len(remaining)
            if n_rem == T:
                indices = np.arange(T)
            elif n_rem > T:
                indices = np.linspace(0, n_rem - 1, T, dtype=int)
            else:
                indices = np.arange(n_rem)

            tgt = np.zeros((T, 3), dtype=np.float16)
            for j, idx in enumerate(indices):
                tgt[j, 0] = remaining[idx][0]  # x
                tgt[j, 1] = remaining[idx][1]  # y
                tgt[j, 2] = remaining[idx][2]  # theta
            if n_rem < T:
                for j in range(n_rem, T):
                    tgt[j, 0] = remaining[-1][0]
                    tgt[j, 1] = remaining[-1][1]
                    tgt[j, 2] = remaining[-1][2]
            all_targets.append(tgt)

    features = np.stack(all_features)
    targets = np.stack(all_targets)
    return features, targets, feature_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 3D trapezoidal trajectory .npz dataset.")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of trajectories.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pose_traject_dataset_3d.npz"),
        help="Output .npz path.",
    )
    parser.add_argument(
        "--max-via-points",
        type=int,
        default=5,
        help="Upper bound for random via point count per trajectory.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = GeneratorConfig(
        num_samples=args.num_samples,
        max_via_points=max(0, args.max_via_points),
        seed=args.seed,
    )

    features, targets, feature_names = create_dataset(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, features=features, targets=targets, feature_names=feature_names)
    print(f"Saved {features.shape[0]} windows from {config.num_samples} trajectories to {args.output}")
    print(f"  features:      {features.shape} {features.dtype}")
    print(f"  targets:       {targets.shape} {targets.dtype}")


if __name__ == "__main__":
    main()
