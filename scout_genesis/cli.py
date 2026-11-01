"""Windows-friendly CLI: diagnostics, rigid-body tests, PPO, evaluation and 3-D capture.

Genesis owns the motor/physics dynamics. The odor signal is a mathematical proxy.
None of these commands makes a claim about chemically accurate gas dispersion.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
from .core import SIM_DT, STEPS_PER_ACTION, MAX_STEPS


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OlfaLink Scout | Windows desktop | Genesis + PPO")
    p.add_argument("command", choices=("doctor", "smoke", "train", "evaluate", "demo"))
    p.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--viewer", action="store_true", help="Open Genesis live 3-D viewer")
    p.add_argument("--steps", type=int, default=20000, help="PPO timesteps for train")
    p.add_argument("--episodes", type=int, default=20, help="Episodes per evaluation policy")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--policy", choices=("auto", "ppo", "random", "scripted"), default="auto")
    p.add_argument("--deterministic", action="store_true", help="Use PPO greedy action (can get stuck)")
    p.add_argument("--record", action="store_true", help="Render Genesis 3-D MP4 (uses more resources)")
    p.add_argument("--demo-actions", type=int, default=120, help="Actions without video")
    p.add_argument("--video-actions", type=int, default=400, help="Actions with MP4: 400 ≈ 64 seconds")
    p.add_argument("--output", type=Path, default=Path("outputs"))
    p.add_argument("--model", type=Path, default=Path("outputs/ppo_genesis.zip"))
    return p


def doctor() -> None:
    """Run without importing Genesis or creating a physics scene."""
    print(f"OS: {platform.platform()}")
    print(f"Python: {sys.version.split()[0]} ({'64' if sys.maxsize > 2**32 else '32'}-bit)")
    if not (3, 10) <= sys.version_info[:2] < (3, 14):
        raise RuntimeError("Genesis requires Python >=3.10 and <3.14")
    for module_name in ("numpy", "torch", "genesis", "gymnasium", "stable_baselines3", "matplotlib"):
        try:
            package = __import__(module_name)
            version = getattr(package, "__version__", "installed")
            print(f"  {module_name:20s} {version}")
        except Exception as exc:
            print(f"  {module_name:20s} NOT AVAILABLE: {exc}")
            raise RuntimeError(f"Dependency {module_name!r} is missing or cannot be imported") from exc
    print("[OK] Imports succeeded. Next: smoke (actual Genesis movement).")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary_of(rows: list[dict]) -> dict:
    """Return measured performance; no fabricated scores."""
    summary = {}
    for policy in dict.fromkeys(r["policy"] for r in rows):
        group = [r for r in rows if r["policy"] == policy]
        summary[policy] = {
            "episodes": len(group),
            "successes": int(sum(r["success"] for r in group)),
            "success_rate": float(np.mean([r["success"] for r in group])),
            "mean_steps": float(np.mean([r["steps"] for r in group])),
            "mean_final_distance_m": float(np.mean([r["distance_to_source_m"] for r in group])),
        }
    return summary


def select_demo_policy(requested: str, model_path: Path) -> str:
    if requested == "auto":
        return "ppo" if model_path.is_file() else "scripted"
    return requested


def scripted_action(t: int) -> int:
    """Fixed motor demonstration ONLY. Never present it as learning."""
    phase = t % 80
    if phase < 21:
        return 1
    if phase < 38:
        return 2
    if phase < 64:
        return 1
    return 3


def action_for(policy: str, obs, model, rng, step: int, deterministic: bool) -> int:
    if policy == "ppo":
        return int(model.predict(obs, deterministic=deterministic)[0])
    if policy == "random":
        return int(rng.integers(5))
    if policy == "scripted":
        return scripted_action(step)
    raise ValueError(f"Invalid policy {policy}")


def physics_smoke(env, seed: int, output: Path) -> None:
    """Assert wheel-based physical movement AND turn; fail loudly otherwise."""
    env.reset(seed=seed, options={"start_xy": (-1.4, -1.3), "source_xy": (1.35, 1.25), "yaw": 0.0})
    x0, y0, _ = env.pose()
    obs = None
    for _ in range(10):
        obs, _, term, trunc, _ = env.step(1)
        if term or trunc:
            raise RuntimeError("Forward smoke ended early; inspect contact/pose")
    x1, y1, _ = env.pose()
    forward = x1 - x0
    if not math.isfinite(forward) or forward <= 0.01:
        raise RuntimeError(f"Forward motion failed: +X {forward:.5f}m (expected >0.01m)")

    env.reset(seed=seed, options={"start_xy": (-1.4, -1.3), "source_xy": (1.35, 1.25), "yaw": 0.0})
    _, _, yaw0 = env.pose()
    for _ in range(6):
        obs, _, term, trunc, _ = env.step(2)
        if term or trunc:
            break
    _, _, yaw1 = env.pose()
    yaw_change = (yaw1 - yaw0 + math.pi) % (2 * math.pi) - math.pi
    results = {
        "forward_progress_m": float(forward),
        "turn_yaw_change_rad": float(yaw_change),
        "finite_observation": bool(np.isfinite(obs).all()),
        "wheel_motor_indices": [int(i) for i in env.motor_indices],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "smoke_result.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    if abs(yaw_change) < 0.03:
        raise RuntimeError(f"Wheel turning failed: yaw={yaw_change:.5f}rad (expected abs >0.03)")
    if not results["finite_observation"]:
        raise RuntimeError("Observation has NaN or Inf")
    print("[OK] Genesis physically moved and turned the robot.")


def train(env, args):
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.monitor import Monitor

    check_env(env, warn=True)
    monitor = Monitor(env, filename=str(args.output / "training_monitor.csv"))
    model = PPO(
        "MlpPolicy", monitor, seed=args.seed, verbose=1,
        n_steps=512, batch_size=128, n_epochs=5,
        learning_rate=3e-4, gamma=0.99, device="cpu",
        tensorboard_log=None,
    )
    model.learn(total_timesteps=args.steps, progress_bar=False)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.model))
    meta = {"algorithm": "PPO", "timesteps_requested": args.steps, "seed": args.seed,
            "backend": args.backend, "sim_dt": SIM_DT, "steps_per_action": STEPS_PER_ACTION,
            "actual_Genesis_tested_on_this_computer": True}
    (args.output / "training_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[OK] Trained checkpoint saved: {args.model}")
    print("[NOTE] Successful training completion does NOT imply high task success. Run evaluate.")


def evaluate(env, args, model):
    if model is None:
        raise FileNotFoundError(f"PPO checkpoint not found: {args.model}. Train first.")
    rng = np.random.default_rng(args.seed + 6000)
    outcomes = []
    for policy in ("random", "ppo"):
        for i in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + i + 1000)
            total_reward = 0.0
            info = {"success": False, "steps": 0, "distance_to_source": 999}
            for step in range(MAX_STEPS):
                action = action_for(policy, obs, model, rng, step, args.deterministic)
                obs, reward, term, trunc, info = env.step(action)
                total_reward += reward
                if term or trunc:
                    break
            outcomes.append({
                "policy": policy, "episode": i,
                "success": int(info["success"]), "steps": int(info["steps"]),
                "distance_to_source_m": float(info["distance_to_source"]),
                "total_reward": float(total_reward),
            })
    write_csv(args.output / "eval.csv", outcomes)
    stats = summary_of(outcomes)
    (args.output / "eval_summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print("[OK] Measured metrics saved. Check whether PPO actually beats random.")


def demo(env, args, model):
    from .viz import plot_trace
    policy = select_demo_policy(args.policy, args.model)
    if policy == "ppo" and model is None:
        raise FileNotFoundError(f"PPO checkpoint not found: {args.model}. Train first.")
    print(f"[DEMO POLICY] {policy}" +
          (" — scripted motion, NOT AI" if policy == "scripted" else ""))
    rng = np.random.default_rng(args.seed + 2000)
    limit = args.video_actions if args.record else args.demo_actions
    obs, _ = env.reset(seed=args.seed + 1000)
    record_path = args.output / "genesis_3d_demo.mp4"
    all_traces, last_trace, episodes = [], [], []
    last_source_xy, last_wind_angle = env.source_xy, env.field.wind_angle
    episode = 0
    info = {}
    if args.record:
        print(f"[RECORDING] {limit} actions * {SIM_DT * STEPS_PER_ACTION:.2f}s ~= "
              f"{limit * SIM_DT * STEPS_PER_ACTION:.1f}s of simulation")
        env.start_video(record_path, fps=15)
    try:
        for step in range(limit):
            action = action_for(policy, obs, model, rng, step, args.deterministic)
            obs, _, term, trunc, info = env.step(action)
            last_trace = env.trace
            last_source_xy, last_wind_angle = env.source_xy, env.field.wind_angle
            row = dict(last_trace[-1]); row["episode"] = episode
            all_traces.append(row)
            if term or trunc:
                episodes.append({"episode": episode, "success": bool(info["success"]), "steps": info["steps"]})
                episode += 1
                obs, _ = env.reset(seed=args.seed + 1000 + episode)
        episodes.append({"episode": episode, "success": bool(info.get("success", False)),
                         "steps": info.get("steps", 0)})
    finally:
        if args.record:
            env.stop_video()  # always finalize encoder, including on exceptions
    write_csv(args.output / "demo_trace.csv", all_traces)
    plot_trace(last_trace, last_source_xy, last_wind_angle, args.output / "demo_trajectory.png")
    report = {"policy": policy, "trained_policy": policy == "ppo", "episodes": episodes,
              "simulated_seconds": limit * SIM_DT * STEPS_PER_ACTION,
              "camera_video": str(record_path) if args.record else None}
    (args.output / "demo_metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] Trace: {args.output / 'demo_trace.csv'}")
    print(f"[OK] Plot: {args.output / 'demo_trajectory.png'}")
    if args.record:
        if not record_path.is_file() or record_path.stat().st_size == 0:
            raise RuntimeError("Genesis encoder returned, but video is missing or empty")
        print(f"[OK] 3-D MP4: {record_path} ({record_path.stat().st_size} bytes)")
    print(f"[RESULT] Policy={policy}; episodes observed={len(episodes)}")


def main(argv: list[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    if min(args.steps, args.episodes, args.demo_actions, args.video_actions) <= 0:
        raise SystemExit("All step/episode counts must be positive")
    if args.command == "doctor":
        doctor()
        return

    try:
        import genesis as gs
    except ImportError as exc:
        raise SystemExit("Genesis not installed. Run setup_windows.bat") from exc
    from .env import GenesisOdorEnv

    args.output.mkdir(parents=True, exist_ok=True)
    backend = gs.cpu if args.backend == "cpu" else gs.cuda
    gs.init(backend=backend, logging_level="warning", seed=args.seed)
    env = GenesisOdorEnv(
        backend=args.backend, render_camera=args.record,
        show_viewer=args.viewer, seed=args.seed,
    )
    try:
        if args.command == "smoke":
            physics_smoke(env, args.seed, args.output)
            return
        if args.command == "train":
            train(env, args)
            return
        model = None
        if args.model.is_file():
            from stable_baselines3 import PPO
            model = PPO.load(str(args.model), device="cpu")
        if args.command == "evaluate":
            evaluate(env, args, model)
        else:
            demo(env, args, model)
    finally:
        env.close()


if __name__ == "__main__":
    main()
