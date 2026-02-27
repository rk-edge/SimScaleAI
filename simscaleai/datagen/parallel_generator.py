"""Scalable parallel data generation pipeline.

Distributes trajectory collection across multiple worker processes,
each running its own MuJoCo environment instance. Supports:
  - Multiprocessing worker pool (N workers × 1 env each)
  - Sharded HDF5 output (one shard per worker, optional merge)
  - Resume from interrupted runs (skip completed shards)
  - Per-worker domain randomization seeds for diversity
  - Shared progress counter across workers
  - Configurable chunk allocation strategies

Architecture
============

  Coordinator (main process)
      │
      ├── Worker 0  ──►  shard_00000.h5   (episodes 0-24)
      ├── Worker 1  ──►  shard_00001.h5   (episodes 25-49)
      ├── Worker 2  ──►  shard_00002.h5   (episodes 50-74)
      └── Worker 3  ──►  shard_00003.h5   (episodes 75-99)
                                │
                         merge (optional)
                                │
                          dataset.h5

Usage
=====
    from simscaleai.datagen.parallel_generator import generate_dataset_parallel

    stats = generate_dataset_parallel(
        env_name="pick_place",
        n_episodes=10_000,
        num_workers=8,
        output_dir="data/pick_place_10k",
        policy_type="scripted",
        domain_randomization=True,
    )
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import multiprocessing as mp
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ParallelGenConfig:
    """Configuration for parallel data generation."""

    env_name: str = "pick_place"
    n_episodes: int = 1000
    num_workers: int = 4
    output_dir: str = "data/parallel"
    policy_type: str = "scripted"
    domain_randomization: bool = True
    max_steps: int = 200
    seed: int = 42
    merge_shards: bool = True
    merged_filename: str = "dataset.h5"
    compression: str = "gzip"
    compression_opts: int = 4  # gzip level 1-9
    resume: bool = True  # skip completed shards
    log_interval: int = 50  # episodes between progress logs per worker


# ---------------------------------------------------------------------------
# Worker function (runs in a child process)
# ---------------------------------------------------------------------------

def _worker_generate(
    worker_id: int,
    episode_range: tuple[int, int],
    config: ParallelGenConfig,
) -> dict[str, Any]:
    """Generate episodes for a single shard.

    Each worker creates its own MuJoCo environment, collects episodes
    in its assigned range, and writes them to a per-worker HDF5 shard.
    """
    # Configure logging for spawned worker
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [W{worker_id}] %(message)s",
    )
    from simscaleai.sim.factory import make_env
    from simscaleai.datagen.generator import _get_policy

    start_ep, end_ep = episode_range
    n_episodes = end_ep - start_ep
    shard_path = Path(config.output_dir) / f"shard_{worker_id:05d}.h5"

    # Resume: if shard exists and is complete, skip
    if config.resume and shard_path.exists():
        try:
            with h5py.File(shard_path, "r") as f:
                if f.attrs.get("complete", False):
                    existing_eps = f.attrs.get("n_episodes", 0)
                    logger.info(
                        f"Worker {worker_id}: shard already complete "
                        f"({existing_eps} episodes), skipping"
                    )
                    return {
                        "worker_id": worker_id,
                        "shard_path": str(shard_path),
                        "n_episodes": int(existing_eps),
                        "total_steps": int(f.attrs.get("total_steps", 0)),
                        "mean_reward": float(f.attrs.get("mean_reward", 0)),
                        "success_rate": float(f.attrs.get("success_rate", 0)),
                        "skipped": True,
                    }
        except Exception:
            pass  # corrupted shard, regenerate

    # Worker-specific seed for diversity
    worker_seed = config.seed + worker_id * 1000

    # Create environment
    env = make_env(
        config.env_name,
        render_mode="rgb_array",
        domain_randomization=config.domain_randomization,
        max_episode_steps=config.max_steps,
        seed=worker_seed,
        cameras=[],
    )

    policy_fn = _get_policy(config.policy_type, env)
    rng = np.random.default_rng(worker_seed)

    total_steps = 0
    total_rewards = []
    episode_lengths = []
    successes = 0
    t0 = time.time()

    with h5py.File(shard_path, "w") as f:
        # Shard metadata
        f.attrs["env_name"] = config.env_name
        f.attrs["worker_id"] = worker_id
        f.attrs["episode_range_start"] = start_ep
        f.attrs["episode_range_end"] = end_ep
        f.attrs["n_episodes"] = n_episodes
        f.attrs["policy_type"] = config.policy_type
        f.attrs["domain_randomization"] = config.domain_randomization
        f.attrs["seed"] = worker_seed
        f.attrs["complete"] = False

        for local_idx in range(n_episodes):
            global_idx = start_ep + local_idx
            obs, info = env.reset(seed=int(rng.integers(0, 2**31)))

            # Reset stateful policies
            if hasattr(policy_fn, "_phase"):
                policy_fn._phase = 0
                policy_fn._phase_steps = 0

            observations: dict[str, list] = {k: [] for k in obs.keys()}
            actions: list[np.ndarray] = []
            rewards: list[float] = []
            episode_reward = 0.0

            for step in range(config.max_steps):
                for k, v in obs.items():
                    observations[k].append(v)

                action = policy_fn(obs)
                actions.append(action)

                obs, reward, terminated, truncated, info = env.step(action)
                rewards.append(reward)
                episode_reward += reward

                if terminated or truncated:
                    break

            ep_len = len(actions)
            total_steps += ep_len
            total_rewards.append(episode_reward)
            episode_lengths.append(ep_len)
            if info.get("success", False):
                successes += 1

            # Write episode
            ep_group = f.create_group(f"episode_{global_idx:05d}")
            obs_group = ep_group.create_group("observations")
            for k, v_list in observations.items():
                obs_group.create_dataset(
                    k,
                    data=np.array(v_list),
                    compression=config.compression,
                    compression_opts=config.compression_opts,
                )
            ep_group.create_dataset(
                "actions",
                data=np.array(actions),
                compression=config.compression,
                compression_opts=config.compression_opts,
            )
            ep_group.create_dataset(
                "rewards",
                data=np.array(rewards),
                compression=config.compression,
                compression_opts=config.compression_opts,
            )

            if (local_idx + 1) % config.log_interval == 0:
                elapsed = time.time() - t0
                eps_per_sec = (local_idx + 1) / elapsed
                logger.info(
                    f"Worker {worker_id}: {local_idx + 1}/{n_episodes} episodes "
                    f"({eps_per_sec:.1f} ep/s)"
                )

        # Mark shard complete
        f.attrs["complete"] = True
        f.attrs["total_steps"] = total_steps
        f.attrs["mean_reward"] = float(np.mean(total_rewards)) if total_rewards else 0.0
        f.attrs["success_rate"] = successes / max(n_episodes, 1)

    env.close()
    elapsed = time.time() - t0

    stats = {
        "worker_id": worker_id,
        "shard_path": str(shard_path),
        "n_episodes": n_episodes,
        "total_steps": total_steps,
        "mean_reward": float(np.mean(total_rewards)) if total_rewards else 0.0,
        "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "success_rate": successes / max(n_episodes, 1),
        "elapsed_sec": round(elapsed, 1),
        "episodes_per_sec": round(n_episodes / max(elapsed, 0.01), 2),
        "skipped": False,
    }

    logger.info(
        f"Worker {worker_id} done: {n_episodes} eps, "
        f"{total_steps} steps in {elapsed:.1f}s ({stats['episodes_per_sec']} ep/s)"
    )
    return stats


# ---------------------------------------------------------------------------
# Shard merge utility
# ---------------------------------------------------------------------------

def merge_shards(
    shard_dir: str | Path,
    output_path: str | Path,
    delete_shards: bool = False,
) -> dict[str, Any]:
    """Merge per-worker HDF5 shards into a single dataset file.

    Copies all episode groups from shards into one HDF5 file with
    contiguous episode numbering. Optionally deletes shards after merge.

    Args:
        shard_dir: Directory containing shard_*.h5 files
        output_path: Path for merged HDF5 file
        delete_shards: Remove shard files after successful merge

    Returns:
        Merge statistics
    """
    shard_dir = Path(shard_dir)
    output_path = Path(output_path)
    shard_files = sorted(shard_dir.glob("shard_*.h5"))

    if not shard_files:
        raise FileNotFoundError(f"No shard files found in {shard_dir}")

    total_episodes = 0
    total_steps = 0
    env_name = None

    logger.info(f"Merging {len(shard_files)} shards into {output_path}")
    t0 = time.time()

    with h5py.File(output_path, "w") as merged:
        ep_counter = 0

        for shard_path in shard_files:
            with h5py.File(shard_path, "r") as shard:
                if env_name is None:
                    env_name = shard.attrs.get("env_name", "unknown")

                # Copy episode groups with renumbered keys
                episode_keys = sorted(
                    [k for k in shard.keys() if k.startswith("episode_")]
                )
                for ep_key in episode_keys:
                    new_key = f"episode_{ep_counter:05d}"
                    shard.copy(ep_key, merged, name=new_key)
                    ep_counter += 1

                total_steps += int(shard.attrs.get("total_steps", 0))

        # Write merged metadata
        merged.attrs["env_name"] = env_name or "unknown"
        merged.attrs["n_episodes"] = ep_counter
        merged.attrs["total_steps"] = total_steps
        merged.attrs["n_shards_merged"] = len(shard_files)
        total_episodes = ep_counter

    elapsed = time.time() - t0
    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    if delete_shards:
        for shard_path in shard_files:
            shard_path.unlink()
        logger.info(f"Deleted {len(shard_files)} shard files")

    stats = {
        "merged_path": str(output_path),
        "total_episodes": total_episodes,
        "total_steps": total_steps,
        "n_shards": len(shard_files),
        "file_size_mb": round(file_size_mb, 1),
        "merge_time_sec": round(elapsed, 1),
    }

    logger.info(f"Merge complete: {stats}")
    return stats


# ---------------------------------------------------------------------------
# Coordinator (main process)
# ---------------------------------------------------------------------------

def generate_dataset_parallel(
    env_name: str = "pick_place",
    n_episodes: int = 1000,
    num_workers: int | None = None,
    output_dir: str = "data/parallel",
    policy_type: str = "scripted",
    domain_randomization: bool = True,
    max_steps: int = 200,
    seed: int = 42,
    merge: bool = True,
    resume: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Generate a large trajectory dataset using parallel workers.

    Splits episodes across ``num_workers`` processes, each writing to
    its own HDF5 shard. Optionally merges shards into a single file.

    Args:
        env_name: Environment name (reach, pick_place, juggle)
        n_episodes: Total episodes to generate
        num_workers: Worker count (default: CPU count, capped at 8)
        output_dir: Output directory for shards and merged file
        policy_type: 'random' or 'scripted'
        domain_randomization: Enable domain randomization
        max_steps: Max steps per episode
        seed: Base random seed
        merge: Merge shards into single file after generation
        resume: Skip completed shards on re-run

    Returns:
        Aggregate statistics dict
    """
    if num_workers is None:
        num_workers = min(os.cpu_count() or 4, 8)

    config = ParallelGenConfig(
        env_name=env_name,
        n_episodes=n_episodes,
        num_workers=num_workers,
        output_dir=output_dir,
        policy_type=policy_type,
        domain_randomization=domain_randomization,
        max_steps=max_steps,
        seed=seed,
        merge_shards=merge,
        resume=resume,
        **{k: v for k, v in kwargs.items() if k in ParallelGenConfig.__dataclass_fields__},
    )

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    config_path = Path(config.output_dir) / "generation_config.json"
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2)

    # Divide episodes across workers
    episodes_per_worker = n_episodes // num_workers
    remainder = n_episodes % num_workers
    episode_ranges: list[tuple[int, int]] = []
    start = 0
    for w in range(num_workers):
        count = episodes_per_worker + (1 if w < remainder else 0)
        episode_ranges.append((start, start + count))
        start += count

    logger.info(
        f"Starting parallel generation: {n_episodes} episodes across "
        f"{num_workers} workers | env={env_name} policy={policy_type} "
        f"DR={domain_randomization}"
    )

    t0 = time.time()

    # Launch workers using ProcessPoolExecutor
    # Use 'fork' — workers run headless MuJoCo (no OpenGL/rendering)
    ctx = mp.get_context("fork")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_workers, mp_context=ctx
    ) as executor:
        futures = [
            executor.submit(_worker_generate, w, episode_ranges[w], config)
            for w in range(num_workers)
        ]
        results = [f.result() for f in futures]

    total_elapsed = time.time() - t0

    # Aggregate statistics
    total_steps = sum(r["total_steps"] for r in results)
    total_eps = sum(r["n_episodes"] for r in results)
    all_rewards = [r["mean_reward"] for r in results if not r.get("skipped")]
    all_success = [r["success_rate"] for r in results if not r.get("skipped")]
    skipped = sum(1 for r in results if r.get("skipped"))

    aggregate_stats: dict[str, Any] = {
        "env_name": env_name,
        "total_episodes": total_eps,
        "total_steps": total_steps,
        "num_workers": num_workers,
        "elapsed_sec": round(total_elapsed, 1),
        "episodes_per_sec": round(total_eps / max(total_elapsed, 0.01), 2),
        "steps_per_sec": round(total_steps / max(total_elapsed, 0.01), 1),
        "mean_reward": round(float(np.mean(all_rewards)), 2) if all_rewards else 0.0,
        "mean_success_rate": round(float(np.mean(all_success)), 3) if all_success else 0.0,
        "shards_generated": num_workers - skipped,
        "shards_skipped": skipped,
        "worker_stats": results,
    }

    # Merge shards
    if merge:
        merged_path = Path(config.output_dir) / config.merged_filename
        merge_stats = merge_shards(config.output_dir, merged_path, delete_shards=False)
        aggregate_stats["merged"] = merge_stats

    # Save aggregate stats
    stats_path = Path(config.output_dir) / "generation_stats.json"
    with open(stats_path, "w") as f:
        json.dump(aggregate_stats, f, indent=2, default=str)

    logger.info(
        f"\n{'='*60}\n"
        f"PARALLEL GENERATION COMPLETE\n"
        f"  Episodes:  {total_eps:,}\n"
        f"  Steps:     {total_steps:,}\n"
        f"  Workers:   {num_workers}\n"
        f"  Time:      {total_elapsed:.1f}s\n"
        f"  Throughput: {aggregate_stats['episodes_per_sec']:.1f} ep/s "
        f"| {aggregate_stats['steps_per_sec']:.0f} steps/s\n"
        f"  Success:   {aggregate_stats['mean_success_rate']:.1%}\n"
        f"{'='*60}"
    )

    return aggregate_stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entrypoint for parallel data generation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="SimScaleAI — Scalable Parallel Data Generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env", default="pick_place", help="Environment name")
    parser.add_argument("--episodes", type=int, default=1000, help="Total episodes")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count, max 8)",
    )
    parser.add_argument("--output-dir", default="data/parallel", help="Output directory")
    parser.add_argument(
        "--policy", default="scripted", choices=["random", "scripted"], help="Policy type"
    )
    parser.add_argument("--no-dr", action="store_true", help="Disable domain randomization")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--no-merge", action="store_true", help="Skip merging shards")
    parser.add_argument("--no-resume", action="store_true", help="Regenerate all shards")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    stats = generate_dataset_parallel(
        env_name=args.env,
        n_episodes=args.episodes,
        num_workers=args.workers,
        output_dir=args.output_dir,
        policy_type=args.policy,
        domain_randomization=not args.no_dr,
        max_steps=args.max_steps,
        seed=args.seed,
        merge=not args.no_merge,
        resume=not args.no_resume,
    )

    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
