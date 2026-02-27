"""SimScaleAI CLI — unified entry point for all platform operations.

Commands:
    simscale train      — Launch model training
    simscale eval       — Evaluate a checkpoint in simulation
    simscale datagen    — Generate synthetic datasets
    simscale record     — Record demonstrations
    simscale rl         — Run RL training
    simscale list-envs  — List available environments
    simscale list-models — List available models
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="simscale",
    help="SimScaleAI — Robotics AI Training & Simulation Platform",
    no_args_is_help=True,
)
console = Console()


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


@app.command()
def train(
    model: str = typer.Option("bc", help="Model name (bc, vla)"),
    dataset: str = typer.Option("", help="Path to HDF5 dataset (empty = dummy data)"),
    batch_size: int = typer.Option(32, help="Batch size"),
    max_steps: int = typer.Option(1000, help="Max training steps"),
    lr: float = typer.Option(1e-4, help="Learning rate"),
    checkpoint_dir: str = typer.Option("checkpoints", help="Checkpoint save directory"),
    use_wandb: bool = typer.Option(False, help="Enable WandB logging"),
    device: str = typer.Option("auto", help="Device: auto, cuda, mps, cpu"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Train a model on trajectory data."""
    _setup_logging(verbose)

    from torch.utils.data import DataLoader

    from simscaleai.models.registry import create_model
    from simscaleai.training.data.dataset import DummyTrajectoryDataset
    from simscaleai.training.trainer import TrainConfig, Trainer

    console.print(f"[bold green]Training model:[/bold green] {model}")

    # Create dataset
    if dataset:
        from simscaleai.training.data.dataset import TrajectoryDataset
        train_ds = TrajectoryDataset(dataset, obs_keys=["state"])
    else:
        console.print("[yellow]No dataset specified — using dummy data[/yellow]")
        use_image = model == "vla"
        train_ds = DummyTrajectoryDataset(
            num_samples=max(1000, max_steps * batch_size // 10),
            include_image=use_image,
            include_language=use_image,
        )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    # Create model
    model_kwargs = {}
    if model == "vla":
        model_kwargs = {"image_size": 128, "embed_dim": 128, "num_heads": 4, "num_layers": 2}
    net = create_model(model, **model_kwargs)

    # Train
    config = TrainConfig(
        model_name=model,
        batch_size=batch_size,
        max_steps=max_steps,
        lr=lr,
        checkpoint_dir=checkpoint_dir,
        use_wandb=use_wandb,
        device=device,
        log_every=max(1, max_steps // 20),
        save_every=max(1, max_steps // 5),
    )

    trainer = Trainer(model=net, train_loader=train_loader, config=config)
    metrics = trainer.train()

    console.print(f"[bold green]Training complete![/bold green] Final metrics: {metrics}")


@app.command()
def eval(
    checkpoint: str = typer.Argument(..., help="Path to model checkpoint"),
    env_name: str = typer.Option("reach", help="Environment name"),
    n_episodes: int = typer.Option(20, help="Number of evaluation episodes"),
    render: bool = typer.Option(False, help="Render simulation"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Evaluate a trained model in simulation (closed-loop)."""
    _setup_logging(verbose)

    import torch

    from simscaleai.models.registry import create_model
    from simscaleai.rl.evaluator import EvalConfig, evaluate_policy
    from simscaleai.sim.factory import make_env

    console.print(f"[bold blue]Evaluating:[/bold blue] {checkpoint} on {env_name}")

    # Load model
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_name = ckpt.get("config", {})
    if hasattr(model_name, "model_name"):
        model_name = model_name.model_name
    else:
        model_name = "bc"

    net = create_model(model_name)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    # Create env
    render_mode = "human" if render else None
    env = make_env(env_name, render_mode=render_mode)

    # Evaluate
    eval_config = EvalConfig(n_episodes=n_episodes, render=render)
    metrics = evaluate_policy(
        env=env,
        predict_fn=lambda obs: net.predict(obs),
        config=eval_config,
    )

    # Display results
    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for k, v in metrics.items():
        table.add_row(k, f"{v:.4f}")
    console.print(table)

    env.close()


@app.command()
def datagen(
    env_name: str = typer.Option("reach", help="Environment name"),
    n_episodes: int = typer.Option(100, help="Number of episodes to generate"),
    output: str = typer.Option("data/dataset.h5", help="Output HDF5 path"),
    policy: str = typer.Option("random", help="Data collection policy: random, scripted"),
    randomize: bool = typer.Option(True, help="Enable domain randomization"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Generate synthetic trajectory datasets from simulation."""
    _setup_logging(verbose)

    from simscaleai.datagen.generator import generate_dataset

    console.print(
        f"[bold magenta]Generating data:[/bold magenta] "
        f"{n_episodes} episodes from {env_name} → {output}"
    )

    stats = generate_dataset(
        env_name=env_name,
        n_episodes=n_episodes,
        output_path=output,
        policy_type=policy,
        domain_randomization=randomize,
    )

    table = Table(title="Dataset Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for k, v in stats.items():
        table.add_row(k, str(v))
    console.print(table)


@app.command()
def rl(
    env_name: str = typer.Option("reach", help="Environment name"),
    total_steps: int = typer.Option(100_000, help="Total training timesteps"),
    lr: float = typer.Option(3e-4, help="Learning rate"),
    save_path: str = typer.Option("checkpoints/ppo_agent.pt", help="Save path"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Train an RL agent (PPO) in simulation."""
    _setup_logging(verbose)

    from simscaleai.rl.agents.ppo import PPOAgent, PPOConfig
    from simscaleai.sim.factory import make_env

    console.print(f"[bold yellow]RL Training:[/bold yellow] PPO on {env_name}")

    env = make_env(env_name)

    # Determine obs/action dims
    obs, _ = env.reset()
    import numpy as np
    obs_dim = sum(v.flatten().shape[0] for k, v in obs.items() if k != "image")
    action_dim = env.action_space.shape[0]

    config = PPOConfig(total_timesteps=total_steps, lr=lr)
    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=config)

    metrics = agent.train(env)

    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    agent.save(save_path)
    console.print(f"[bold green]Saved agent to {save_path}[/bold green]")

    env.close()


@app.command(name="list-envs")
def list_envs() -> None:
    """List available simulation environments."""
    from simscaleai.sim.factory import list_envs as _list_envs

    table = Table(title="Available Environments")
    table.add_column("Name", style="cyan")
    for name in _list_envs():
        table.add_row(name)
    console.print(table)


@app.command(name="list-models")
def list_models() -> None:
    """List available model architectures."""
    from simscaleai.models.registry import list_models as _list_models

    table = Table(title="Available Models")
    table.add_column("Name", style="cyan")
    for name in _list_models():
        table.add_row(name)
    console.print(table)


# ── Visualization Commands ─────────────────────────────────────────────


@app.command(name="viz-env")
def viz_env(
    env_name: str = typer.Option("reach", help="Environment name"),
    n_steps: int = typer.Option(20, help="Number of steps to render"),
    save: str = typer.Option("", help="Save image to this path (PNG)"),
    seed: int = typer.Option(42, help="Random seed"),
) -> None:
    """Visualize simulation environment as a frame grid."""
    from simscaleai.tools.visualize import render_env_grid

    save_path = save if save else None
    render_env_grid(env_name=env_name, n_steps=n_steps, seed=seed, save_path=save_path)
    if save_path:
        console.print(f"[bold green]Saved to {save_path}[/bold green]")


@app.command(name="viz-cameras")
def viz_cameras(
    env_name: str = typer.Option("reach", help="Environment name"),
    save: str = typer.Option("", help="Save image to this path (PNG)"),
    seed: int = typer.Option(42, help="Random seed"),
) -> None:
    """Visualize camera modalities (RGB, depth, segmentation)."""
    from simscaleai.tools.visualize import render_camera_modalities

    save_path = save if save else None
    render_camera_modalities(env_name=env_name, seed=seed, save_path=save_path)
    if save_path:
        console.print(f"[bold green]Saved to {save_path}[/bold green]")


@app.command(name="viz-dataset")
def viz_dataset(
    data_path: str = typer.Argument(..., help="Path to HDF5 dataset"),
    save: str = typer.Option("", help="Save image to this path (PNG)"),
) -> None:
    """Visualize statistics of an HDF5 trajectory dataset."""
    from simscaleai.tools.visualize import plot_dataset_stats

    save_path = save if save else None
    stats = plot_dataset_stats(data_path=data_path, save_path=save_path)

    table = Table(title="Dataset Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for k, v in stats.items():
        if isinstance(v, float):
            table.add_row(k, f"{v:.4f}")
        elif isinstance(v, list):
            table.add_row(k, str([f"{x:.3f}" for x in v]))
        else:
            table.add_row(k, str(v))
    console.print(table)
    if save_path:
        console.print(f"[bold green]Saved to {save_path}[/bold green]")


@app.command(name="viz-trajectory")
def viz_trajectory(
    data_path: str = typer.Argument(..., help="Path to HDF5 dataset"),
    episode: int = typer.Option(0, help="Episode index to plot"),
    save: str = typer.Option("", help="Save image to this path (PNG)"),
) -> None:
    """Plot a single trajectory from a dataset."""
    from simscaleai.tools.visualize import plot_trajectory

    save_path = save if save else None
    plot_trajectory(data_path=data_path, episode_idx=episode, save_path=save_path)
    if save_path:
        console.print(f"[bold green]Saved to {save_path}[/bold green]")


@app.command(name="viz-live")
def viz_live(
    env_name: str = typer.Option("reach", help="Environment name"),
    n_episodes: int = typer.Option(3, help="Number of episodes"),
    seed: int = typer.Option(42, help="Random seed"),
) -> None:
    """Launch interactive MuJoCo viewer with random actions."""
    from simscaleai.tools.visualize import run_interactive

    console.print(f"[bold]Launching interactive viewer for [cyan]{env_name}[/cyan]...[/bold]")
    run_interactive(env_name=env_name, n_episodes=n_episodes, seed=seed)


if __name__ == "__main__":
    app()
