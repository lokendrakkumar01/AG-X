"""
AG-X 2026 CLI Application
==========================

Command-line interface for running simulations and experiments.
"""

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from loguru import logger

from agx import DISCLAIMER, __version__
from agx.config import get_config, setup_logging, AGXConfig
from agx.physics import PhysicsEngine
from agx.experiments import ExperimentManager
from agx.viz import Renderer

console = Console()


def print_banner():
    """Print the AG-X banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║     █████╗  ██████╗       ██╗  ██╗    ██████╗  ██████╗    ║
    ║    ██╔══██╗██╔════╝       ╚██╗██╔╝    ╚════██╗██╔═████╗   ║
    ║    ███████║██║  ███╗█████╗ ╚███╔╝      █████╔╝██║██╔██║   ║
    ║    ██╔══██║██║   ██║╚════╝ ██╔██╗     ██╔═══╝ ████╔╝██║   ║
    ║    ██║  ██║╚██████╔╝      ██╔╝ ██╗    ███████╗╚██████╔╝   ║
    ║    ╚═╝  ╚═╝ ╚═════╝       ╚═╝  ╚═╝    ╚══════╝ ╚═════╝    ║
    ║                                                           ║
    ║       Advanced Gravity Research Simulation Platform       ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold cyan")
    console.print(Panel(
        "[yellow]⚠️ All simulations are THEORETICAL and for EDUCATIONAL purposes only.[/yellow]",
        title="Disclaimer",
        border_style="yellow"
    ))


@click.group()
@click.version_option(version=__version__, prog_name="AG-X")
def cli():
    """AG-X 2026 - Advanced Gravity Research Simulation Platform"""
    setup_logging()


@cli.command()
@click.option("--config", "-c", default=None, help="Path to config YAML file")
@click.option("--timesteps", "-t", default=1000, help="Number of simulation timesteps")
@click.option("--dt", default=0.01, help="Time step size")
@click.option("--scenario", "-s", default="two_body", 
              type=click.Choice(["two_body", "three_body", "solar_system"]),
              help="Preset scenario to run")
@click.option("--output", "-o", default=None, help="Output directory for results")
def simulate(config, timesteps, dt, scenario, output):
    """Run a physics simulation."""
    print_banner()
    
    console.print(f"\n[bold green]Starting simulation...[/bold green]")
    console.print(f"  • Scenario: {scenario}")
    console.print(f"  • Timesteps: {timesteps}")
    console.print(f"  • dt: {dt}")
    
    # Load configuration
    cfg = get_config(config)
    cfg.physics.timesteps = timesteps
    cfg.physics.dt = dt
    
    # Create engine and run
    engine = PhysicsEngine(cfg)
    engine.create_scenario(scenario)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running simulation...", total=None)
        result = engine.run_simulation()
        progress.update(task, completed=True, description="Simulation complete!")
    
    # Display results
    console.print("\n[bold]Results:[/bold]")
    console.print(f"  • Final time: {result.final_state.time:.4f}")
    console.print(f"  • Energy conservation: {100*(1-engine.solver.energy_conservation_check(result.energy_history).get('max_relative_error', 0)):.4f}%")
    
    if result.chaos_analysis.get("is_chaotic"):
        console.print("  • [red]⚠️ Chaotic behavior detected![/red]")
    else:
        console.print("  • [green]✓ System is stable[/green]")
    
    console.print("\n[dim]All results are theoretical simulations only.[/dim]")


@cli.command()
@click.option("--name", "-n", required=True, help="Experiment name")
@click.option("--config", "-c", default=None, help="Path to config YAML file")
@click.option("--seed", default=42, help="Random seed for reproducibility")
@click.option("--tags", "-t", multiple=True, help="Tags for the experiment")
def experiment(name, config, seed, tags):
    """Create and run a reproducible experiment."""
    print_banner()
    
    console.print(f"\n[bold green]Creating experiment: {name}[/bold green]")
    
    cfg = get_config(config) if config else AGXConfig()
    
    manager = ExperimentManager()
    exp = manager.create_experiment(
        name=name,
        config=cfg.model_dump(),
        seed=seed,
        tags=list(tags),
    )
    
    console.print(f"  • Experiment ID: {exp.id}")
    console.print(f"  • Random seed: {seed}")
    
    manager.start_run()
    
    # Run simulation
    engine = PhysicsEngine(cfg)
    engine.create_scenario("two_body")
    result = engine.run_simulation()
    
    # Log results
    manager.log_metric("final_energy", result.final_state.energy or 0)
    manager.log_result("chaos_analysis", result.chaos_analysis)
    
    exp = manager.complete_run()
    console.print(f"\n[bold green]✓ Experiment completed![/bold green]")
    console.print(f"  • Results saved to: experiments/{exp.id}/")


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", "-p", default=8050, help="Port number")
@click.option("--debug", is_flag=True, help="Enable debug mode")
def web(host, port, debug):
    """Launch the web dashboard."""
    print_banner()
    console.print(f"\n[bold green]Starting web dashboard...[/bold green]")
    console.print(f"  • URL: http://localhost:{port}")
    
    from agx.viz.dashboard import Dashboard
    dashboard = Dashboard()
    dashboard.run(host=host, port=port, debug=debug)


@cli.command()
def learn():
    """Start interactive learning mode."""
    print_banner()
    
    from agx.advanced.education import EducationMode
    edu = EducationMode()
    
    console.print("\n[bold]Available Learning Modules:[/bold]\n")
    
    for mod in edu.list_modules():
        console.print(f"  • [cyan]{mod['name']}[/cyan] ({mod['difficulty']})")
        console.print(f"    {mod['description']}")
        console.print(f"    Steps: {mod['steps']}\n")


@cli.command()
@click.option("--list", "list_experiments", is_flag=True, help="List all experiments")
@click.option("--id", "exp_id", default=None, help="Experiment ID to view")
def experiments(list_experiments, exp_id):
    """Manage experiments."""
    manager = ExperimentManager()
    
    if list_experiments:
        exps = manager.list_experiments()
        console.print(f"\n[bold]Experiments ({len(exps)}):[/bold]\n")
        for exp in exps[:10]:
            status_icon = "✓" if exp.status == "completed" else "○"
            console.print(f"  {status_icon} [{exp.id}] {exp.name} ({exp.status})")
    
    elif exp_id:
        exp = manager.load_experiment(exp_id)
        console.print(f"\n[bold]Experiment: {exp.name}[/bold]")
        console.print(f"  • ID: {exp.id}")
        console.print(f"  • Status: {exp.status}")
        console.print(f"  • Created: {exp.created_at}")
        console.print(f"  • Seed: {exp.random_seed}")


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
