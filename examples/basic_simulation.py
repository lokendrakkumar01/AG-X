"""
Basic Simulation Example
=========================

Demonstrates how to run a simple two-body gravitational simulation.
"""

import numpy as np
from agx import PhysicsEngine
from agx.physics import Body
from agx.viz import Renderer


def main():
    print("=" * 60)
    print("AG-X 2026 - Basic Simulation Example")
    print("=" * 60)
    print("\n⚠️  All results are THEORETICAL simulations only.\n")
    
    # Create physics engine with default configuration
    engine = PhysicsEngine()
    
    # Create a two-body system (like Earth-Moon)
    print("Creating two-body system...")
    bodies = engine.create_scenario("two_body")
    
    print(f"  • Body 1: mass={bodies[0].mass}, pos={bodies[0].position}")
    print(f"  • Body 2: mass={bodies[1].mass}, pos={bodies[1].position}")
    
    # Run simulation
    print("\nRunning simulation (1000 timesteps)...")
    result = engine.run_simulation(timesteps=1000, dt=0.01)
    
    # Display results
    print("\n" + "=" * 40)
    print("RESULTS")
    print("=" * 40)
    
    print(f"\nSimulation completed!")
    print(f"  • Total time: {result.final_state.time:.2f}")
    print(f"  • States recorded: {len(result.states)}")
    
    # Energy conservation check
    initial_energy = result.energy_history[0]
    final_energy = result.energy_history[-1]
    energy_drift = abs(final_energy - initial_energy) / abs(initial_energy) * 100
    print(f"  • Energy drift: {energy_drift:.4f}%")
    
    # Chaos analysis
    if result.chaos_analysis.get("is_chaotic"):
        print("  • ⚠️  Chaotic behavior detected!")
        print(f"    Lyapunov exponent: {result.chaos_analysis['lyapunov_exponent']:.4f}")
    else:
        print("  • ✓ System is stable")
    
    # Create visualization
    print("\nGenerating visualization...")
    renderer = Renderer()
    
    # Get final positions
    final_positions = result.final_state.positions
    masses = np.array([b.mass for b in engine.newtonian.bodies])
    
    fig = renderer.render_particles_2d(final_positions, masses)
    
    # Save to HTML
    output_file = "basic_simulation_result.html"
    fig.write_html(output_file)
    print(f"  • Saved visualization to: {output_file}")
    
    print("\n" + "=" * 60)
    print("Simulation complete! All results are theoretical only.")
    print("=" * 60)


if __name__ == "__main__":
    main()
