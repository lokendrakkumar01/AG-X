"""
AI Optimization Example
========================

Demonstrates AI-assisted parameter optimization using Bayesian exploration.
"""

import numpy as np
from agx.ai import ParameterExplorer, RLOptimizer
from agx.advanced import EvolutionaryOptimizer


def physics_objective(params):
    """
    Simulated physics objective function.
    
    In a real use case, this would run a physics simulation
    and return a metric like anti-gravity efficiency.
    """
    # Example: Rosenbrock-like function (inverted for maximization)
    x, y = params[0], params[1]
    value = -((1 - x)**2 + 100 * (y - x**2)**2)
    
    # Add noise to simulate stochastic physics
    value += np.random.normal(0, 0.1)
    
    return value


def main():
    print("=" * 60)
    print("AG-X 2026 - AI Optimization Example")
    print("=" * 60)
    print("\n⚠️  All results are THEORETICAL simulations only.\n")
    
    # Define parameter bounds
    bounds = np.array([
        [-5, 5],   # Parameter 1
        [-5, 5],   # Parameter 2
    ])
    
    # =========================================
    # Method 1: Bayesian Optimization
    # =========================================
    print("=" * 40)
    print("Method 1: Bayesian Optimization")
    print("=" * 40)
    
    explorer = ParameterExplorer(input_dim=2)
    
    print("\nExploring parameter space...")
    result = explorer.explore(
        objective_fn=physics_objective,
        bounds=bounds,
        n_initial=10,
        n_iterations=20,
        batch_size=3,
    )
    
    print(f"\nBest parameters found: {result.best_parameters}")
    print(f"Best value: {result.best_value:.4f}")
    print(f"Total evaluations: {len(result.all_values)}")
    
    # =========================================
    # Method 2: Evolutionary Algorithm
    # =========================================
    print("\n" + "=" * 40)
    print("Method 2: Evolutionary Optimization")
    print("=" * 40)
    
    evo = EvolutionaryOptimizer(
        gene_bounds=bounds,
        population_size=20,
        mutation_rate=0.1,
    )
    
    print("\nEvolving population...")
    best = evo.optimize(
        fitness_fn=physics_objective,
        n_generations=30,
        early_stop_generations=10,
    )
    
    print(f"\nBest individual: {best.genes}")
    print(f"Best fitness: {best.fitness:.4f}")
    print(f"Generations: {evo.generation}")
    
    # =========================================
    # Method 3: Reinforcement Learning
    # =========================================
    print("\n" + "=" * 40)
    print("Method 3: RL Optimization")
    print("=" * 40)
    
    rl_opt = RLOptimizer(algorithm="PPO")
    
    print("\nTraining RL agent (this may take a moment)...")
    rl_result = rl_opt.optimize(
        objective_fn=physics_objective,
        param_dim=2,
        total_timesteps=1000,  # Short for demo
    )
    
    print(f"\nBest action found: {rl_result.best_action}")
    print(f"Best reward: {rl_result.best_reward:.4f}")
    
    # =========================================
    # Summary
    # =========================================
    print("\n" + "=" * 60)
    print("OPTIMIZATION SUMMARY")
    print("=" * 60)
    print("\nAll methods attempted to find the optimum near (1, 1)")
    print("The Rosenbrock function has its minimum there.")
    print("\nNote: These are simplified examples. Real physics")
    print("simulations would replace the objective function.")
    print("\n⚠️  All results are for educational purposes only.")


if __name__ == "__main__":
    main()
