"""Quick test script for AG-X 2026"""
from agx import PhysicsEngine

# Create engine
engine = PhysicsEngine()
print("1. PhysicsEngine created successfully")

# Create a two-body scenario
bodies = engine.create_scenario("two_body")
print(f"2. Created two-body scenario with {len(bodies)} bodies")

# Run simulation
result = engine.run_simulation(timesteps=100)
print(f"3. Simulation completed!")
print(f"   - States recorded: {len(result.states)}")
print(f"   - Final time: {result.final_state.time:.4f}")
print(f"   - Chaotic: {result.chaos_analysis.get('is_chaotic', False)}")

# Test visualization
from agx.viz import Renderer
renderer = Renderer()
print("4. Renderer created successfully")

# Test AI
from agx.ai import ParameterExplorer
explorer = ParameterExplorer(input_dim=2)
print("5. AI Explorer created successfully")

print("\n=== All tests passed! ===")
