"""
Unified Physics Engine
======================

Integrates all physics modules into a single simulation engine.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from agx.config import AGXConfig, PhysicsConfig, get_config
from agx.physics.constants import Units, SI_UNITS, CONSTANTS
from agx.physics.newtonian import NewtonianEngine, Body
from agx.physics.general_relativity import GREngine, GRBody
from agx.physics.quantum_field import QuantumFieldEngine, QuantumFieldConfig
from agx.physics.speculative import SpeculativeEngine
from agx.physics.solver import NumericalSolver, SimulationState, SolverConfig


@dataclass
class SimulationResult:
    """Complete results from a simulation run."""
    states: List[SimulationState]
    energy_history: np.ndarray
    chaos_analysis: Dict[str, Any]
    final_state: SimulationState
    metadata: Dict[str, Any]


class PhysicsEngine:
    """
    Unified Multi-Scale Physics Engine.
    
    Combines Newtonian mechanics, GR approximations, quantum fields,
    and speculative physics into a single simulation framework.
    
    ⚠️ All speculative physics results are THEORETICAL ONLY.
    """
    
    DISCLAIMER = "All results contain theoretical/speculative components for educational use only."
    
    def __init__(self, config: Optional[AGXConfig | str] = None):
        """
        Initialize physics engine.
        
        Args:
            config: AGXConfig object or path to YAML config file
        """
        if isinstance(config, str):
            self.config = get_config(config)
        elif config is None:
            self.config = AGXConfig()
        else:
            self.config = config
        
        self.physics_config = self.config.physics
        self.units = SI_UNITS
        
        # Initialize sub-engines
        self.newtonian = NewtonianEngine(
            units=self.units,
            softening=self.physics_config.newtonian.softening_length,
        )
        
        self.gr = GREngine() if self.physics_config.general_relativity.enabled else None
        
        self.quantum = QuantumFieldEngine(
            QuantumFieldConfig(
                vacuum_energy_density=self.physics_config.quantum_field.vacuum_energy_density,
                fluctuation_amplitude=self.physics_config.quantum_field.fluctuation_amplitude,
            )
        ) if self.physics_config.quantum_field.enabled else None
        
        self.speculative = SpeculativeEngine(
            stability_enforcement=True,
            max_negative_ratio=self.physics_config.speculative.max_negative_mass_ratio,
        ) if self.physics_config.speculative.enabled else None
        
        self.solver = NumericalSolver(SolverConfig(
            method=self.physics_config.solver.method,
            dt=self.physics_config.dt,
            adaptive=self.physics_config.solver.adaptive_step,
            rtol=self.physics_config.solver.rtol,
            atol=self.physics_config.solver.atol,
            stochastic=self.physics_config.solver.stochastic_noise,
            noise_amplitude=self.physics_config.solver.noise_amplitude,
        ))
        
        self.time = 0.0
        self._history: List[Dict[str, Any]] = []
    
    def add_body(self, body: Body) -> None:
        """Add a body to the Newtonian simulation."""
        self.newtonian.add_body(body)
        if self.gr and abs(body.mass) > 1e10:
            self.gr.add_body(GRBody(mass=body.mass, position=body.position))
    
    def create_scenario(self, scenario: str = "two_body") -> List[Body]:
        """Create a predefined scenario."""
        if scenario == "two_body":
            return self.newtonian.create_two_body_system(m1=1.0, m2=0.1, separation=5.0)
        elif scenario == "solar_system":
            return self.newtonian.create_solar_system_simple()
        elif scenario == "three_body":
            bodies = [
                Body(name="A", mass=1.0, position=np.array([1, 0, 0]), velocity=np.array([0, 0.5, 0])),
                Body(name="B", mass=1.0, position=np.array([-1, 0, 0]), velocity=np.array([0, -0.5, 0])),
                Body(name="C", mass=1.0, position=np.array([0, 1.5, 0]), velocity=np.array([0.5, 0, 0])),
            ]
            self.newtonian.add_bodies(bodies)
            return bodies
        return []
    
    def derivative_function(self, t: float, state: np.ndarray) -> np.ndarray:
        """ODE right-hand side for scipy solver."""
        n = len(state) // 6  # Number of bodies (3 pos + 3 vel each)
        
        positions = state[:3*n].reshape(n, 3)
        velocities = state[3*n:].reshape(n, 3)
        
        # Update newtonian engine state
        for i, body in enumerate(self.newtonian.bodies[:n]):
            body.position = positions[i]
            body.velocity = velocities[i]
        
        accelerations = self.newtonian.calculate_accelerations()
        
        accel_array = np.array([
            accelerations.get(b.id, np.zeros(3)) 
            for b in self.newtonian.bodies[:n]
        ])
        
        # Add GR corrections if enabled
        if self.gr:
            for i, body in enumerate(self.newtonian.bodies[:n]):
                gr_accel = self.gr.geodesic_acceleration(positions[i], velocities[i])
                accel_array[i] += gr_accel * 0.01  # Scaled correction
        
        return np.concatenate([velocities.flatten(), accel_array.flatten()])
    
    def run_simulation(self, timesteps: Optional[int] = None, 
                       dt: Optional[float] = None) -> SimulationResult:
        """
        Run physics simulation.
        
        Args:
            timesteps: Number of timesteps (default from config)
            dt: Time step size (default from config)
        """
        timesteps = timesteps or self.config.physics.timesteps
        dt = dt or self.config.physics.dt
        duration = timesteps * dt
        
        # Initial state
        positions, velocities = self.newtonian.get_state_vectors()
        initial = SimulationState(
            time=self.time,
            positions=positions.reshape(-1, 3),
            velocities=velocities.reshape(-1, 3),
            energy=self.newtonian.total_energy(),
        )
        
        # Run solver
        states = self.solver.run_simulation(
            self.derivative_function,
            initial,
            duration,
            save_interval=max(1, timesteps // 100),
        )
        
        # Collect energy history
        energy_history = np.array([
            s.energy if s.energy else 0.0 for s in states
        ])
        
        # Update time
        self.time += duration
        
        # Chaos analysis
        trajectory = np.array([s.positions.flatten() for s in states])
        chaos = self.solver.detect_chaos(trajectory, dt)
        
        return SimulationResult(
            states=states,
            energy_history=energy_history,
            chaos_analysis=chaos,
            final_state=states[-1] if states else initial,
            metadata={
                "timesteps": timesteps,
                "dt": dt,
                "duration": duration,
                "disclaimer": self.DISCLAIMER,
            }
        )
    
    def get_curvature_field(self, size: float = 10.0, 
                           resolution: int = 32) -> Dict[str, np.ndarray]:
        """Get spacetime curvature visualization data."""
        if not self.gr:
            return {}
        
        X, Y, curvature = self.gr.curvature_grid(
            center=np.zeros(3), size=size, resolution=resolution
        )
        return {"X": X, "Y": Y, "curvature": curvature}
    
    def get_quantum_field(self, size: float = 10.0, 
                          resolution: int = 32) -> Dict[str, np.ndarray]:
        """Get quantum vacuum fluctuation field."""
        if not self.quantum:
            return {}
        
        X, Y, field = self.quantum.generate_fluctuation_field(
            (size, size), resolution
        )
        return {"X": X, "Y": Y, "field": field}
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get complete engine state for serialization."""
        return {
            "time": self.time,
            "newtonian": self.newtonian.get_state_dict(),
            "gr_enabled": self.gr is not None,
            "quantum_enabled": self.quantum is not None,
            "speculative_enabled": self.speculative is not None,
            "config": self.config.model_dump(),
        }
    
    def reset(self) -> None:
        """Reset simulation to initial state."""
        self.time = 0.0
        self.newtonian.bodies.clear()
        self._history.clear()
        if self.gr:
            self.gr.bodies.clear()
        if self.speculative:
            self.speculative.exotic_matter.clear()
