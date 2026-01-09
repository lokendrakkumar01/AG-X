"""
High-Precision Numerical Solver Engine
=======================================

Adaptive solvers, symplectic integrators, chaos detection, and stability analysis.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, Dict, Any, List
from enum import Enum
from scipy.integrate import solve_ivp, OdeSolution


class SolverMethod(str, Enum):
    """Available integration methods."""
    EULER = "euler"
    LEAPFROG = "leapfrog"
    RK4 = "rk4"
    RK45 = "rk45"
    DOP853 = "dop853"
    RADAU = "radau"


@dataclass
class SimulationState:
    """Complete state of a simulation at a time point."""
    time: float
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: Optional[np.ndarray] = None
    energy: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def copy(self) -> "SimulationState":
        return SimulationState(
            time=self.time,
            positions=self.positions.copy(),
            velocities=self.velocities.copy(),
            accelerations=self.accelerations.copy() if self.accelerations is not None else None,
            energy=self.energy,
            metadata=self.metadata.copy(),
        )


@dataclass
class SolverConfig:
    """Solver configuration."""
    method: SolverMethod = SolverMethod.DOP853
    dt: float = 0.01
    adaptive: bool = True
    rtol: float = 1e-8
    atol: float = 1e-10
    max_step: float = 1.0
    min_step: float = 1e-12
    stochastic: bool = False
    noise_amplitude: float = 1e-10


class NumericalSolver:
    """
    High-precision numerical solver for physics simulations.
    
    Features:
    - Multiple integration methods (Euler to DOP853)
    - Adaptive step sizing
    - Symplectic integrators for energy conservation
    - Stochastic differential equation support
    - Lyapunov exponent calculation for chaos detection
    """
    
    def __init__(self, config: Optional[SolverConfig] = None):
        self.config = config or SolverConfig()
        self.history: List[SimulationState] = []
        self._rng = np.random.default_rng(42)
        self._lyapunov_buffer: List[float] = []
    
    def solve(self, 
              derivative_fn: Callable[[float, np.ndarray], np.ndarray],
              initial_state: np.ndarray,
              t_span: Tuple[float, float],
              t_eval: Optional[np.ndarray] = None) -> OdeSolution:
        """
        Solve ODE system using scipy's solve_ivp.
        
        Args:
            derivative_fn: Function f(t, y) returning dy/dt
            initial_state: Initial state vector
            t_span: (t_start, t_end)
            t_eval: Optional array of times to evaluate at
        """
        method_map = {
            SolverMethod.RK45: "RK45",
            SolverMethod.DOP853: "DOP853",
            SolverMethod.RADAU: "Radau",
        }
        
        method = method_map.get(self.config.method, "DOP853")
        
        return solve_ivp(
            derivative_fn,
            t_span,
            initial_state,
            method=method,
            t_eval=t_eval,
            rtol=self.config.rtol,
            atol=self.config.atol,
            max_step=self.config.max_step,
        )
    
    def step_rk4(self, derivative_fn: Callable, state: np.ndarray, 
                 t: float, dt: float) -> np.ndarray:
        """Classic 4th-order Runge-Kutta step."""
        k1 = derivative_fn(t, state)
        k2 = derivative_fn(t + dt/2, state + dt*k1/2)
        k3 = derivative_fn(t + dt/2, state + dt*k2/2)
        k4 = derivative_fn(t + dt, state + dt*k3)
        
        return state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    
    def step_leapfrog(self, accel_fn: Callable, pos: np.ndarray, 
                      vel: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Symplectic leapfrog (Verlet) integrator."""
        # Half kick
        a = accel_fn(pos)
        vel_half = vel + 0.5 * dt * a
        
        # Full drift
        pos_new = pos + dt * vel_half
        
        # Half kick with new acceleration
        a_new = accel_fn(pos_new)
        vel_new = vel_half + 0.5 * dt * a_new
        
        return pos_new, vel_new
    
    def add_stochastic_noise(self, state: np.ndarray, dt: float) -> np.ndarray:
        """Add Wiener process noise for SDE simulation."""
        if not self.config.stochastic:
            return state
        
        noise = self._rng.normal(0, 1, state.shape)
        return state + self.config.noise_amplitude * np.sqrt(dt) * noise
    
    def calculate_lyapunov_exponent(self, trajectory: np.ndarray, 
                                     dt: float, 
                                     dimension: int = 3) -> float:
        """
        Estimate largest Lyapunov exponent from trajectory.
        
        Uses the method of monitoring divergence of nearby trajectories.
        λ > 0 indicates chaos.
        """
        if len(trajectory) < 100:
            return 0.0
        
        # Take differences between successive points
        diffs = np.diff(trajectory, axis=0)
        norms = np.linalg.norm(diffs.reshape(-1, dimension), axis=1)
        
        # Avoid log(0)
        norms = np.maximum(norms, 1e-15)
        
        # Lyapunov approximation: λ ≈ (1/t) * Σ log(|δx(t)|/|δx(0)|)
        log_ratios = np.log(norms[1:] / norms[:-1])
        lyapunov = np.mean(log_ratios) / dt
        
        return lyapunov
    
    def detect_chaos(self, trajectory: np.ndarray, dt: float, 
                    threshold: float = 0.01) -> Dict[str, Any]:
        """Detect chaotic behavior in trajectory."""
        lyapunov = self.calculate_lyapunov_exponent(trajectory, dt)
        
        # Calculate other chaos indicators
        velocity_changes = np.diff(trajectory, axis=0)
        velocity_std = np.std(velocity_changes)
        
        return {
            "lyapunov_exponent": lyapunov,
            "is_chaotic": lyapunov > threshold,
            "velocity_variance": velocity_std,
            "chaos_strength": "strong" if lyapunov > 0.1 else "weak" if lyapunov > threshold else "none",
        }
    
    def energy_conservation_check(self, energies: np.ndarray, 
                                  tolerance: float = 1e-6) -> Dict[str, Any]:
        """Check energy conservation quality."""
        if len(energies) < 2:
            return {"conserved": True, "error": 0.0}
        
        initial = energies[0]
        if abs(initial) < 1e-30:
            initial = 1.0
        
        relative_errors = np.abs(energies - initial) / abs(initial)
        max_error = np.max(relative_errors)
        mean_error = np.mean(relative_errors)
        
        return {
            "conserved": max_error < tolerance,
            "max_relative_error": max_error,
            "mean_relative_error": mean_error,
            "energy_drift": energies[-1] - energies[0],
        }
    
    def stability_analysis(self, jacobian_fn: Callable[[np.ndarray], np.ndarray],
                          equilibrium: np.ndarray) -> Dict[str, Any]:
        """Analyze stability at an equilibrium point via eigenvalues."""
        J = jacobian_fn(equilibrium)
        eigenvalues = np.linalg.eigvals(J)
        
        real_parts = np.real(eigenvalues)
        max_real = np.max(real_parts)
        
        if max_real < -1e-10:
            stability = "stable"
        elif max_real > 1e-10:
            stability = "unstable"
        else:
            stability = "marginally_stable"
        
        return {
            "eigenvalues": eigenvalues.tolist(),
            "max_real_part": max_real,
            "stability": stability,
            "is_stable": max_real < 0,
        }
    
    def adaptive_step(self, error: float, dt: float, 
                     safety: float = 0.9, p: int = 5) -> float:
        """Calculate optimal step size based on error."""
        if error < 1e-15:
            return min(dt * 2, self.config.max_step)
        
        optimal = safety * dt * (self.config.atol / error)**(1/p)
        return np.clip(optimal, self.config.min_step, self.config.max_step)
    
    def run_simulation(self, 
                       derivative_fn: Callable,
                       initial_state: SimulationState,
                       duration: float,
                       save_interval: int = 1) -> List[SimulationState]:
        """
        Run complete simulation with state tracking.
        
        Args:
            derivative_fn: ODE right-hand side
            initial_state: Initial simulation state
            duration: Total simulation time
            save_interval: Save state every N steps
        """
        self.history = [initial_state.copy()]
        
        state_vec = np.concatenate([
            initial_state.positions.flatten(),
            initial_state.velocities.flatten()
        ])
        
        t = initial_state.time
        dt = self.config.dt
        step_count = 0
        
        while t < initial_state.time + duration:
            state_vec = self.step_rk4(derivative_fn, state_vec, t, dt)
            
            if self.config.stochastic:
                state_vec = self.add_stochastic_noise(state_vec, dt)
            
            t += dt
            step_count += 1
            
            if step_count % save_interval == 0:
                n = len(state_vec) // 2
                current_state = SimulationState(
                    time=t,
                    positions=state_vec[:n].reshape(initial_state.positions.shape),
                    velocities=state_vec[n:].reshape(initial_state.velocities.shape),
                )
                self.history.append(current_state)
        
        return self.history
