"""
AG-X 2026 Physics Tests
========================
"""

import pytest
import numpy as np

from agx.physics.constants import PhysicsConstants, Units, CONSTANTS
from agx.physics.newtonian import NewtonianEngine, Body
from agx.physics.general_relativity import GREngine, GRBody
from agx.physics.solver import NumericalSolver, SimulationState


class TestPhysicsConstants:
    """Tests for physics constants module."""
    
    def test_gravitational_constant(self):
        """G should be approximately 6.674e-11."""
        assert abs(CONSTANTS.G - 6.67430e-11) < 1e-15
    
    def test_speed_of_light(self):
        """c should be exactly 299792458 m/s."""
        assert CONSTANTS.c == 299792458.0
    
    def test_planck_length(self):
        """Planck length should be ~1.6e-35 m."""
        l_p = CONSTANTS.l_P
        assert 1e-36 < l_p < 1e-34
    
    def test_schwarzschild_radius(self):
        """Test Schwarzschild radius calculation."""
        from agx.physics.constants import get_schwarzschild_radius
        
        # For the Sun (1 solar mass)
        r_s = get_schwarzschild_radius(1.989e30)
        assert abs(r_s - 2954) < 10  # ~3 km


class TestNewtonianEngine:
    """Tests for Newtonian gravity engine."""
    
    def test_create_engine(self):
        """Should create engine with default settings."""
        engine = NewtonianEngine()
        assert engine.G > 0
        assert len(engine.bodies) == 0
    
    def test_add_body(self):
        """Should add bodies correctly."""
        engine = NewtonianEngine()
        body = Body(mass=1.0, position=np.array([1, 0, 0]))
        engine.add_body(body)
        
        assert len(engine.bodies) == 1
        assert engine.bodies[0].mass == 1.0
    
    def test_gravitational_force(self):
        """Force should follow inverse square law."""
        engine = NewtonianEngine()
        
        body1 = Body(mass=1.0, position=np.zeros(3))
        body2 = Body(mass=1.0, position=np.array([1, 0, 0]))
        
        engine.add_bodies([body1, body2])
        
        force = engine.calculate_force(body1, body2)
        
        # Force should be attractive (pointing toward body2)
        assert force.force_vector[0] > 0
        assert force.distance == pytest.approx(1.0, rel=1e-6)
    
    def test_two_body_system(self):
        """Two-body system should conserve momentum."""
        engine = NewtonianEngine()
        engine.create_two_body_system(m1=1.0, m2=1.0, separation=1.0)
        
        initial_momentum = engine.total_momentum()
        
        # Run a few steps
        for _ in range(10):
            engine.step_leapfrog(0.01)
        
        final_momentum = engine.total_momentum()
        
        # Momentum should be conserved
        np.testing.assert_array_almost_equal(initial_momentum, final_momentum, decimal=10)
    
    def test_energy_conservation(self):
        """Energy should be approximately conserved with leapfrog."""
        engine = NewtonianEngine()
        engine.create_two_body_system(m1=1.0, m2=0.1, separation=1.0)
        
        initial_energy = engine.total_energy()
        
        # Run simulation
        for _ in range(100):
            engine.step_leapfrog(0.001)
        
        final_energy = engine.total_energy()
        
        # Energy should be conserved within 0.1%
        assert abs(final_energy - initial_energy) / abs(initial_energy) < 0.001


class TestGREngine:
    """Tests for General Relativity engine."""
    
    def test_minkowski_metric(self):
        """Flat spacetime metric should be diagonal."""
        engine = GREngine()
        eta = engine.minkowski_metric()
        
        assert eta.shape == (4, 4)
        assert eta[0, 0] < 0  # Time component is negative
        assert eta[1, 1] > 0  # Space components are positive
    
    def test_time_dilation_far_from_mass(self):
        """Far from mass, time dilation should approach 1."""
        engine = GREngine()
        engine.add_body(GRBody(mass=1e20, position=np.zeros(3)))
        
        far_position = np.array([1e10, 0, 0])
        dilation = engine.gravitational_time_dilation(far_position)
        
        assert dilation > 0.99
    
    def test_time_dilation_near_mass(self):
        """Close to mass, time should run slower."""
        engine = GREngine()
        engine.add_body(GRBody(mass=1e30, position=np.zeros(3)))
        
        near_position = np.array([1e8, 0, 0])
        dilation = engine.gravitational_time_dilation(near_position)
        
        assert dilation < 1.0


class TestNumericalSolver:
    """Tests for numerical solver."""
    
    def test_rk4_step(self):
        """RK4 should accurately solve simple ODE."""
        solver = NumericalSolver()
        
        # dy/dt = y, y(0) = 1 -> y = e^t
        def derivative(t, y):
            return y
        
        y = np.array([1.0])
        y = solver.step_rk4(derivative, y, 0, 0.1)
        
        # Should be close to e^0.1 ≈ 1.105
        assert abs(y[0] - np.exp(0.1)) < 0.001
    
    def test_lyapunov_detection(self):
        """Should detect chaos vs stability."""
        solver = NumericalSolver()
        
        # Stable trajectory (linear)
        stable = np.linspace(0, 10, 100).reshape(-1, 1)
        lyap_stable = solver.calculate_lyapunov_exponent(stable, 0.1, dimension=1)
        
        # Should have small/zero Lyapunov exponent
        assert abs(lyap_stable) < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
