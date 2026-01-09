"""
Newtonian Gravity Simulation Engine
====================================

Classical N-body gravitational physics with energy tracking,
orbital mechanics, and collision detection.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum
import uuid

from agx.physics.constants import PhysicsConstants, Units, CONSTANTS, SI_UNITS


class BodyType(str, Enum):
    """Classification of gravitating bodies."""
    STAR = "star"
    PLANET = "planet"
    MOON = "moon"
    ASTEROID = "asteroid"
    PARTICLE = "particle"
    EXOTIC = "exotic"           # For speculative physics
    TEST_PARTICLE = "test"      # Massless test particle


@dataclass
class Body:
    """
    Represents a gravitating body in the simulation.
    
    Supports both positive and negative mass (for speculative physics).
    Negative mass is clearly marked as THEORETICAL.
    """
    
    # Identification
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    body_type: BodyType = BodyType.PARTICLE
    
    # Physical properties
    mass: float = 1.0               # Can be negative for [THEORETICAL] exotic matter
    radius: float = 0.1
    charge: float = 0.0             # Optional electromagnetic charge
    
    # State vectors (3D)
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Tracking
    is_fixed: bool = False          # Immovable body
    is_active: bool = True          # Part of simulation
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure arrays are numpy arrays."""
        self.position = np.asarray(self.position, dtype=np.float64)
        self.velocity = np.asarray(self.velocity, dtype=np.float64)
        self.acceleration = np.asarray(self.acceleration, dtype=np.float64)
        
        if self.mass < 0:
            self.metadata["theoretical"] = True
            self.metadata["warning"] = "[THEORETICAL] Negative mass entity"
    
    @property
    def momentum(self) -> np.ndarray:
        """Linear momentum p = mv."""
        return self.mass * self.velocity
    
    @property
    def kinetic_energy(self) -> float:
        """Kinetic energy KE = 0.5 * m * v²."""
        return 0.5 * self.mass * np.dot(self.velocity, self.velocity)
    
    @property
    def speed(self) -> float:
        """Magnitude of velocity."""
        return np.linalg.norm(self.velocity)
    
    def distance_to(self, other: "Body") -> float:
        """Calculate distance to another body."""
        return np.linalg.norm(self.position - other.position)
    
    def direction_to(self, other: "Body") -> np.ndarray:
        """Unit vector pointing toward another body."""
        diff = other.position - self.position
        dist = np.linalg.norm(diff)
        if dist < 1e-15:
            return np.zeros(3)
        return diff / dist
    
    def copy(self) -> "Body":
        """Create a deep copy of this body."""
        return Body(
            id=self.id,
            name=self.name,
            body_type=self.body_type,
            mass=self.mass,
            radius=self.radius,
            charge=self.charge,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            acceleration=self.acceleration.copy(),
            is_fixed=self.is_fixed,
            is_active=self.is_active,
            metadata=self.metadata.copy(),
        )


@dataclass
class GravitationalForce:
    """
    Represents a gravitational force calculation result.
    """
    source_id: str
    target_id: str
    force_vector: np.ndarray
    magnitude: float
    distance: float
    potential_energy: float


class NewtonianEngine:
    """
    Classical Newtonian N-body gravity simulation engine.
    
    Features:
    - Direct N-body calculation with O(N²) complexity
    - Softening parameter to prevent singularities
    - Energy conservation tracking
    - Support for negative mass (theoretical)
    - Collision detection
    
    For large N, consider using Barnes-Hut tree or FMM methods.
    """
    
    def __init__(
        self,
        units: Units = SI_UNITS,
        softening: float = 1e-6,
        enable_collisions: bool = True,
        collision_merge: bool = False,
    ):
        """
        Initialize the Newtonian engine.
        
        Args:
            units: Unit system for the simulation
            softening: Softening length to prevent numerical singularities
            enable_collisions: Whether to detect collisions
            collision_merge: Whether to merge colliding bodies
        """
        self.units = units
        self.softening = softening
        self.enable_collisions = enable_collisions
        self.collision_merge = collision_merge
        
        self.G = units.G_sim  # Gravitational constant in simulation units
        self.bodies: List[Body] = []
        
        # Energy tracking
        self._initial_energy: Optional[float] = None
        self._collision_events: List[Tuple[str, str, float]] = []
    
    def add_body(self, body: Body) -> None:
        """Add a body to the simulation."""
        self.bodies.append(body)
    
    def add_bodies(self, bodies: List[Body]) -> None:
        """Add multiple bodies to the simulation."""
        self.bodies.extend(bodies)
    
    def remove_body(self, body_id: str) -> bool:
        """Remove a body by ID. Returns True if found and removed."""
        for i, body in enumerate(self.bodies):
            if body.id == body_id:
                self.bodies.pop(i)
                return True
        return False
    
    def get_body(self, body_id: str) -> Optional[Body]:
        """Get a body by ID."""
        for body in self.bodies:
            if body.id == body_id:
                return body
        return None
    
    def calculate_force(self, body1: Body, body2: Body) -> GravitationalForce:
        """
        Calculate gravitational force between two bodies.
        
        F = -G * m1 * m2 / r² * r̂
        
        Note: With negative mass, this can produce repulsive forces (THEORETICAL).
        """
        r_vec = body2.position - body1.position
        r_sq = np.dot(r_vec, r_vec) + self.softening**2
        r = np.sqrt(r_sq)
        
        # Force magnitude (negative for attraction with positive masses)
        F_mag = self.G * body1.mass * body2.mass / r_sq
        
        # Force vector on body1 due to body2
        r_hat = r_vec / r if r > 1e-15 else np.zeros(3)
        F_vec = F_mag * r_hat
        
        # Gravitational potential energy
        U = -self.G * body1.mass * body2.mass / r
        
        return GravitationalForce(
            source_id=body2.id,
            target_id=body1.id,
            force_vector=F_vec,
            magnitude=abs(F_mag),
            distance=r,
            potential_energy=U,
        )
    
    def calculate_all_forces(self) -> Dict[str, np.ndarray]:
        """
        Calculate net gravitational force on each body.
        
        Returns:
            Dictionary mapping body ID to net force vector
        """
        forces = {body.id: np.zeros(3) for body in self.bodies if body.is_active}
        
        active_bodies = [b for b in self.bodies if b.is_active and not b.is_fixed]
        
        for i, body1 in enumerate(active_bodies):
            for body2 in self.bodies[i+1:]:
                if not body2.is_active:
                    continue
                
                gf = self.calculate_force(body1, body2)
                
                if not body1.is_fixed:
                    forces[body1.id] += gf.force_vector
                if not body2.is_fixed:
                    forces[body2.id] -= gf.force_vector  # Newton's 3rd law
        
        return forces
    
    def calculate_accelerations(self) -> Dict[str, np.ndarray]:
        """
        Calculate accelerations for all bodies.
        
        a = F / m
        """
        forces = self.calculate_all_forces()
        accelerations = {}
        
        for body in self.bodies:
            if body.is_active and not body.is_fixed and abs(body.mass) > 1e-30:
                accelerations[body.id] = forces[body.id] / body.mass
            else:
                accelerations[body.id] = np.zeros(3)
        
        return accelerations
    
    def get_state_vectors(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get flattened position and velocity state vectors.
        
        Returns:
            (positions, velocities) as flattened arrays
        """
        positions = np.array([b.position for b in self.bodies if b.is_active])
        velocities = np.array([b.velocity for b in self.bodies if b.is_active])
        return positions.flatten(), velocities.flatten()
    
    def set_state_vectors(self, positions: np.ndarray, velocities: np.ndarray) -> None:
        """Set state from flattened vectors."""
        active_bodies = [b for b in self.bodies if b.is_active]
        n = len(active_bodies)
        
        pos_reshaped = positions.reshape(n, 3)
        vel_reshaped = velocities.reshape(n, 3)
        
        for i, body in enumerate(active_bodies):
            if not body.is_fixed:
                body.position = pos_reshaped[i].copy()
                body.velocity = vel_reshaped[i].copy()
    
    def kinetic_energy(self) -> float:
        """Total kinetic energy of the system."""
        return sum(b.kinetic_energy for b in self.bodies if b.is_active)
    
    def potential_energy(self) -> float:
        """Total gravitational potential energy."""
        U = 0.0
        active = [b for b in self.bodies if b.is_active]
        
        for i, b1 in enumerate(active):
            for b2 in active[i+1:]:
                r = b1.distance_to(b2) + self.softening
                U -= self.G * b1.mass * b2.mass / r
        
        return U
    
    def total_energy(self) -> float:
        """Total mechanical energy (KE + PE)."""
        return self.kinetic_energy() + self.potential_energy()
    
    def total_momentum(self) -> np.ndarray:
        """Total linear momentum of the system."""
        return sum((b.momentum for b in self.bodies if b.is_active), np.zeros(3))
    
    def angular_momentum(self) -> np.ndarray:
        """Total angular momentum about origin."""
        L = np.zeros(3)
        for b in self.bodies:
            if b.is_active:
                L += np.cross(b.position, b.momentum)
        return L
    
    def center_of_mass(self) -> np.ndarray:
        """Calculate center of mass position."""
        total_mass = sum(b.mass for b in self.bodies if b.is_active)
        if abs(total_mass) < 1e-30:
            return np.zeros(3)
        
        com = np.zeros(3)
        for b in self.bodies:
            if b.is_active:
                com += b.mass * b.position
        return com / total_mass
    
    def check_collisions(self) -> List[Tuple[Body, Body]]:
        """Detect colliding pairs based on radii overlap."""
        if not self.enable_collisions:
            return []
        
        collisions = []
        active = [b for b in self.bodies if b.is_active]
        
        for i, b1 in enumerate(active):
            for b2 in active[i+1:]:
                dist = b1.distance_to(b2)
                if dist < (b1.radius + b2.radius):
                    collisions.append((b1, b2))
        
        return collisions
    
    def energy_error(self) -> float:
        """Calculate relative energy error from initial state."""
        if self._initial_energy is None:
            self._initial_energy = self.total_energy()
            return 0.0
        
        current = self.total_energy()
        if abs(self._initial_energy) < 1e-30:
            return abs(current - self._initial_energy)
        return abs((current - self._initial_energy) / self._initial_energy)
    
    def step_euler(self, dt: float) -> None:
        """Simple Euler integration step (for testing, not recommended)."""
        accelerations = self.calculate_accelerations()
        
        for body in self.bodies:
            if body.is_active and not body.is_fixed:
                body.velocity += accelerations[body.id] * dt
                body.position += body.velocity * dt
                body.acceleration = accelerations[body.id]
    
    def step_leapfrog(self, dt: float) -> None:
        """
        Leapfrog (Verlet) integration step.
        
        Symplectic integrator that conserves energy better than Euler.
        """
        # Kick: v(n+1/2) = v(n) + a(n) * dt/2
        accelerations = self.calculate_accelerations()
        for body in self.bodies:
            if body.is_active and not body.is_fixed:
                body.velocity += 0.5 * accelerations[body.id] * dt
        
        # Drift: x(n+1) = x(n) + v(n+1/2) * dt
        for body in self.bodies:
            if body.is_active and not body.is_fixed:
                body.position += body.velocity * dt
        
        # Kick: v(n+1) = v(n+1/2) + a(n+1) * dt/2
        accelerations = self.calculate_accelerations()
        for body in self.bodies:
            if body.is_active and not body.is_fixed:
                body.velocity += 0.5 * accelerations[body.id] * dt
                body.acceleration = accelerations[body.id]
    
    def create_two_body_system(
        self,
        m1: float = 1.0,
        m2: float = 1.0,
        separation: float = 1.0,
        eccentricity: float = 0.0,
    ) -> List[Body]:
        """
        Create a two-body system (e.g., binary star, planet-moon).
        
        Bodies are placed in the x-y plane with circular or elliptical orbits.
        """
        # Reduced mass and total mass
        M = m1 + m2
        mu = m1 * m2 / M
        
        # Positions relative to center of mass
        r1 = separation * m2 / M
        r2 = separation * m1 / M
        
        # Orbital velocity for circular orbit
        v_orb = np.sqrt(self.G * M / separation)
        
        # Adjust for eccentricity (velocity at perihelion)
        if eccentricity > 0:
            v_orb *= np.sqrt((1 + eccentricity) / (1 - eccentricity))
        
        body1 = Body(
            name="Body 1",
            mass=m1,
            position=np.array([-r1, 0, 0]),
            velocity=np.array([0, -v_orb * m2/M, 0]),
        )
        
        body2 = Body(
            name="Body 2", 
            mass=m2,
            position=np.array([r2, 0, 0]),
            velocity=np.array([0, v_orb * m1/M, 0]),
        )
        
        self.add_bodies([body1, body2])
        return [body1, body2]
    
    def create_solar_system_simple(self) -> List[Body]:
        """Create a simplified solar system (Sun + 4 inner planets)."""
        # Using astronomical units
        sun = Body(
            name="Sun",
            mass=1.0,  # Solar mass = 1 in these units
            body_type=BodyType.STAR,
            position=np.zeros(3),
            velocity=np.zeros(3),
            is_fixed=False,
        )
        
        planets_data = [
            ("Mercury", 1.66e-7, 0.387, 0.205),
            ("Venus", 2.45e-6, 0.723, 0.007),
            ("Earth", 3.00e-6, 1.0, 0.017),
            ("Mars", 3.23e-7, 1.524, 0.093),
        ]
        
        bodies = [sun]
        
        for name, mass, a, e in planets_data:
            r = a * (1 - e)  # Perihelion distance
            v = np.sqrt(self.G * (1 + mass) * (2/r - 1/a))
            
            planet = Body(
                name=name,
                mass=mass,
                body_type=BodyType.PLANET,
                position=np.array([r, 0, 0]),
                velocity=np.array([0, v, 0]),
            )
            bodies.append(planet)
        
        self.add_bodies(bodies)
        return bodies
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get complete state as dictionary for serialization."""
        return {
            "bodies": [
                {
                    "id": b.id,
                    "name": b.name,
                    "mass": b.mass,
                    "position": b.position.tolist(),
                    "velocity": b.velocity.tolist(),
                    "radius": b.radius,
                }
                for b in self.bodies
            ],
            "energy": {
                "kinetic": self.kinetic_energy(),
                "potential": self.potential_energy(),
                "total": self.total_energy(),
            },
            "momentum": self.total_momentum().tolist(),
            "angular_momentum": self.angular_momentum().tolist(),
            "center_of_mass": self.center_of_mass().tolist(),
        }
