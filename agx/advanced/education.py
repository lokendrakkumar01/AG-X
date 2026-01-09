"""
Education Mode
===============

Step-by-step explanations and learning modules for students and researchers.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class LearningStep:
    """A single step in a learning module."""
    title: str
    explanation: str
    equation: str = ""
    code_example: str = ""
    visualization_hint: str = ""


@dataclass
class LearningModule:
    """A complete learning module on a physics topic."""
    name: str
    description: str
    difficulty: str  # beginner, intermediate, advanced
    steps: List[LearningStep] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)


class EducationMode:
    """
    Educational Mode for AG-X 2026.
    
    Provides step-by-step explanations of simulations,
    physics concepts, and guided tutorials.
    """
    
    MODULES: Dict[str, LearningModule] = {}
    
    def __init__(self):
        self._init_modules()
        self.current_module: Optional[LearningModule] = None
        self.current_step: int = 0
        self.explanation_level: str = "intermediate"  # beginner, intermediate, advanced
    
    def _init_modules(self):
        """Initialize built-in learning modules."""
        
        # Newton's Gravity Module
        self.MODULES["newtonian_gravity"] = LearningModule(
            name="Newtonian Gravity",
            description="Learn the fundamentals of classical gravitational physics",
            difficulty="beginner",
            steps=[
                LearningStep(
                    title="Newton's Law of Universal Gravitation",
                    explanation="""
Every particle in the universe attracts every other particle with a force 
proportional to the product of their masses and inversely proportional to 
the square of the distance between them.

This elegant law explains why apples fall and why planets orbit!
""",
                    equation="F = G × M × m / r²",
                    code_example="""
# Calculate gravitational force
G = 6.674e-11  # Gravitational constant
M = 5.97e24    # Earth's mass (kg)
m = 70         # Your mass (kg)
r = 6.371e6    # Earth's radius (m)

F = G * M * m / r**2
print(f"Your weight: {F:.1f} N")
""",
                    visualization_hint="particles",
                ),
                LearningStep(
                    title="Gravitational Potential Energy",
                    explanation="""
Objects in a gravitational field have potential energy. This energy is 
released as kinetic energy when objects fall toward each other.

The formula shows that potential energy is negative - this means you need 
to add energy to separate objects!
""",
                    equation="U = -G × M × m / r",
                ),
                LearningStep(
                    title="Orbital Mechanics",
                    explanation="""
When an object moves fast enough perpendicular to gravity, it falls 
around the massive body instead of into it - this is an orbit!

The orbital velocity at distance r is determined by balancing 
gravitational and centripetal forces.
""",
                    equation="v_orbital = √(G × M / r)",
                ),
            ],
        )
        
        # General Relativity Module
        self.MODULES["general_relativity"] = LearningModule(
            name="General Relativity Basics",
            description="Introduction to Einstein's theory of gravity",
            difficulty="advanced",
            prerequisites=["newtonian_gravity"],
            steps=[
                LearningStep(
                    title="Space-Time Curvature",
                    explanation="""
Einstein's revolutionary insight: gravity isn't a force at all! 
Instead, massive objects curve the fabric of space and time itself.

Objects move along the straightest possible paths (geodesics) 
in this curved spacetime, which appears to us as gravitational attraction.
""",
                    visualization_hint="spacetime",
                ),
                LearningStep(
                    title="Time Dilation",
                    explanation="""
Time runs slower in stronger gravitational fields! This effect has been 
measured with atomic clocks on airplanes and satellites.

GPS satellites must account for this effect - without it, GPS would 
accumulate errors of about 10 km per day!
""",
                    equation="τ = t × √(1 - 2GM/(rc²))",
                ),
                LearningStep(
                    title="Black Holes",
                    explanation="""
When mass is compressed inside its Schwarzschild radius, spacetime curves 
so severely that nothing - not even light - can escape.

The Schwarzschild radius for any object is:
""",
                    equation="r_s = 2GM/c²",
                ),
            ],
        )
        
        # Speculative Physics Module
        self.MODULES["speculative_physics"] = LearningModule(
            name="Speculative Anti-Gravity Concepts",
            description="[THEORETICAL] Explore hypothetical physics concepts",
            difficulty="advanced",
            prerequisites=["newtonian_gravity", "general_relativity"],
            steps=[
                LearningStep(
                    title="⚠️ Disclaimer: Theoretical Exploration",
                    explanation="""
IMPORTANT: The concepts in this module are PURELY HYPOTHETICAL and do not 
represent real physics or technology. They are explored here for:
- Creative thinking about physics
- Understanding why certain ideas violate known physics
- Inspiration for science fiction
- Computational modeling of "what if" scenarios

No claims of real anti-gravity creation are made.
""",
                ),
                LearningStep(
                    title="Negative Mass (Hypothetical)",
                    explanation="""
[THEORETICAL] What if mass could be negative? This would lead to bizarre 
behavior: negative mass would move toward pushing forces!

A positive and negative mass pair would create "runaway" motion - 
they accelerate forever without external energy input.

⚠️ Negative mass has never been observed and likely violates 
fundamental physics principles.
""",
                ),
            ],
        )
    
    def list_modules(self) -> List[Dict[str, str]]:
        """List available learning modules."""
        return [
            {
                "name": mod.name,
                "description": mod.description,
                "difficulty": mod.difficulty,
                "steps": len(mod.steps),
            }
            for mod in self.MODULES.values()
        ]
    
    def start_module(self, module_key: str) -> Optional[LearningStep]:
        """Start a learning module."""
        if module_key not in self.MODULES:
            return None
        
        self.current_module = self.MODULES[module_key]
        self.current_step = 0
        
        return self.current_module.steps[0] if self.current_module.steps else None
    
    def next_step(self) -> Optional[LearningStep]:
        """Go to next step in current module."""
        if not self.current_module:
            return None
        
        self.current_step += 1
        
        if self.current_step >= len(self.current_module.steps):
            return None  # Module complete
        
        return self.current_module.steps[self.current_step]
    
    def previous_step(self) -> Optional[LearningStep]:
        """Go to previous step."""
        if not self.current_module or self.current_step <= 0:
            return None
        
        self.current_step -= 1
        return self.current_module.steps[self.current_step]
    
    def explain_simulation_step(self, step_data: Dict[str, Any]) -> str:
        """Explain what happened in a simulation step."""
        lines = ["## Simulation Step Explanation\n"]
        
        if "forces" in step_data:
            lines.append("### Forces Calculated")
            lines.append("The gravitational forces between all bodies were computed using F = GMm/r²")
        
        if "accelerations" in step_data:
            lines.append("\n### Accelerations Applied")
            lines.append("Each body's acceleration was calculated using Newton's second law: a = F/m")
        
        if "positions" in step_data:
            lines.append("\n### Positions Updated")
            lines.append("Bodies moved according to their velocities and accelerations")
        
        if "energy" in step_data:
            energy = step_data["energy"]
            lines.append(f"\n### Energy Check")
            lines.append(f"Total system energy: {energy:.6f}")
            lines.append("In a closed system, total energy should be conserved!")
        
        return "\n".join(lines)
    
    def get_concept_explanation(self, concept: str) -> str:
        """Get explanation for a physics concept."""
        explanations = {
            "gravitational_constant": """
**Gravitational Constant (G)**

G = 6.67430 × 10⁻¹¹ m³/(kg·s²)

This fundamental constant determines the strength of gravity in our universe.
It was first measured by Henry Cavendish in 1798 using a torsion balance.

Fun fact: G is one of the least precisely known fundamental constants!
""",
            "energy_conservation": """
**Conservation of Energy**

In a closed system with no external forces, the total mechanical energy 
(kinetic + potential) remains constant.

E_total = KE + PE = constant

If energy appears to not be conserved in a simulation, it usually means:
1. The numerical integrator has errors
2. There are external forces we're not accounting for
3. Energy is being converted to other forms (heat, etc.)
""",
            "chaos": """
**Chaos in Gravitational Systems**

Some gravitational systems are "chaotic" - tiny changes in initial conditions 
lead to dramatically different outcomes over time.

The famous three-body problem is a prime example of chaos. Unlike the 
two-body problem, there's no general analytical solution!

We detect chaos by measuring the "Lyapunov exponent" - if it's positive, 
the system is chaotic.
""",
        }
        
        return explanations.get(concept.lower(), f"No explanation found for '{concept}'")
