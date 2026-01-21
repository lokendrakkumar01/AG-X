"""
Molecular Structure and Visualization
=====================================

3D molecular structure handling and visualization tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from loguru import logger


@dataclass
class Atom:
    """Represents an atom in a molecule."""
    element: str
    position: Tuple[float, float, float]
    atomic_number: int
    
    # Atomic radii in Angstroms (covalent radii)
    RADII = {
        'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
        'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Br': 1.20, 'I': 1.39,
    }
    
    # Atomic colors (CPK coloring)
    COLORS = {
        'H': '#FFFFFF', 'C': '#909090', 'N': '#3050F8', 'O': '#FF0D0D',
        'F': '#90E050', 'P': '#FF8000', 'S': '#FFFF30', 'Cl': '#1FF01F',
        'Br': '#A62929', 'I': '#940094',
    }
    
    @property
    def radius(self) -> float:
        """Get covalent radius of atom."""
        return self.RADII.get(self.element, 1.0)
    
    @property
    def color(self) -> str:
        """Get CPK color of atom."""
        return self.COLORS.get(self.element, '#FF1493')


@dataclass
class Bond:
    """Represents a chemical bond."""
    atom1_idx: int
    atom2_idx: int
    bond_order: int = 1  # 1=single, 2=double, 3=triple


class MolecularStructure:
    """Represents a molecular structure with atoms and bonds."""
    
    def __init__(self, name: str = "Molecule"):
        """Initialize molecular structure.
        
        Args:
            name: Name of the molecule
        """
        self.name = name
        self.atoms: List[Atom] = []
        self.bonds: List[Bond] = []
    
    def add_atom(self, element: str, position: Tuple[float, float, float]) -> int:
        """Add an atom to the molecule.
        
        Args:
            element: Element symbol (e.g., 'C', 'H', 'O')
            position: (x, y, z) coordinates in Angstroms
            
        Returns:
            Index of the added atom
        """
        # Get atomic number (simplified)
        atomic_numbers = {
            'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9,
            'P': 15, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53,
        }
        atomic_number = atomic_numbers.get(element, 1)
        
        atom = Atom(
            element=element,
            position=position,
            atomic_number=atomic_number
        )
        self.atoms.append(atom)
        return len(self.atoms) - 1
    
    def add_bond(self, atom1_idx: int, atom2_idx: int, bond_order: int = 1):
        """Add a bond between two atoms.
        
        Args:
            atom1_idx: Index of first atom
            atom2_idx: Index of second atom
            bond_order: Bond order (1, 2, or 3)
        """
        bond = Bond(atom1_idx, atom2_idx, bond_order)
        self.bonds.append(bond)
    
    def get_molecular_formula(self) -> str:
        """Get molecular formula (e.g., 'H2O', 'C6H12O6').
        
        Returns:
            Molecular formula string
        """
        from collections import Counter
        
        element_counts = Counter(atom.element for atom in self.atoms)
        
        # Standard order: C, H, then alphabetical
        formula = ""
        if 'C' in element_counts:
            count = element_counts.pop('C')
            formula += f"C{count if count > 1 else ''}"
        if 'H' in element_counts:
            count = element_counts.pop('H')
            formula += f"H{count if count > 1 else ''}"
        
        for element in sorted(element_counts.keys()):
            count = element_counts[element]
            formula += f"{element}{count if count > 1 else ''}"
        
        return formula
    
    def get_molecular_weight(self) -> float:
        """Calculate molecular weight in g/mol.
        
        Returns:
            Molecular weight
        """
        # Atomic masses (simplified)
        atomic_masses = {
            'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999,
            'F': 18.998, 'P': 30.974, 'S': 32.065, 'Cl': 35.453,
            'Br': 79.904, 'I': 126.904,
        }
        
        total_mass = sum(
            atomic_masses.get(atom.element, 0) for atom in self.atoms
        )
        return total_mass
    
    def get_center_of_mass(self) -> Tuple[float, float, float]:
        """Calculate center of mass.
        
        Returns:
            (x, y, z) coordinates of center of mass
        """
        if not self.atoms:
            return (0.0, 0.0, 0.0)
        
        positions = np.array([atom.position for atom in self.atoms])
        return tuple(np.mean(positions, axis=0))
    
    @classmethod
    def create_water(cls) -> MolecularStructure:
        """Create a water molecule (H2O).
        
        Returns:
            Water molecular structure
        """
        mol = cls("Water")
        
        # Add atoms (tetrahedral geometry)
        o_idx = mol.add_atom('O', (0.0, 0.0, 0.0))
        h1_idx = mol.add_atom('H', (0.96, 0.0, 0.0))
        h2_idx = mol.add_atom('H', (-0.24, 0.93, 0.0))
        
        # Add bonds
        mol.add_bond(o_idx, h1_idx)
        mol.add_bond(o_idx, h2_idx)
        
        return mol
    
    @classmethod
    def create_methane(cls) -> MolecularStructure:
        """Create a methane molecule (CH4).
        
        Returns:
            Methane molecular structure
        """
        mol = cls("Methane")
        
        # Tetrahedral geometry
        c_idx = mol.add_atom('C', (0.0, 0.0, 0.0))
        
        # Hydrogen positions (tetrahedral)
        bond_length = 1.09  # Angstroms
        angle = 109.5 * np.pi / 180  # Tetrahedral angle
        
        h1_idx = mol.add_atom('H', (bond_length, 0.0, 0.0))
        h2_idx = mol.add_atom('H', (-bond_length/3, bond_length*np.sqrt(8/9), 0.0))
        h3_idx = mol.add_atom('H', (-bond_length/3, -bond_length*np.sqrt(2/9), bond_length*np.sqrt(2/3)))
        h4_idx = mol.add_atom('H', (-bond_length/3, -bond_length*np.sqrt(2/9), -bond_length*np.sqrt(2/3)))
        
        # Add bonds
        for h_idx in [h1_idx, h2_idx, h3_idx, h4_idx]:
            mol.add_bond(c_idx, h_idx)
        
        return mol


class MolecularVisualizer:
    """Visualize molecular structures in 3D."""
    
    @staticmethod
    def visualize_3d(molecule: MolecularStructure) -> go.Figure:
        """Create 3D visualization of molecule.
        
        Args:
            molecule: Molecular structure to visualize
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        
        if not molecule.atoms:
            return fig
        
        # Draw atoms
        for atom in molecule.atoms:
            x, y, z = atom.position
            
            fig.add_trace(go.Scatter3d(
                x=[x], y=[y], z=[z],
                mode='markers',
                marker=dict(
                    size=atom.radius * 40,  # Scale for visibility
                    color=atom.color,
                    line=dict(color='black', width=2)
                ),
                text=atom.element,
                hoverinfo='text',
                showlegend=False,
            ))
        
        # Draw bonds
        for bond in molecule.bonds:
            atom1 = molecule.atoms[bond.atom1_idx]
            atom2 = molecule.atoms[bond.atom2_idx]
            
            x1, y1, z1 = atom1.position
            x2, y2, z2 = atom2.position
            
            # Draw bond as line
            fig.add_trace(go.Scatter3d(
                x=[x1, x2, None],
                y=[y1, y2, None],
                z=[z1, z2, None],
                mode='lines',
                line=dict(
                    color='gray',
                    width=bond.bond_order * 5
                ),
                hoverinfo='skip',
                showlegend=False,
            ))
        
        # Update layout
        formula = molecule.get_molecular_formula()
        mw = molecule.get_molecular_weight()
        
        fig.update_layout(
            title=f"{molecule.name} ({formula})<br>MW: {mw:.2f} g/mol",
            scene=dict(
                xaxis_title='X (Å)',
                yaxis_title='Y (Å)',
                zaxis_title='Z (Å)',
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            height=600,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        return fig
