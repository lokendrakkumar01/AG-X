"""
Chemical Thermodynamics Calculator
==================================

Calculate thermodynamic properties of chemical reactions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from loguru import logger


@dataclass
class ThermodynamicData:
    """Standard thermodynamic data for a substance."""
    substance: str
    delta_h_formation: float  # kJ/mol
    delta_g_formation: float  # kJ/mol
    entropy: float  # J/(mol·K)
    heat_capacity: Optional[float] = None  # J/(mol·K)


class ThermodynamicsCalculator:
    """Calculate thermodynamic properties of reactions."""
    
    # Standard thermodynamic data at 298.15 K
    STD_DATA = {
        'H2O(l)': ThermodynamicData('H2O(l)', -285.8, -237.1, 70.0, 75.3),
        'H2O(g)': ThermodynamicData('H2O(g)', -241.8, -228.6, 188.8, 33.6),
        'CO2(g)': ThermodynamicData('CO2(g)', -393.5, -394.4, 213.8, 37.1),
        'O2(g)': ThermodynamicData('O2(g)', 0.0, 0.0, 205.2, 29.4),
        'H2(g)': ThermodynamicData('H2(g)', 0.0, 0.0, 130.7, 28.8),
        'N2(g)': ThermodynamicData('N2(g)', 0.0, 0.0, 191.6, 29.1),
        'CH4(g)': ThermodynamicData('CH4(g)', -74.6, -50.5, 186.3, 35.3),
        'NH3(g)': ThermodynamicData('NH3(g)', -45.9, -16.4, 192.8, 35.1),
        'HCl(g)': ThermodynamicData('HCl(g)', -92.3, -95.3, 186.9, 29.1),
        'NaCl(s)': ThermodynamicData('NaCl(s)', -411.2, -384.1, 72.1, 50.5),
    }
    
    def __init__(self):
        """Initialize thermodynamics calculator."""
        self.R = 8.314  # Gas constant J/(mol·K)
        self.std_temp = 298.15  # Standard temperature K
        self.std_pressure = 1.0  # Standard pressure bar
    
    def calculate_delta_h(
        self,
        products: Dict[str, float],
        reactants: Dict[str, float]
    ) -> float:
        """Calculate enthalpy change of reaction (ΔH°).
        
        Args:
            products: Dict of {substance: stoichiometric coefficient}
            reactants: Dict of {substance: stoichiometric coefficient}
            
        Returns:
            ΔH° in kJ/mol
        """
        delta_h = 0.0
        
        # Sum products
        for substance, coef in products.items():
            if substance in self.STD_DATA:
                delta_h += coef * self.STD_DATA[substance].delta_h_formation
        
        # Subtract reactants
        for substance, coef in reactants.items():
            if substance in self.STD_DATA:
                delta_h -= coef * self.STD_DATA[substance].delta_h_formation
        
        return delta_h
    
    def calculate_delta_g(
        self,
        products: Dict[str, float],
        reactants: Dict[str, float]
    ) -> float:
        """Calculate Gibbs free energy change (ΔG°).
        
        Args:
            products: Dict of {substance: stoichiometric coefficient}
            reactants: Dict of {substance: stoichiometric coefficient}
            
        Returns:
            ΔG° in kJ/mol
        """
        delta_g = 0.0
        
        # Sum products
        for substance, coef in products.items():
            if substance in self.STD_DATA:
                delta_g += coef * self.STD_DATA[substance].delta_g_formation
        
        # Subtract reactants
        for substance, coef in reactants.items():
            if substance in self.STD_DATA:
                delta_g -= coef * self.STD_DATA[substance].delta_g_formation
        
        return delta_g
    
    def calculate_delta_s(
        self,
        products: Dict[str, float],
        reactants: Dict[str, float]
    ) -> float:
        """Calculate entropy change (ΔS°).
        
        Args:
            products: Dict of {substance: stoichiometric coefficient}
            reactants: Dict of {substance: stoichiometric coefficient}
            
        Returns:
            ΔS° in J/(mol·K)
        """
        delta_s = 0.0
        
        # Sum products
        for substance, coef in products.items():
            if substance in self.STD_DATA:
                delta_s += coef * self.STD_DATA[substance].entropy
        
        # Subtract reactants
        for substance, coef in reactants.items():
            if substance in self.STD_DATA:
                delta_s -= coef * self.STD_DATA[substance].entropy
        
        return delta_s
    
    def calculate_equilibrium_constant(
        self,
        delta_g: float,
        temperature: float = 298.15
    ) -> float:
        """Calculate equilibrium constant from ΔG°.
        
        Using: ΔG° = -RT ln(K)
        
        Args:
            delta_g: Gibbs free energy change in kJ/mol
            temperature: Temperature in Kelvin
            
        Returns:
            Equilibrium constant K
        """
        # Convert kJ to J
        delta_g_j = delta_g * 1000
        
        # K = exp(-ΔG / RT)
        K = np.exp(-delta_g_j / (self.R * temperature))
        
        return K
    
    def predict_spontaneity(
        self,
        delta_g: float
    ) -> str:
        """Predict if reaction is spontaneous based on ΔG°.
        
        Args:
            delta_g: Gibbs free energy change in kJ/mol
            
        Returns:
            Spontaneity description
        """
        if delta_g < -10:
            return "Highly spontaneous (favorable)"
        elif delta_g < 0:
            return "Spontaneous (favorable)"
        elif abs(delta_g) < 1:
            return "Near equilibrium"
        elif delta_g < 10:
            return "Non-spontaneous (unfavorable)"
        else:
            return "Highly non-spontaneous (unfavorable)"
    
    def vant_hoff_equation(
        self,
        delta_h: float,
        T1: float,
        T2: float,
        K1: float
    ) -> float:
        """Calculate equilibrium constant at different temperature using van't Hoff equation.
        
        ln(K2/K1) = -ΔH°/R * (1/T2 - 1/T1)
        
        Args:
            delta_h: Enthalpy change in kJ/mol
            T1: Initial temperature in K
            T2: Final temperature in K
            K1: Equilibrium constant at T1
            
        Returns:
            Equilibrium constant K2 at T2
        """
        # Convert kJ to J
        delta_h_j = delta_h * 1000
        
        # Calculate K2
        ln_ratio = -(delta_h_j / self.R) * (1/T2 - 1/T1)
        K2 = K1 * np.exp(ln_ratio)
        
        return K2
