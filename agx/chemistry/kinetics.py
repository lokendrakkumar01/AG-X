"""
Chemical Kinetics Calculator
============================

Calculate reaction rates, rate laws, and kinetic parameters.
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np
from scipy.optimize import curve_fit
from loguru import logger


class KineticsCalculator:
    """Calculate chemical kinetics parameters."""
    
    def __init__(self):
        """Initialize kinetics calculator."""
        self.R = 8.314  # Gas constant J/(mol·K)
    
    def calculate_rate_constant(
        self,
        A: float,
        Ea: float,
        temperature: float
    ) -> float:
        """Calculate rate constant using Arrhenius equation.
        
        k = A × exp(-Ea / RT)
        
        Args:
            A: Pre-exponential factor (frequency factor)
            Ea: Activation energy in kJ/mol
            temperature: Temperature in Kelvin
            
        Returns:
            Rate constant k
        """
        # Convert kJ to J
        Ea_j = Ea * 1000
        
        k = A * np.exp(-Ea_j / (self.R * temperature))
        
        return k
    
    def calculate_half_life(
        self,
        k: float,
        order: int,
        initial_concentration: float = 1.0
    ) -> float:
        """Calculate half-life for different reaction orders.
        
        Args:
            k: Rate constant
            order: Reaction order (0, 1, or 2)
            initial_concentration: Initial concentration (for 0th and 2nd order)
            
        Returns:
            Half-life time
        """
        if order == 0:
            # t_1/2 = [A]0 / (2k)
            return initial_concentration / (2 * k)
        
        elif order == 1:
            # t_1/2 = ln(2) / k
            return np.log(2) / k
        
        elif order == 2:
            # t_1/2 = 1 / (k[A]0)
            return 1 / (k * initial_concentration)
        
        else:
            raise ValueError(f"Unsupported reaction order: {order}")
    
    def fit_rate_law(
        self,
        time_data: List[float],
        concentration_data: List[float]
    ) -> Tuple[int, float, float]:
        """Fit concentration vs time data to determine reaction order and rate constant.
        
        Args:
            time_data: List of time points
            concentration_data: List of concentrations at each time point
            
        Returns:
            Tuple of (order, rate_constant, r_squared)
        """
        t = np.array(time_data)
        C = np.array(concentration_data)
        
        # Try different orders and find best fit
        best_order = 0
        best_k = 0.0
        best_r2 = -1.0
        
        # Zero order: C = C0 - kt
        try:
            def zero_order(t, C0, k):
                return C0 - k * t
            
            popt, _ = curve_fit(zero_order, t, C, p0=[C[0], 0.01])
            C_pred = zero_order(t, *popt)
            r2 = 1 - np.sum((C - C_pred)**2) / np.sum((C - np.mean(C))**2)
            
            if r2 > best_r2:
                best_order, best_k, best_r2 = 0, popt[1], r2
        except:
            pass
        
        # First order: C = C0 × exp(-kt)
        try:
            def first_order(t, C0, k):
                return C0 * np.exp(-k * t)
            
            popt, _ = curve_fit(first_order, t, C, p0=[C[0], 0.01])
            C_pred = first_order(t, *popt)
            r2 = 1 - np.sum((C - C_pred)**2) / np.sum((C - np.mean(C))**2)
            
            if r2 > best_r2:
                best_order, best_k, best_r2 = 1, popt[1], r2
        except:
            pass
        
        # Second order: 1/C = 1/C0 + kt
        try:
            def second_order_inv(t, C0_inv, k):
                return C0_inv + k * t
            
            C_inv = 1 / C
            popt, _ = curve_fit(second_order_inv, t, C_inv, p0=[1/C[0], 0.01])
            C_inv_pred = second_order_inv(t, *popt)
            r2 = 1 - np.sum((C_inv - C_inv_pred)**2) / np.sum((C_inv - np.mean(C_inv))**2)
            
            if r2 > best_r2:
                best_order, best_k, best_r2 = 2, popt[1], r2
        except:
            pass
        
        return best_order, best_k, best_r2
    
    def integrated_rate_law(
        self,
        order: int,
        k: float,
        C0: float,
        time: float
    ) -> float:
        """Calculate concentration at time t using integrated rate law.
        
        Args:
            order: Reaction order (0, 1, or 2)
            k: Rate constant
            C0: Initial concentration
            time: Time
            
        Returns:
            Concentration at time t
        """
        if order == 0:
            # C = C0 - kt
            return max(0, C0 - k * time)
        
        elif order == 1:
            # C = C0 × exp(-kt)
            return C0 * np.exp(-k * time)
        
        elif order == 2:
            # 1/C = 1/C0 + kt
            return 1 / (1/C0 + k * time)
        
        else:
            raise ValueError(f"Unsupported reaction order: {order}")
    
    def arrhenius_parameters(
        self,
        temperatures: List[float],
        rate_constants: List[float]
    ) -> Tuple[float, float]:
        """Determine Arrhenius parameters (A and Ea) from temperature-dependent rate data.
        
        Using: ln(k) = ln(A) - Ea/(R×T)
        
        Args:
            temperatures: List of temperatures in Kelvin
            rate_constants: List of rate constants at each temperature
            
        Returns:
            Tuple of (activation_energy_kJ/mol, pre_exponential_factor)
        """
        T = np.array(temperatures)
        k = np.array(rate_constants)
        
        # Linear fit of ln(k) vs 1/T
        inv_T = 1 / T
        ln_k = np.log(k)
        
        # Slope = -Ea/R, Intercept = ln(A)
        coeffs = np.polyfit(inv_T, ln_k, 1)
        slope, intercept = coeffs
        
        Ea_j = -slope * self.R  # Activation energy in J/mol
        Ea_kj = Ea_j / 1000  # Convert to kJ/mol
        A = np.exp(intercept)  # Pre-exponential factor
        
        return Ea_kj, A
