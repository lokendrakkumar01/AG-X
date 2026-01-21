"""
Chemical Equation Balancer
==========================

Balance chemical equations using matrix algebra and linear programming.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.linalg import null_space
from loguru import logger


@dataclass
class ChemicalEquation:
    """Represents a chemical equation."""
    reactants: List[str]
    products: List[str]
    coefficients_reactants: List[int]
    coefficients_products: List[int]
    
    def __str__(self) -> str:
        """String representation of balanced equation."""
        reactant_str = " + ".join(
            f"{coef if coef > 1 else ''}{mol}".strip()
            for coef, mol in zip(self.coefficients_reactants, self.reactants)
        )
        product_str = " + ".join(
            f"{coef if coef > 1 else ''}{mol}".strip()
            for coef, mol in zip(self.coefficients_products, self.products)
        )
        return f"{reactant_str} → {product_str}"
    
    @property
    def is_balanced(self) -> bool:
        """Check if equation is balanced."""
        balancer = EquationBalancer()
        element_count_left = balancer._count_elements_side(
            self.reactants, self.coefficients_reactants
        )
        element_count_right = balancer._count_elements_side(
            self.products, self.coefficients_products
        )
        return element_count_left == element_count_right


class EquationBalancer:
    """Balance chemical equations using matrix methods."""
    
    def __init__(self):
        """Initialize equation balancer."""
        self.element_pattern = re.compile(r'([A-Z][a-z]?)(\d*)')
        self.parentheses_pattern = re.compile(r'\(([^)]+)\)(\d*)')
    
    def parse_formula(self, formula: str) -> Dict[str, int]:
        """Parse a chemical formula and return element counts.
        
        Args:
            formula: Chemical formula (e.g., "H2O", "Ca(OH)2")
            
        Returns:
            Dictionary mapping elements to their counts
        """
        # Clean the formula
        formula = formula.strip()
        
        # Handle parentheses first
        while '(' in formula:
            match = self.parentheses_pattern.search(formula)
            if not match:
                break
            
            group = match.group(1)
            multiplier = int(match.group(2)) if match.group(2) else 1
            
            # Parse the group
            group_elements = self.element_pattern.findall(group)
            replacement = ""
            for element, count in group_elements:
                count = int(count) if count else 1
                total = count * multiplier
                replacement += f"{element}{total if total > 1 else ''}"
            
            formula = formula[:match.start()] + replacement + formula[match.end():]
        
        # Parse remaining elements
        elements = defaultdict(int)
        for element, count in self.element_pattern.findall(formula):
            count = int(count) if count else 1
            elements[element] += count
        
        return dict(elements)
    
    def balance_equation(
        self, 
        reactants: List[str], 
        products: List[str]
    ) -> Tuple[List[int], List[int]]:
        """Balance a chemical equation.
        
        Args:
            reactants: List of reactant formulas
            products: List of product formulas
            
        Returns:
            Tuple of (reactant coefficients, product coefficients)
        """
        # Parse all molecules
        reactant_elements = [self.parse_formula(r) for r in reactants]
        product_elements = [self.parse_formula(p) for p in products]
        
        # Get all unique elements
        all_elements = set()
        for elem_dict in reactant_elements + product_elements:
            all_elements.update(elem_dict.keys())
        elements = sorted(all_elements)
        
        # Build matrix A where A*x = 0
        # Rows are elements, columns are molecules
        n_molecules = len(reactants) + len(products)
        A = np.zeros((len(elements), n_molecules))
        
        # Fill reactants (positive)
        for col, elem_dict in enumerate(reactant_elements):
            for row, element in enumerate(elements):
                A[row, col] = elem_dict.get(element, 0)
        
        # Fill products (negative)
        for col, elem_dict in enumerate(product_elements):
            for row, element in enumerate(elements):
                A[row, len(reactants) + col] = -elem_dict.get(element, 0)
        
        # Find null space to get coefficients
        try:
            ns = null_space(A)
            if ns.size == 0:
                logger.warning("Could not balance equation automatically")
                return ([1] * len(reactants), [1] * len(products))
            
            # Get the first solution
            solution = ns[:, 0]
            
            # Make all coefficients positive
            if np.any(solution < 0):
                solution = -solution
            
            # Convert to integers
            # Find LCM to make all coefficients integers
            coeffs = self._to_integers(solution)
            
            reactant_coeffs = coeffs[:len(reactants)].tolist()
            product_coeffs = coeffs[len(reactants):].tolist()
            
            return reactant_coeffs, product_coeffs
        
        except Exception as e:
            logger.error(f"Error balancing equation: {e}")
            return ([1] * len(reactants), [1] * len(products))
    
    def _to_integers(self, arr: np.ndarray, max_denominator: int = 100) -> np.ndarray:
        """Convert float array to integers using fractions.
        
        Args:
            arr: Float array
            max_denominator: Maximum denominator to try
            
        Returns:
            Integer array
        """
        from fractions import Fraction
        from math import gcd
        from functools import reduce
        
        # Convert to fractions
        fractions = [Fraction(x).limit_denominator(max_denominator) for x in arr]
        
        # Find LCM of denominators
        denominators = [f.denominator for f in fractions]
        lcm = denominators[0]
        for d in denominators[1:]:
            lcm = lcm * d // gcd(lcm, d)
        
        # Multiply all by LCM
        integers = np.array([int(f * lcm) for f in fractions])
        
        # Reduce by GCD
        common_gcd = reduce(gcd, integers)
        integers = integers // common_gcd
        
        return integers
    
    def _count_elements_side(
        self, 
        molecules: List[str], 
        coefficients: List[int]
    ) -> Dict[str, int]:
        """Count total elements on one side of equation.
        
        Args:
            molecules: List of molecule formulas
            coefficients: Coefficients for each molecule
            
        Returns:
            Dictionary of element counts
        """
        total = defaultdict(int)
        for molecule, coef in zip(molecules, coefficients):
            elements = self.parse_formula(molecule)
            for element, count in elements.items():
                total[element] += count * coef
        return dict(total)
    
    def balance_from_string(self, equation_str: str) -> ChemicalEquation:
        """Balance an equation from string format.
        
        Args:
            equation_str: Equation string (e.g., "H2 + O2 -> H2O")
            
        Returns:
            Balanced ChemicalEquation object
        """
        # Split by arrow
        arrow_patterns = ['→', '->', '=']
        for arrow in arrow_patterns:
            if arrow in equation_str:
                left, right = equation_str.split(arrow, 1)
                break
        else:
            raise ValueError("No arrow found in equation")
        
        # Parse reactants and products
        reactants = [r.strip() for r in left.split('+')]
        products = [p.strip() for p in right.split('+')]
        
        # Remove any existing coefficients (we'll balance from scratch)
        reactants = [re.sub(r'^\d+\s*', '', r) for r in reactants]
        products = [re.sub(r'^\d+\s*', '', p) for p in products]
        
        # Balance
        coef_reactants, coef_products = self.balance_equation(reactants, products)
        
        return ChemicalEquation(
            reactants=reactants,
            products=products,
            coefficients_reactants=coef_reactants,
            coefficients_products=coef_products,
        )
    
    def get_balancing_steps(
        self, 
        reactants: List[str], 
        products: List[str]
    ) -> List[str]:
        """Get step-by-step balancing explanation.
        
        Args:
            reactants: List of reactant formulas
            products: List of product formulas
            
        Returns:
            List of explanation steps
        """
        steps = []
        
        # Step 1: Write unbalanced equation
        unbalanced = " + ".join(reactants) + " → " + " + ".join(products)
        steps.append(f"1. Write the unbalanced equation: {unbalanced}")
        
        # Step 2: List elements
        reactant_elements = [self.parse_formula(r) for r in reactants]
        product_elements = [self.parse_formula(p) for p in products]
        all_elements = set()
        for elem_dict in reactant_elements + product_elements:
            all_elements.update(elem_dict.keys())
        
        steps.append(f"2. Elements present: {', '.join(sorted(all_elements))}")
        
        # Step 3: Balance
        coef_r, coef_p = self.balance_equation(reactants, products)
        
        balanced_eq = ChemicalEquation(
            reactants=reactants,
            products=products,
            coefficients_reactants=coef_r,
            coefficients_products=coef_p,
        )
        
        steps.append(f"3. Balance using matrix method")
        steps.append(f"4. Balanced equation: {balanced_eq}")
        
        # Step 5: Verify
        steps.append("5. Verify by counting atoms on each side:")
        left_count = self._count_elements_side(reactants, coef_r)
        right_count = self._count_elements_side(products, coef_p)
        
        for element in sorted(all_elements):
            left = left_count.get(element, 0)
            right = right_count.get(element, 0)
            check = "✓" if left == right else "✗"
            steps.append(f"   {element}: {left} (left) = {right} (right) {check}")
        
        return steps
