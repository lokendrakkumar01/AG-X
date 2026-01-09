"""
Evolutionary Optimization
==========================

Genetic algorithms and NSGA-II for multi-objective optimization.
"""

from __future__ import annotations
import numpy as np
from typing import List, Callable, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class Individual:
    """An individual in the evolutionary population."""
    genes: np.ndarray
    fitness: float = 0.0
    objectives: List[float] = field(default_factory=list)
    rank: int = 0
    crowding_distance: float = 0.0
    
    def copy(self) -> "Individual":
        return Individual(
            genes=self.genes.copy(),
            fitness=self.fitness,
            objectives=self.objectives.copy(),
        )


class EvolutionaryOptimizer:
    """
    Evolutionary Optimization for Physics Parameters.
    
    Supports single and multi-objective optimization using:
    - Genetic Algorithm (GA)
    - NSGA-II multi-objective optimization
    - Novelty search
    """
    
    def __init__(self,
                 gene_bounds: np.ndarray,
                 population_size: int = 50,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8):
        """
        Args:
            gene_bounds: (n_genes, 2) array of [min, max] bounds
            population_size: Number of individuals
            mutation_rate: Probability of mutation per gene
            crossover_rate: Probability of crossover
        """
        self.gene_bounds = np.asarray(gene_bounds)
        self.n_genes = len(self.gene_bounds)
        self.pop_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        self.population: List[Individual] = []
        self.generation = 0
        self.history: List[Dict[str, Any]] = []
    
    def initialize_population(self) -> List[Individual]:
        """Create initial random population."""
        self.population = []
        
        for _ in range(self.pop_size):
            genes = np.random.uniform(
                self.gene_bounds[:, 0],
                self.gene_bounds[:, 1]
            )
            self.population.append(Individual(genes=genes))
        
        return self.population
    
    def evaluate_population(self, 
                           fitness_fn: Callable[[np.ndarray], float]) -> None:
        """Evaluate fitness for all individuals."""
        for ind in self.population:
            ind.fitness = fitness_fn(ind.genes)
    
    def tournament_selection(self, tournament_size: int = 3) -> Individual:
        """Select individual via tournament selection."""
        candidates = np.random.choice(len(self.population), tournament_size, replace=False)
        winner = max(candidates, key=lambda i: self.population[i].fitness)
        return self.population[winner].copy()
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Blend crossover between two parents."""
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        alpha = np.random.uniform(0, 1, self.n_genes)
        child1_genes = alpha * parent1.genes + (1 - alpha) * parent2.genes
        child2_genes = (1 - alpha) * parent1.genes + alpha * parent2.genes
        
        return Individual(genes=child1_genes), Individual(genes=child2_genes)
    
    def mutate(self, individual: Individual) -> Individual:
        """Gaussian mutation."""
        for i in range(self.n_genes):
            if np.random.random() < self.mutation_rate:
                range_i = self.gene_bounds[i, 1] - self.gene_bounds[i, 0]
                individual.genes[i] += np.random.normal(0, range_i * 0.1)
                individual.genes[i] = np.clip(
                    individual.genes[i],
                    self.gene_bounds[i, 0],
                    self.gene_bounds[i, 1]
                )
        return individual
    
    def evolve_generation(self, fitness_fn: Callable[[np.ndarray], float]) -> None:
        """Evolve one generation."""
        # Evaluate current population
        self.evaluate_population(fitness_fn)
        
        # Create new population
        new_population = []
        
        # Elitism: keep best individual
        best = max(self.population, key=lambda x: x.fitness)
        new_population.append(best.copy())
        
        # Create offspring
        while len(new_population) < self.pop_size:
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            new_population.extend([child1, child2])
        
        self.population = new_population[:self.pop_size]
        self.generation += 1
        
        # Record history
        self.history.append({
            "generation": self.generation,
            "best_fitness": best.fitness,
            "best_genes": best.genes.tolist(),
            "avg_fitness": np.mean([ind.fitness for ind in self.population]),
        })
    
    def optimize(self,
                fitness_fn: Callable[[np.ndarray], float],
                n_generations: int = 100,
                early_stop_generations: int = 20) -> Individual:
        """
        Run full optimization.
        
        Args:
            fitness_fn: Function to maximize
            n_generations: Maximum generations
            early_stop_generations: Stop if no improvement for this many generations
        """
        self.initialize_population()
        
        best_fitness = float('-inf')
        generations_without_improvement = 0
        
        for gen in range(n_generations):
            self.evolve_generation(fitness_fn)
            
            current_best = max(self.population, key=lambda x: x.fitness)
            
            if current_best.fitness > best_fitness:
                best_fitness = current_best.fitness
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
            
            if generations_without_improvement >= early_stop_generations:
                break
        
        return max(self.population, key=lambda x: x.fitness)
    
    def nsga2_optimize(self,
                       objective_fns: List[Callable[[np.ndarray], float]],
                       n_generations: int = 100) -> List[Individual]:
        """
        Multi-objective optimization using NSGA-II.
        
        Returns Pareto front of non-dominated solutions.
        """
        self.initialize_population()
        
        for gen in range(n_generations):
            # Evaluate all objectives
            for ind in self.population:
                ind.objectives = [fn(ind.genes) for fn in objective_fns]
            
            # Non-dominated sorting
            fronts = self._non_dominated_sort()
            
            # Calculate crowding distance
            for front in fronts:
                self._calculate_crowding_distance(front)
            
            # Create offspring
            offspring = []
            while len(offspring) < self.pop_size:
                p1 = self._nsga2_tournament()
                p2 = self._nsga2_tournament()
                c1, c2 = self.crossover(p1, p2)
                offspring.extend([self.mutate(c1), self.mutate(c2)])
            
            # Combine and select
            combined = self.population + offspring[:self.pop_size]
            
            # Evaluate offspring
            for ind in combined:
                if not ind.objectives:
                    ind.objectives = [fn(ind.genes) for fn in objective_fns]
            
            # Select next generation
            fronts = self._non_dominated_sort_combined(combined)
            self.population = []
            
            for front in fronts:
                if len(self.population) + len(front) <= self.pop_size:
                    self.population.extend(front)
                else:
                    # Sort by crowding distance
                    self._calculate_crowding_distance(front)
                    front.sort(key=lambda x: x.crowding_distance, reverse=True)
                    self.population.extend(front[:self.pop_size - len(self.population)])
                    break
        
        # Return Pareto front
        return self._get_pareto_front()
    
    def _non_dominated_sort(self) -> List[List[Individual]]:
        """Perform non-dominated sorting."""
        fronts = [[]]
        
        for p in self.population:
            p.dominated_by = 0
            p.dominates = []
            
            for q in self.population:
                if self._dominates(p, q):
                    p.dominates.append(q)
                elif self._dominates(q, p):
                    p.dominated_by += 1
            
            if p.dominated_by == 0:
                p.rank = 0
                fronts[0].append(p)
        
        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in p.dominates:
                    q.dominated_by -= 1
                    if q.dominated_by == 0:
                        q.rank = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        
        return fronts[:-1]
    
    def _non_dominated_sort_combined(self, population: List[Individual]) -> List[List[Individual]]:
        """Non-dominated sort for combined population."""
        # Simplified version
        fronts = [[]]
        for ind in population:
            dominated = False
            for other in population:
                if self._dominates(other, ind):
                    dominated = True
                    break
            if not dominated:
                ind.rank = 0
                fronts[0].append(ind)
        return fronts
    
    def _dominates(self, p: Individual, q: Individual) -> bool:
        """Check if p dominates q."""
        better_in_one = False
        for i in range(len(p.objectives)):
            if p.objectives[i] < q.objectives[i]:
                return False
            if p.objectives[i] > q.objectives[i]:
                better_in_one = True
        return better_in_one
    
    def _calculate_crowding_distance(self, front: List[Individual]) -> None:
        """Calculate crowding distance for a front."""
        n = len(front)
        if n == 0:
            return
        
        for ind in front:
            ind.crowding_distance = 0
        
        n_obj = len(front[0].objectives)
        
        for m in range(n_obj):
            front.sort(key=lambda x: x.objectives[m])
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            obj_range = front[-1].objectives[m] - front[0].objectives[m]
            if obj_range == 0:
                continue
            
            for i in range(1, n - 1):
                front[i].crowding_distance += (
                    front[i + 1].objectives[m] - front[i - 1].objectives[m]
                ) / obj_range
    
    def _nsga2_tournament(self) -> Individual:
        """NSGA-II tournament selection."""
        i1, i2 = np.random.choice(len(self.population), 2, replace=False)
        p1, p2 = self.population[i1], self.population[i2]
        
        if p1.rank < p2.rank:
            return p1.copy()
        elif p2.rank < p1.rank:
            return p2.copy()
        elif p1.crowding_distance > p2.crowding_distance:
            return p1.copy()
        else:
            return p2.copy()
    
    def _get_pareto_front(self) -> List[Individual]:
        """Get the Pareto-optimal solutions."""
        return [ind for ind in self.population if ind.rank == 0]
