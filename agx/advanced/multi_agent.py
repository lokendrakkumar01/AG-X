"""
Multi-Agent Parallel Simulation
================================

Run multiple AI agents in parallel exploring different hypotheses.
"""

from __future__ import annotations
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass, field
import uuid


@dataclass
class AgentResult:
    """Result from a single agent simulation."""
    agent_id: str
    hypothesis: Dict[str, Any]
    results: Dict[str, Any]
    score: float
    runtime_seconds: float
    status: str = "completed"


@dataclass
class ResearchAgent:
    """An autonomous research agent with its own hypothesis."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:6])
    name: str = ""
    hypothesis: Dict[str, Any] = field(default_factory=dict)
    specialization: str = "general"  # general, optimization, exploration, validation
    
    def set_hypothesis(self, **params):
        self.hypothesis.update(params)


class MultiAgentSimulator:
    """
    Multi-Agent Research System.
    
    Coordinates multiple AI agents running parallel simulations
    with different hypotheses and aggregates results.
    """
    
    def __init__(self, n_agents: int = 4, max_workers: int = None):
        self.n_agents = n_agents
        self.max_workers = max_workers or n_agents
        self.agents: List[ResearchAgent] = []
        self.results: List[AgentResult] = []
    
    def create_agents(self, 
                     base_hypothesis: Dict[str, Any],
                     variation_params: List[str],
                     variation_ranges: Dict[str, tuple]) -> List[ResearchAgent]:
        """Create agents with varied hypotheses."""
        self.agents = []
        
        for i in range(self.n_agents):
            agent = ResearchAgent(
                name=f"Agent-{i}",
                hypothesis=base_hypothesis.copy(),
            )
            
            # Vary specified parameters
            for param in variation_params:
                if param in variation_ranges:
                    low, high = variation_ranges[param]
                    agent.hypothesis[param] = np.random.uniform(low, high)
            
            self.agents.append(agent)
        
        return self.agents
    
    def run_parallel(self,
                    simulation_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
                    score_fn: Callable[[Dict[str, Any]], float] = None) -> List[AgentResult]:
        """
        Run simulations in parallel across all agents.
        
        Args:
            simulation_fn: Function that takes hypothesis dict, returns results
            score_fn: Function that scores results (higher is better)
        """
        import time
        self.results = []
        
        def run_agent(agent: ResearchAgent) -> AgentResult:
            start = time.time()
            try:
                results = simulation_fn(agent.hypothesis)
                score = score_fn(results) if score_fn else 0.0
                status = "completed"
            except Exception as e:
                results = {"error": str(e)}
                score = float('-inf')
                status = "failed"
            
            return AgentResult(
                agent_id=agent.id,
                hypothesis=agent.hypothesis,
                results=results,
                score=score,
                runtime_seconds=time.time() - start,
                status=status,
            )
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(run_agent, agent): agent for agent in self.agents}
            
            for future in as_completed(futures):
                result = future.result()
                self.results.append(result)
        
        # Sort by score
        self.results.sort(key=lambda r: r.score, reverse=True)
        return self.results
    
    def get_best_results(self, n: int = 3) -> List[AgentResult]:
        """Get top N results by score."""
        return self.results[:n]
    
    def aggregate_insights(self) -> Dict[str, Any]:
        """Aggregate insights from all agent results."""
        if not self.results:
            return {}
        
        successful = [r for r in self.results if r.status == "completed"]
        
        # Parameter analysis
        param_scores = {}
        for result in successful:
            for param, value in result.hypothesis.items():
                if param not in param_scores:
                    param_scores[param] = []
                param_scores[param].append((value, result.score))
        
        # Find best parameters
        best_params = {}
        for param, values in param_scores.items():
            if values:
                best_idx = np.argmax([v[1] for v in values])
                best_params[param] = values[best_idx][0]
        
        return {
            "total_agents": len(self.agents),
            "successful_runs": len(successful),
            "best_score": self.results[0].score if self.results else 0,
            "best_hypothesis": self.results[0].hypothesis if self.results else {},
            "recommended_params": best_params,
        }
    
    def digital_twin_test(self,
                         hypothesis: Dict[str, Any],
                         simulation_fn: Callable,
                         n_replications: int = 10) -> Dict[str, Any]:
        """
        Test hypothesis with multiple replications (digital twin style).
        """
        results = []
        
        for i in range(n_replications):
            # Add small random noise to hypothesis
            noisy_hypothesis = hypothesis.copy()
            for key, value in noisy_hypothesis.items():
                if isinstance(value, (int, float)):
                    noisy_hypothesis[key] = value * (1 + np.random.normal(0, 0.01))
            
            result = simulation_fn(noisy_hypothesis)
            results.append(result)
        
        return {
            "n_replications": n_replications,
            "results": results,
            "consistency": self._calculate_consistency(results),
        }
    
    def _calculate_consistency(self, results: List[Dict]) -> float:
        """Calculate consistency score across replications."""
        if not results:
            return 0.0
        
        # Check if key metrics are consistent
        try:
            scores = [r.get("score", r.get("energy", 0)) for r in results]
            mean = np.mean(scores)
            std = np.std(scores)
            return 1.0 / (1.0 + std / (abs(mean) + 1e-10))
        except:
            return 0.5
