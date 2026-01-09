"""
Experiment Manager
==================

Reproducible experiment lifecycle management with versioning and checkpointing.
"""

from __future__ import annotations
import json
import hashlib
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict

import numpy as np


@dataclass
class Experiment:
    """Represents a single experiment run."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    random_seed: int = 42
    
    # Results
    results: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    status: str = "created"  # created, running, completed, failed
    
    # Versioning
    version: int = 1
    parent_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def get_hash(self) -> str:
        """Generate hash of experiment configuration."""
        config_str = json.dumps(self.config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experiment":
        return cls(**data)


class ExperimentManager:
    """
    Manage experiment lifecycle with reproducibility.
    
    Features:
    - Reproducible random seeds
    - Configuration versioning
    - Checkpointing and resume
    - Metadata tracking
    """
    
    def __init__(self, experiments_dir: str = "./experiments"):
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.current_experiment: Optional[Experiment] = None
        self._experiments_index: Dict[str, Experiment] = {}
        self._load_index()
    
    def _load_index(self) -> None:
        """Load experiments index from disk."""
        index_file = self.experiments_dir / "index.json"
        if index_file.exists():
            with open(index_file, "r") as f:
                data = json.load(f)
                for exp_data in data.get("experiments", []):
                    exp = Experiment.from_dict(exp_data)
                    self._experiments_index[exp.id] = exp
    
    def _save_index(self) -> None:
        """Save experiments index to disk."""
        index_file = self.experiments_dir / "index.json"
        with open(index_file, "w") as f:
            json.dump({
                "experiments": [e.to_dict() for e in self._experiments_index.values()]
            }, f, indent=2)
    
    def create_experiment(self, 
                         name: str,
                         config: Dict[str, Any],
                         description: str = "",
                         seed: int = 42,
                         tags: List[str] = None) -> Experiment:
        """Create a new experiment."""
        # Set random seeds for reproducibility
        np.random.seed(seed)
        
        exp = Experiment(
            name=name,
            description=description,
            config=config,
            random_seed=seed,
            tags=tags or [],
        )
        
        self.current_experiment = exp
        self._experiments_index[exp.id] = exp
        
        # Create experiment directory
        exp_dir = self.experiments_dir / exp.id
        exp_dir.mkdir(exist_ok=True)
        
        self._save_experiment(exp)
        return exp
    
    def _save_experiment(self, exp: Experiment) -> None:
        """Save experiment to disk."""
        exp_dir = self.experiments_dir / exp.id
        exp_dir.mkdir(exist_ok=True)
        
        with open(exp_dir / "experiment.json", "w") as f:
            json.dump(exp.to_dict(), f, indent=2)
        
        self._save_index()
    
    def load_experiment(self, exp_id: str) -> Experiment:
        """Load an experiment by ID."""
        exp_dir = self.experiments_dir / exp_id
        exp_file = exp_dir / "experiment.json"
        
        if not exp_file.exists():
            raise ValueError(f"Experiment {exp_id} not found")
        
        with open(exp_file, "r") as f:
            data = json.load(f)
        
        exp = Experiment.from_dict(data)
        self.current_experiment = exp
        
        # Restore random seed
        np.random.seed(exp.random_seed)
        
        return exp
    
    def start_run(self) -> None:
        """Mark current experiment as running."""
        if self.current_experiment:
            self.current_experiment.status = "running"
            self._save_experiment(self.current_experiment)
    
    def log_metric(self, name: str, value: float) -> None:
        """Log a metric value."""
        if self.current_experiment:
            self.current_experiment.metrics[name] = value
    
    def log_result(self, name: str, value: Any) -> None:
        """Log a result value."""
        if self.current_experiment:
            # Convert numpy arrays to lists for JSON serialization
            if isinstance(value, np.ndarray):
                value = value.tolist()
            self.current_experiment.results[name] = value
    
    def save_checkpoint(self, state: Dict[str, Any], name: str = "checkpoint") -> Path:
        """Save simulation state checkpoint."""
        if not self.current_experiment:
            raise ValueError("No active experiment")
        
        exp_dir = self.experiments_dir / self.current_experiment.id
        checkpoint_file = exp_dir / f"{name}.npz"
        
        # Save numpy arrays
        arrays = {k: v for k, v in state.items() if isinstance(v, np.ndarray)}
        if arrays:
            np.savez_compressed(checkpoint_file, **arrays)
        
        # Save other state as JSON
        other = {k: v for k, v in state.items() if not isinstance(v, np.ndarray)}
        if other:
            with open(exp_dir / f"{name}.json", "w") as f:
                json.dump(other, f)
        
        return checkpoint_file
    
    def load_checkpoint(self, name: str = "checkpoint") -> Dict[str, Any]:
        """Load simulation state checkpoint."""
        if not self.current_experiment:
            raise ValueError("No active experiment")
        
        exp_dir = self.experiments_dir / self.current_experiment.id
        state = {}
        
        # Load numpy arrays
        npz_file = exp_dir / f"{name}.npz"
        if npz_file.exists():
            with np.load(npz_file) as data:
                state.update({k: data[k] for k in data.files})
        
        # Load JSON state
        json_file = exp_dir / f"{name}.json"
        if json_file.exists():
            with open(json_file, "r") as f:
                state.update(json.load(f))
        
        return state
    
    def complete_run(self) -> Experiment:
        """Mark current experiment as completed."""
        if self.current_experiment:
            self.current_experiment.status = "completed"
            self._save_experiment(self.current_experiment)
            return self.current_experiment
        raise ValueError("No active experiment")
    
    def list_experiments(self, tags: List[str] = None, status: str = None) -> List[Experiment]:
        """List experiments with optional filtering."""
        experiments = list(self._experiments_index.values())
        
        if tags:
            experiments = [e for e in experiments if any(t in e.tags for t in tags)]
        if status:
            experiments = [e for e in experiments if e.status == status]
        
        return sorted(experiments, key=lambda e: e.created_at, reverse=True)
    
    def fork_experiment(self, exp_id: str, new_config: Dict[str, Any] = None) -> Experiment:
        """Create a new experiment based on an existing one."""
        parent = self.load_experiment(exp_id)
        
        config = parent.config.copy()
        if new_config:
            config.update(new_config)
        
        new_exp = Experiment(
            name=f"{parent.name} (fork)",
            description=f"Forked from {parent.id}",
            config=config,
            random_seed=parent.random_seed + 1,
            parent_id=parent.id,
            version=parent.version + 1,
            tags=parent.tags.copy(),
        )
        
        self.current_experiment = new_exp
        self._experiments_index[new_exp.id] = new_exp
        self._save_experiment(new_exp)
        
        return new_exp
