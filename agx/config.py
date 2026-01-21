"""
AG-X 2026 Configuration System
==============================

Centralized configuration management with validation, environment variable support,
and hierarchical config loading from YAML files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal
from enum import Enum

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
from loguru import logger


# =============================================================================
# Environment Configuration
# =============================================================================

class Environment(str, Enum):
    """Deployment environment."""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


class AGXSettings(BaseSettings):
    """Global environment settings loaded from .env or environment variables."""
    
    # Environment
    env: Environment = Field(default=Environment.DEVELOPMENT, alias="AGX_ENV")
    debug: bool = Field(default=False, alias="AGX_DEBUG")
    log_level: str = Field(default="INFO", alias="AGX_LOG_LEVEL")
    
    # Paths
    data_dir: Path = Field(default=Path("./data"), alias="AGX_DATA_DIR")
    experiments_dir: Path = Field(default=Path("./experiments"), alias="AGX_EXPERIMENTS_DIR")
    reports_dir: Path = Field(default=Path("./reports"), alias="AGX_REPORTS_DIR")
    
    # Compute
    use_gpu: bool = Field(default=False, alias="AGX_USE_GPU")
    num_workers: int = Field(default=4, alias="AGX_NUM_WORKERS")
    random_seed: int = Field(default=42, alias="AGX_RANDOM_SEED")
    
    # Web Dashboard
    web_host: str = Field(default="0.0.0.0", alias="AGX_WEB_HOST")
    web_port: int = Field(default=8050, alias="AGX_WEB_PORT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# =============================================================================
# Physics Configuration
# =============================================================================

class NewtonianConfig(BaseModel):
    """Newtonian mechanics configuration."""
    enabled: bool = True
    gravitational_constant: float = Field(default=6.67430e-11, description="G in m³/(kg·s²)")
    max_bodies: int = Field(default=1000, ge=1, le=100000)
    softening_length: float = Field(default=1e-6, ge=0, description="Softening to prevent singularities")


class GeneralRelativityConfig(BaseModel):
    """General Relativity approximation configuration."""
    enabled: bool = True
    speed_of_light: float = Field(default=299792458.0, description="c in m/s")
    schwarzschild_approx: bool = True
    include_frame_dragging: bool = False
    curvature_resolution: int = Field(default=64, ge=8, le=512)


class QuantumFieldConfig(BaseModel):
    """Quantum field simulation configuration."""
    enabled: bool = True
    vacuum_energy_density: float = Field(default=1e-9, description="Hypothetical vacuum energy")
    field_resolution: int = Field(default=32, ge=8, le=256)
    fluctuation_amplitude: float = Field(default=1e-12, ge=0)
    casimir_effect: bool = True


class SpeculativeConfig(BaseModel):
    """Speculative/hypothetical physics configuration.
    
    ⚠️ WARNING: All parameters in this section are PURELY THEORETICAL
    and do not represent real physics. They are for simulation exploration only.
    """
    enabled: bool = True
    negative_mass_enabled: bool = Field(default=True, description="[THEORETICAL] Negative mass particles")
    exotic_energy_fields: bool = Field(default=True, description="[THEORETICAL] Exotic energy constructs")
    dark_energy_repulsion: bool = Field(default=True, description="[THEORETICAL] Dark-energy-like forces")
    warp_field_model: bool = Field(default=False, description="[THEORETICAL] Alcubierre-inspired concepts")
    
    # Stability constraints to prevent non-physical instabilities
    max_negative_mass_ratio: float = Field(default=0.1, ge=0, le=1.0)
    energy_conservation_tolerance: float = Field(default=1e-6, ge=0)
    stability_check_interval: int = Field(default=10, ge=1)


class SolverConfig(BaseModel):
    """Numerical solver configuration."""
    method: Literal["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"] = "DOP853"
    adaptive_step: bool = True
    min_step: float = Field(default=1e-12, gt=0)
    max_step: float = Field(default=1.0, gt=0)
    rtol: float = Field(default=1e-8, gt=0, description="Relative tolerance")
    atol: float = Field(default=1e-10, gt=0, description="Absolute tolerance")
    
    # Advanced options
    stochastic_noise: bool = False
    noise_amplitude: float = Field(default=1e-10, ge=0)
    chaos_detection: bool = True
    lyapunov_window: int = Field(default=100, ge=10)


class PhysicsConfig(BaseModel):
    """Complete physics simulation configuration."""
    newtonian: NewtonianConfig = Field(default_factory=NewtonianConfig)
    general_relativity: GeneralRelativityConfig = Field(default_factory=GeneralRelativityConfig)
    quantum_field: QuantumFieldConfig = Field(default_factory=QuantumFieldConfig)
    speculative: SpeculativeConfig = Field(default_factory=SpeculativeConfig)
    solver: SolverConfig = Field(default_factory=SolverConfig)
    
    # Simulation parameters
    timesteps: int = Field(default=1000, ge=1, le=1000000)
    dt: float = Field(default=0.01, gt=0, description="Time step in simulation units")
    dimensions: Literal[2, 3] = 3


# =============================================================================
# AI Configuration
# =============================================================================

class ExplorerConfig(BaseModel):
    """Parameter space exploration configuration."""
    enabled: bool = True
    surrogate_model: Literal["mlp", "gp", "ensemble"] = "mlp"
    hidden_layers: List[int] = Field(default=[256, 128, 64])
    bayesian_samples: int = Field(default=100, ge=10)
    uncertainty_threshold: float = Field(default=0.1, ge=0)


class OptimizerConfig(BaseModel):
    """Reinforcement learning optimizer configuration."""
    enabled: bool = True
    algorithm: Literal["PPO", "SAC", "A2C", "TD3"] = "PPO"
    learning_rate: float = Field(default=3e-4, gt=0)
    batch_size: int = Field(default=64, ge=1)
    training_steps: int = Field(default=10000, ge=100)
    reward_scaling: float = Field(default=1.0, gt=0)


class AnomalyConfig(BaseModel):
    """Anomaly detection configuration."""
    enabled: bool = True
    method: Literal["autoencoder", "isolation_forest", "lstm"] = "autoencoder"
    contamination: float = Field(default=0.01, ge=0, le=0.5)
    threshold_std: float = Field(default=3.0, gt=0)


class ExplainerConfig(BaseModel):
    """Explainable AI configuration."""
    enabled: bool = True
    method: Literal["shap", "lime", "attention"] = "shap"
    num_samples: int = Field(default=100, ge=10)
    generate_text: bool = True


class AIConfig(BaseModel):
    """Complete AI layer configuration."""
    explorer: ExplorerConfig = Field(default_factory=ExplorerConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    anomaly: AnomalyConfig = Field(default_factory=AnomalyConfig)
    explainer: ExplainerConfig = Field(default_factory=ExplainerConfig)


# =============================================================================
# Visualization Configuration
# =============================================================================

class VisualizationConfig(BaseModel):
    """Visualization and dashboard configuration."""
    theme: Literal["dark", "light", "scientific"] = "dark"
    colormap: str = "viridis"
    particle_size: float = Field(default=5.0, gt=0)
    vector_scale: float = Field(default=1.0, gt=0)
    grid_resolution: int = Field(default=32, ge=8, le=128)
    animation_fps: int = Field(default=30, ge=1, le=120)
    
    # 3D rendering
    camera_distance: float = Field(default=10.0, gt=0)
    fov: float = Field(default=45.0, gt=0, le=180)
    
    # Dashboard
    update_interval_ms: int = Field(default=100, ge=16)
    max_history_points: int = Field(default=1000, ge=100)


# =============================================================================
# Chemistry Configuration
# =============================================================================

class ChemistryConfig(BaseModel):
    """Chemistry module configuration."""
    enabled: bool = True
    molecular_viz_enabled: bool = True
    equation_balancer: bool = True
    virtual_lab: bool = True
    
    # Molecular visualization
    max_atoms: int = Field(default=500, ge=1, le=10000)
    visualization_format: Literal["3d", "2d", "both"] = "3d"
    
    # Virtual lab safety
    enable_hazard_warnings: bool = True
    allow_explosive_reactions: bool = Field(default=False, description="Educational only")
    

class MathematicsConfig(BaseModel):
    """Mathematics module configuration."""
    enabled: bool = True
    symbolic_computation: bool = True
    graphing_enabled: bool = True
    step_by_step_solutions: bool = True
    
    # Computation limits
    max_equation_complexity: int = Field(default=1000, ge=1)
    timeout_seconds: int = Field(default=30, ge=1, le=300)
    
    # Graphing
    graph_resolution: int = Field(default=1000, ge=100, le=10000)
    max_plot_points: int = Field(default=100000, ge=1000)
    

class ComputerScienceConfig(BaseModel):
    """Computer Science module configuration."""
    enabled: bool = True
    algorithm_visualizer: bool = True
    os_simulator: bool = True
    db_visualizer: bool = True
    network_simulator: bool = True
    
    # Simulation limits
    max_processes: int = Field(default=50, ge=1, le=1000)
    max_network_nodes: int = Field(default=100, ge=1, le=1000)
    

class DSAConfig(BaseModel):
    """Data Structures & Algorithms practice configuration."""
    enabled: bool = True
    code_execution_enabled: bool = True
    visualization_enabled: bool = True
    
    # Supported languages
    supported_languages: List[str] = Field(default=[
        "python", "java", "cpp", "c", "javascript", "go", 
        "rust", "php", "csharp", "kotlin", "swift", "ruby"
    ])
    
    # Execution limits
    max_execution_time_seconds: int = Field(default=10, ge=1, le=60)
    max_memory_mb: int = Field(default=256, ge=32, le=2048)
    max_output_size_kb: int = Field(default=1024, ge=1, le=10240)
    
    # Problem settings
    difficulty_levels: List[str] = Field(default=["beginner", "intermediate", "advanced"])
    enable_hints: bool = True
    show_complexity_analysis: bool = True
    

class ProgrammingConfig(BaseModel):
    """Programming languages module configuration."""
    enabled: bool = True
    
    # Language support
    available_languages: List[str] = Field(default=[
        "python", "java", "cpp", "c", "javascript", "typescript",
        "go", "rust", "php", "csharp", "kotlin", "swift", "ruby", "scala"
    ])
    
    code_formatting: bool = True
    syntax_highlighting: bool = True
    

class CommunityConfig(BaseModel):
    """Community and user-generated content configuration."""
    enabled: bool = True
    user_uploads_enabled: bool = True
    moderation_enabled: bool = True
    
    # Content limits
    max_note_size_kb: int = Field(default=500, ge=1, le=5120)
    max_uploads_per_day: int = Field(default=20, ge=1, le=100)
    
    # Moderation
    auto_moderation: bool = True
    require_approval: bool = Field(default=False, description="New content requires approval")
    plagiarism_check: bool = True
    
    # Interactions
    enable_comments: bool = True
    enable_ratings: bool = True
    enable_bookmarks: bool = True
    

class EnglishConfig(BaseModel):
    """English communication module configuration."""
    enabled: bool = True
    daily_vocabulary: bool = True
    grammar_checker: bool = True
    conversation_practice: bool = True
    
    # Vocabulary
    words_per_day: int = Field(default=5, ge=1, le=50)
    difficulty_levels: List[str] = Field(default=["basic", "intermediate", "advanced"])
    
    # AI conversation
    ai_conversation_enabled: bool = True
    max_conversation_turns: int = Field(default=20, ge=1, le=100)
    

class DatabaseConfig(BaseModel):
    """Database configuration for user data and content."""
    database_url: str = Field(default="sqlite:///agx_2026.db")
    echo_sql: bool = Field(default=False, description="Print SQL queries")
    
    # Connection pool
    pool_size: int = Field(default=5, ge=1, le=100)
    max_overflow: int = Field(default=10, ge=0, le=100)
    
    # Performance
    enable_query_cache: bool = True
    cache_ttl_seconds: int = Field(default=300, ge=0)
    

class AuthConfig(BaseModel):
    """Authentication and authorization configuration."""
    enabled: bool = True
    
    # JWT settings
    secret_key: str = Field(default="CHANGE_THIS_SECRET_KEY_IN_PRODUCTION")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = Field(default=30, ge=5, le=1440)
    refresh_token_expire_days: int = Field(default=7, ge=1, le=30)
    
    # Password settings
    min_password_length: int = Field(default=8, ge=6, le=128)
    require_special_char: bool = True
    require_number: bool = True
    
    # OAuth
    enable_oauth: bool = False
    oauth_providers: List[str] = Field(default=[])


# =============================================================================
# Experiment Configuration
# =============================================================================

class ExperimentConfig(BaseModel):
    """Experiment management configuration."""
    auto_save: bool = True
    save_interval: int = Field(default=100, ge=1, description="Save checkpoint every N steps")
    versioning: bool = True
    compression: Literal["none", "gzip", "lz4"] = "gzip"
    
    # Reports
    auto_report: bool = True
    report_format: Literal["pdf", "markdown", "both"] = "both"
    include_equations: bool = True
    include_ai_explanations: bool = True


# =============================================================================
# Master Configuration
# =============================================================================

class AGXConfig(BaseModel):
    """Master configuration for AG-X 2026 platform."""
    
    name: str = Field(default="AG-X 2026 Universal Platform", description="Platform name")
    description: str = Field(default="", description="Platform description")
    
    # Core modules (existing)
    physics: PhysicsConfig = Field(default_factory=PhysicsConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    
    # New educational modules
    chemistry: ChemistryConfig = Field(default_factory=ChemistryConfig)
    mathematics: MathematicsConfig = Field(default_factory=MathematicsConfig)
    computer_science: ComputerScienceConfig = Field(default_factory=ComputerScienceConfig)
    dsa: DSAConfig = Field(default_factory=DSAConfig)
    programming: ProgrammingConfig = Field(default_factory=ProgrammingConfig)
    community: CommunityConfig = Field(default_factory=CommunityConfig)
    english: EnglishConfig = Field(default_factory=EnglishConfig)
    
    # Infrastructure
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "AGXConfig":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"Config file not found: {path}, using defaults")
            return cls()
        
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        return cls(**data) if data else cls()
    
    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)
    
    def merge_with(self, overrides: Dict[str, Any]) -> "AGXConfig":
        """Merge configuration with override dictionary."""
        current = self.model_dump()
        
        def deep_merge(base: dict, update: dict) -> dict:
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
            return base
        
        merged = deep_merge(current, overrides)
        return AGXConfig(**merged)


# =============================================================================
# Global Settings Instance
# =============================================================================

settings = AGXSettings()


def get_config(config_path: Optional[str | Path] = None) -> AGXConfig:
    """Get configuration, optionally loading from file."""
    if config_path:
        return AGXConfig.from_yaml(config_path)
    
    # Try default config locations
    default_paths = [
        Path("configs/default.yaml"),
        Path("config.yaml"),
        Path.home() / ".agx" / "config.yaml",
    ]
    
    for path in default_paths:
        if path.exists():
            return AGXConfig.from_yaml(path)
    
    return AGXConfig()


def setup_logging() -> None:
    """Configure logging based on settings."""
    import sys
    
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
    )
    
    if settings.env == Environment.PRODUCTION:
        logger.add(
            settings.data_dir / "logs" / "agx_{time}.log",
            rotation="10 MB",
            retention="7 days",
            level="DEBUG",
        )
