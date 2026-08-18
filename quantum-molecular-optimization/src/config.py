"""Configuration management for Quantum Molecular Optimization."""

import os
from pathlib import Path
from typing import Any, Optional

import yaml


class Config:
    """Configuration manager for the VQE molecular optimization project.
    
    This class loads configuration from YAML files and provides access
    to all configurable parameters throughout the application.
    
    Attributes:
        config: Dictionary containing all configuration parameters.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration from YAML file.
        
        Args:
            config_path: Path to configuration YAML file. If None, uses
                        default config.yaml in project root.
        """
        if config_path is None:
            # Default to config.yaml in project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Override with local config if it exists
        local_config_path = self.config_path.parent / "config.local.yaml"
        if local_config_path.exists():
            local_config = self._load_yaml(local_config_path)
            self._deep_update(self.config, local_config)
    
    def _load_yaml(self, path: Path) -> dict[str, Any]:
        """Load YAML file into dictionary.
        
        Args:
            path: Path to YAML file.
            
        Returns:
            Dictionary containing YAML contents.
        """
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_config(self) -> dict[str, Any]:
        """Load main configuration file.
        
        Returns:
            Dictionary containing configuration.
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        return self._load_yaml(self.config_path)
    
    def _deep_update(self, base: dict[str, Any], override: dict[str, Any]) -> None:
        """Recursively update nested dictionary.
        
        Args:
            base: Base dictionary to update.
            override: Dictionary with values to override.
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value
    
    def get(self, *keys: str, default: Any = None) -> Any:
        """Get nested configuration value.
        
        Args:
            *keys: Sequence of keys to navigate nested config.
            default: Default value if key not found.
            
        Returns:
            Configuration value or default.
            
        Example:
            >>> config = Config()
            >>> config.get('molecule', 'name')
            'H2'
            >>> config.get('vqe', 'optimizer')
            'COBYLA'
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    @property
    def molecule_name(self) -> str:
        """Get molecule name."""
        return self.get('molecule', 'name', default='H2')
    
    @property
    def basis(self) -> str:
        """Get basis set name."""
        return self.get('molecule', 'basis', default='sto3g')
    
    @property
    def bond_distance(self) -> float:
        """Get bond distance in Angstroms."""
        return self.get('molecule', 'bond_distance', default=0.74)
    
    @property
    def ansatz_type(self) -> str:
        """Get ansatz type."""
        return self.get('vqe', 'ansatz', default='EfficientSU2')
    
    @property
    def optimizer_name(self) -> str:
        """Get optimizer name."""
        return self.get('vqe', 'optimizer', default='COBYLA')
    
    @property
    def max_iterations(self) -> int:
        """Get maximum VQE iterations."""
        return self.get('vqe', 'max_iterations', default=100)
    
    @property
    def shots(self) -> Optional[int]:
        """Get number of measurement shots."""
        return self.get('simulator', 'shots', default=1024)
    
    @property
    def seed(self) -> int:
        """Get random seed."""
        return self.get('simulator', 'seed', default=42)
    
    @property
    def noise_enabled(self) -> bool:
        """Check if noise simulation is enabled."""
        return self.get('noise', 'enabled', default=False)
    
    @property
    def error_mitigation_enabled(self) -> bool:
        """Check if error mitigation is enabled."""
        return self.get('error_mitigation', 'enabled', default=False)
    
    def __repr__(self) -> str:
        return f"Config({self.config_path})"
