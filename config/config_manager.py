# src/config/config_manager.py
"""
Config Manager - Lädt und validiert alle Konfigurationen
"""
import yaml
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class ADSConfig:
    """ADS Design Konfiguration"""
    name: str
    sae_level: int
    functions: Dict[str, bool]
    sensors: Dict[str, Dict[str, Any]]
    tactical_params: Dict[str, float]
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ADSConfig':
        return cls(**data)


@dataclass
class SensorConfig:
    """Sensor Model configuration"""
    name: str
    detection_error_rate: float
    false_positive_rate: float
    distance_noise_std: float
    velocity_noise_std: float
    latency: float
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SensorConfig':
        return cls(**data)


@dataclass
class ScenarioConfig:
    """Szenario configuration"""
    name: str
    environment: str
    duration: int
    traffic_density: str
    vehicle_count: int
    env_params: Dict[str, Any]
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ScenarioConfig':
        return cls(
            name=data['name'],
            environment=data['environment'],
            duration=data['duration'],
            traffic_density=data['traffic_density'],
            vehicle_count=data['vehicle_count'],
            env_params=data.get('env_params', {})
        )


class ConfigManager:
    """loads and validates all configurations"""
    
    def __init__(self, config_dir: str = "src/config"):
        self.config_dir = Path(config_dir)
        self._ads_configs = {}
        self._sensor_configs = {}
        self._scenario_configs = {}
        
    def load_all(self):
        """loads all configurations from YAML files"""
        self.load_ads_configs()
        self.load_sensor_configs()
        self.load_scenario_configs()
        
    def load_ads_configs(self):
        """loads ADS Design configurationen"""
        config_file = self.config_dir / "ads.yaml"
        with open(config_file, 'r') as f:
            data = yaml.safe_load(f)
        
        for name, config in data['ads_designs'].items():
            self._ads_configs[name] = ADSConfig.from_dict({
                'name': name,
                **config
            })
    
    def load_sensor_configs(self):
        """loads Sensor Model configurations"""
        config_file = self.config_dir / "sensor_models.yaml"
        with open(config_file, 'r') as f:
            data = yaml.safe_load(f)
        
        for name, config in data['sensor_models'].items():
            self._sensor_configs[name] = SensorConfig.from_dict({
                'name': name,
                **config
            })
    
    def load_scenario_configs(self):
        """loads Scenario configurations"""
        config_file = self.config_dir / "scenarios.yaml"
        with open(config_file, 'r') as f:
            data = yaml.safe_load(f)
        
        for name, config in data['scenarios'].items():
            self._scenario_configs[name] = ScenarioConfig.from_dict({
                'name': name,
                **config
            })
    
    def get_ads_config(self, name: str) -> ADSConfig:
        """Returns ADS Design configuration"""
        if name not in self._ads_configs:
            raise ValueError(f"ADS Config '{name}' not found. Avaliable: {list(self._ads_configs.keys())}")
        return self._ads_configs[name]
    
    def get_sensor_config(self, name: str) -> SensorConfig:
        """Returns Sensor Model configuration"""
        if name not in self._sensor_configs:
            raise ValueError(f"Sensor Config '{name}' not found. Avaliable: {list(self._sensor_configs.keys())}")
        return self._sensor_configs[name]
    
    def get_scenario_config(self, name: str) -> ScenarioConfig:
        """Returns Scenario configuration"""
        if name not in self._scenario_configs:
            raise ValueError(f"Scenario Config '{name}' not found. Avaliable: {list(self._scenario_configs.keys())}")
        return self._scenario_configs[name]
    
    def list_available_configs(self) -> Dict[str, list]:
        """Returns a list of available configurations"""
        return {
            'ads_designs': list(self._ads_configs.keys()),
            'sensor_models': list(self._sensor_configs.keys()),
            'scenarios': list(self._scenario_configs.keys())
        }