import gymnasium as gym
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from src.config.config_manager import ScenarioConfig


class SimulatorAdapter:
    """
    Improved Interface to Highway-env simulator with better vehicle spawning
    """
    
    def __init__(self, scenario_config: ScenarioConfig):
        self.scenario_config = scenario_config
        self.env: Optional[gym.Env] = None
        self.current_step = 0
        
    def reset(self) -> Dict[str, Any]:
        """
        Initialize/reset the simulator and return initial ground truth.
        
        Returns:
            Ground truth dictionary with ego and vehicle states
        """
        if self.env is None:
            # Create environment
            self.env = gym.make(
                self.scenario_config.environment,
                render_mode='human'
            )
            self._configure_environment()
        
        # Reset environment
        obs, info = self.env.reset()
        self.current_step = 0
        
        return self.extract_ground_truth(obs)
    
    def _configure_environment(self):
        """Configure Highway-env parameters with improved vehicle spawning"""
        # Type guard: Ensure env is not None
        if self.env is None:
            raise RuntimeError("Environment must be created before configuration")
        
        env_params = self.scenario_config.env_params

        config_dict = {
            "lanes_count": env_params.get("lanes_count", 4),
            "vehicles_count": self.scenario_config.vehicle_count,
            "duration": self.scenario_config.duration,
            "policy_frequency": env_params.get('policy_frequency', 1),  # 1 Hz for tactical decisions
            "simulation_frequency": env_params.get('simulation_frequency', 10),  # 10 Hz simulation
            "manual_control": False,

            
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "initial_lane_id": None,
            "ego_spacing": 2.0,
            "vehicles_density": 1.5,
            "spawn_probability": 0.8,


            "collision_reward": -1.0,
            "right_lane_reward": 0.1,  
            "high_speed_reward": 0.6,  
            "lane_change_reward": -0.05,
            "reward_speed_range": [25, 35], 
            "normalize_reward": True,
            "offroad_terminal": True,

            # Rendering configuration
            "screen_width": 800,  
            "screen_height": 200,  
            "centering_position": [0.3, 0.5],
            "scaling": 6.0,  
            "show_trajectories": False,
            "render_agent": True,

            # Observation configuration
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 15,
                "features": ["presence", "x", "y", "vx", "vy"],
                "absolute": False,
                "normalize": False,
                "see_behind": True,  
                "observe_intentions": False
            }
        }
        
        # Access config attribute
        if hasattr(self.env, 'unwrapped'):
            self.env.unwrapped.config.update(config_dict)  # type: ignore[attr-defined]
        elif hasattr(self.env, 'config'):
            self.env.config.update(config_dict)  # type: ignore[attr-defined]
        else:
            raise RuntimeError("Environment does not have 'config' attribute.")
    
    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Execute one simulation step.
        
        Args:
            action: Action code (0-4)
            
        Returns:
            Tuple of (ground_truth, reward, done, truncated, info)
        """
        # Type guard
        if self.env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        # Execute step
        obs, reward, done, truncated, info = self.env.step(action)
        self.current_step += 1
        
        # Extract ground truth
        ground_truth = self.extract_ground_truth(obs)
        
        # Add vehicle count to info for monitoring
        info['vehicle_count'] = len(ground_truth['vehicles'])
        
        # Convert reward to float
        reward_float = float(reward)
        
        return ground_truth, reward_float, done, truncated, info
    
    def extract_ground_truth(self, obs: np.ndarray) -> Dict[str, Any]:
        """
        Extract Ground Truth from Highway-env observation.
        
        Highway-env Observation Format:
        - obs[0] = ego vehicle
        - obs[1:] = other vehicles
        - Each vehicle: [presence, x, y, vx, vy]
        
        Args:
            obs: numpy array from Highway-env [vehicles, features]
            
        Returns:
            Dictionary with Ground Truth data
        """
        # Ego vehicle is always first
        ego_data = obs[0]
        
        # Build ego state
        ego_vx = float(ego_data[3])
        ego_vy = float(ego_data[4])
        ego_x = float(ego_data[1])
        ego_y = float(ego_data[2])
        
        ground_truth = {
            'ego': {
                'position': (ego_x, ego_y),
                'velocity': float(np.sqrt(ego_vx**2 + ego_vy**2)),
                'velocity_x': ego_vx,
                'velocity_y': ego_vy,
                'heading': float(np.arctan2(ego_vy, ego_vx)),
                'lane_index': self._estimate_lane_index(ego_y),
                'on_road': True
            },
            'vehicles': [],
            'timestamp': self.current_step * 0.1  # 10 Hz = 0.1s per step
        }
        
        # Process other vehicles
        for i in range(1, len(obs)):
            vehicle_data = obs[i]
            
            # presence=0 means no vehicle in this slot
            if vehicle_data[0] == 0:
                continue
            
            # Relative position to ego
            rel_x = float(vehicle_data[1])
            rel_y = float(vehicle_data[2])
            vx = float(vehicle_data[3])
            vy = float(vehicle_data[4])
            
            # Reconstruct absolute position
            abs_x = ego_x + rel_x
            abs_y = ego_y + rel_y
            
            vehicle_info = {
                'id': f'vehicle_{i}',
                'position': (abs_x, abs_y),
                'velocity': float(np.sqrt(vx**2 + vy**2)),
                'velocity_x': vx,
                'velocity_y': vy,
                'heading': float(np.arctan2(vy, vx)),
                'lane_index': self._estimate_lane_index(abs_y),
                'relative_longitudinal': rel_x,  # positive = ahead
                'relative_lateral': rel_y  # positive = left
            }
            
            ground_truth['vehicles'].append(vehicle_info)
        
        return ground_truth
    
    def _estimate_lane_index(self, y_position: float) -> int:
        """
        Estimate lane index from Y-position.
        
        Args:
            y_position: Y-coordinate in meters
            
        Returns:
            Lane index (0-3)
        """
        lane_width = self.scenario_config.env_params.get('lane_width', 4.0)
        lanes_count = self.scenario_config.env_params.get('lanes_count', 4)

        # Add half lane width offset and divide by lane width
        lane = int((y_position + lane_width / 2.0) / lane_width)
        
        # Clamp to valid range
        return max(0, min(lanes_count - 1, lane))
    
    def render(self):
        """Render current simulation state (if render_mode is set)"""
        if self.env is not None:
            self.env.render()
    
    def close(self):
        """Close environment and cleanup resources"""
        if self.env is not None:
            self.env.close()
            self.env = None