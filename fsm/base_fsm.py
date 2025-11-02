from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Any, Optional, Union, TypeVar
from src.config.config_manager import ADSConfig

# Generic Type für FSM States
StateType = TypeVar('StateType', bound=Enum)


class BaseFSM(ABC):
    """
    Abstrakte Basis-Klasse für alle FSM-Implementierungen
    
    Verwendet Generics um verschiedene State-Enums zu unterstützen
    """
    
    def __init__(self, ads_config: ADSConfig):
        self.config = ads_config
        self.current_state: Enum = self.get_initial_state()
        self.time_in_state = 0.0
        self.transition_history = []

    @abstractmethod
    def get_initial_state(self) -> Enum:
        """Return the initial state of the FSM."""
        pass

    @abstractmethod
    def update(self, observation: Dict[str, Any], dt: float = 0.1) -> int:
        """
        Update the FSM state based on sensor data and elapsed time.
        
        Args:
            observation: Noisy sensor observations.
            dt: Time step in seconds.
        
        Returns:
            action: Highway-env action (0=LANE_LEFT, 1=IDLE, etc.)
        """
        pass

    @abstractmethod
    def get_available_actions(self) -> List[int]:
        """Return a list of available actions in the current state."""
        pass

    def _transition_to(self, new_state: Enum, reason: str = ""):
        """
        Transition to a new state and log the transition.
        
        Args:
            new_state: The new state (any Enum)
            reason: Reason for transition
        """
        if new_state != self.current_state:
            transition = {
                'timestamp': self.get_current_time(),
                'from_state': self.current_state.value,
                'to_state': new_state.value,
                'reason': reason,
                'time_in_previous_state': self.time_in_state
            }
            self.transition_history.append(transition)

            self.current_state = new_state
            self.time_in_state = 0.0

    def get_state_info(self) -> Dict:
        """Return information about current state."""
        return {
            'current_state': self.current_state.value,
            'time_in_state': self.time_in_state,
            'num_transitions': len(self.transition_history)
        }

    def get_current_time(self) -> float:
        """Return the current simulation time."""
        if not self.transition_history:
            return 0.0
        return sum(t['time_in_previous_state'] for t in self.transition_history)
    
    # Helper methods
    
    def _calculate_ttc(self, ego_velocity: float, vehicle: Dict) -> float:
        """
        Calculate Time-To-Collision (TTC) with another vehicle.
        
        Args:
            ego_velocity: Velocity of the ego vehicle (m/s)
            vehicle: Vehicle dictionary with 'relative_longitudinal' and 'velocity'
        
        Returns:
            TTC value in seconds. Returns inf if no collision is predicted.
        """
        rel_long = vehicle.get('relative_longitudinal', float('inf'))

        if rel_long <= 0:  # Vehicle behind us
            return float('inf')
        
        rel_velocity = ego_velocity - vehicle.get('velocity', 0.0)

        if rel_velocity <= 0:  # We are not faster
            return float('inf')
        
        ttc = rel_long / rel_velocity
        return ttc
    
    def _find_leader(self, observation: Dict) -> Optional[Dict]:
        """
        Find leading vehicle in the same lane.
        
        Args:
            observation: Sensor observation dictionary
            
        Returns:
            Vehicle dictionary of leader, or None if no leader
        """
        ego_lane = observation['ego']['lane_index']
        
        candidates = [
            v for v in observation['vehicles']
            if v.get('lane_index') == ego_lane
            and v['relative_longitudinal'] > 0  # ahead of us
        ]
        
        if not candidates:
            return None
        
        # Nearest vehicle in the same lane
        leader = min(candidates, key=lambda v: v['relative_longitudinal'])
        return leader
    
    def _measure_gap_on_lane(self, observation: Dict, target_lane: int) -> float:
        """
        Measure the smallest gap to vehicles on the target lane.
        
        Args:
            observation: Sensor observation
            target_lane: Lane index to check
        
        Returns:
            Smallest gap in meters. Returns inf if lane is clear.
        """
        vehicles_on_lane = [
            v for v in observation['vehicles']
            if v.get('lane_index') == target_lane
        ]
        
        if not vehicles_on_lane:
            return float('inf')  # Lane is completely clear
        
        # Find vehicles in front and behind
        front = [v for v in vehicles_on_lane if v['relative_longitudinal'] > 0]
        rear = [v for v in vehicles_on_lane if v['relative_longitudinal'] < 0]
        
        gap_front = min([v['relative_longitudinal'] for v in front], default=float('inf'))
        gap_rear = abs(max([v['relative_longitudinal'] for v in rear], default=float('-inf')))
        
        # Return the smallest gap (most critical)
        return min(gap_front, gap_rear)