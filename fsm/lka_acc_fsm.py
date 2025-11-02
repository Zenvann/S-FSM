# src/fsm/lka_acc_fsm.py
from enum import Enum
from typing import Dict, Any, Optional, List
from src.fsm.base_fsm import BaseFSM
from src.config.config_manager import ADSConfig


class LKAACCState(Enum):
    """States for LKA + ACC system (SAE Level 2)"""
    CRUISING = "CRUISING"
    FOLLOWING = "FOLLOWING"
    LANE_KEEPING = "LANE_KEEPING"          # 主动车道保持
    LANE_CORRECTION = "LANE_CORRECTION"    # 纠偏
    EMERGENCY_BRAKE = "EMERGENCY_BRAKE"


class LKAACCFSM(BaseFSM):
    """
    FSM for Lane Keeping Assist + ACC (SAE Level 2)
    """
    
    def __init__(self, ads_config: ADSConfig):
        super().__init__(ads_config)
        
        # ACC parameters (inherited behavior)
        self.target_velocity = ads_config.tactical_params['target_velocity']
        self.time_headway = ads_config.tactical_params['time_headway']
        self.emergency_ttc = ads_config.tactical_params['emergency_ttc']
        self.min_gap = ads_config.tactical_params['min_gap']
        
        # LKA parameters 
        self.lane_center_tolerance = ads_config.tactical_params.get('lane_center_tolerance', 0.3)
        self.max_lateral_acceleration = ads_config.tactical_params.get('max_lateral_acceleration', 2.0)
        self.min_lane_confidence = ads_config.tactical_params.get('min_lane_confidence', 0.6)
        
    def get_initial_state(self) -> Enum:
        return LKAACCState.CRUISING
    
    def update(self, observation: Dict[str, Any], dt: float = 0.1) -> int:
        """
        Update FSM with lateral control consideration.
        
        Returns:
            action: Highway-env action code
                1 = IDLE
                3 = FASTER
                4 = SLOWER
                (NO lane changes: 0, 2 not used)
        """
        self.time_in_state += dt
        
        ego_velocity = observation['ego']['velocity']
        leader = self._find_leader(observation)
        
        # ====================================================================
        # PRIORITY 1: Emergency Check
        # ====================================================================
        if leader:
            ttc = self._calculate_ttc(ego_velocity, leader)
            if ttc < self.emergency_ttc:
                self._transition_to(LKAACCState.EMERGENCY_BRAKE, 
                                  f"TTC={ttc:.2f}s < {self.emergency_ttc}s")
                return 4  # SLOWER
        
        # ====================================================================
        # PRIORITY 2: Lane Position Check (NEW for LKA)
        # ====================================================================
        lane_offset = self._get_lane_offset(observation)
        
        if abs(lane_offset) > self.lane_center_tolerance:
            # Vehicle is drifting - need correction
            if self.current_state != LKAACCState.LANE_CORRECTION:
                self._transition_to(LKAACCState.LANE_CORRECTION,
                                  f"Lane offset={lane_offset:.2f}m")
        
        # ====================================================================
        # PRIORITY 3: Longitudinal Control (Same as ACC)
        # ====================================================================
        
        if self.current_state == LKAACCState.CRUISING:
            return self._handle_cruising_state(observation, ego_velocity, leader)
        
        elif self.current_state == LKAACCState.FOLLOWING:
            return self._handle_following_state(observation, ego_velocity, leader)
        
        elif self.current_state == LKAACCState.LANE_CORRECTION:
            return self._handle_lane_correction_state(observation, ego_velocity, leader, lane_offset)
        
        elif self.current_state == LKAACCState.EMERGENCY_BRAKE:
            return self._handle_emergency_brake_state(observation, ego_velocity, leader)
        
        return 1  # IDLE
    
    def _handle_cruising_state(self, observation: Dict, ego_velocity: float, leader: Optional[Dict]) -> int:
        """Same logic as ACC, but can transition to LANE_CORRECTION"""
        
        if leader:
            distance = leader['relative_longitudinal']
            safe_distance = self.time_headway * ego_velocity
            
            if distance < safe_distance:
                self._transition_to(LKAACCState.FOLLOWING, 
                                  f"Leader at {distance:.1f}m")
                return 4  # SLOWER
        
        # Speed control
        if ego_velocity < self.target_velocity - 0.5:
            return 3  # FASTER
        elif ego_velocity > self.target_velocity + 0.5:
            return 4  # SLOWER
        else:
            return 1  # IDLE
    
    def _handle_following_state(self, observation: Dict, ego_velocity: float, leader: Optional[Dict]) -> int:
        """Same as ACC - adaptive cruise control"""
        
        if not leader:
            self._transition_to(LKAACCState.CRUISING, "Leader disappeared")
            return 3  # FASTER
        
        distance = leader['relative_longitudinal']
        safe_distance = self.time_headway * ego_velocity
        
        if distance > safe_distance * 1.2:
            return 3  # FASTER
        elif distance < safe_distance * 0.8:
            return 4  # SLOWER
        else:
            velocity_diff = ego_velocity - leader.get('velocity', ego_velocity)
            if velocity_diff > 1.0:
                return 4  # SLOWER
            elif velocity_diff < -1.0:
                return 3  # FASTER
            else:
                return 1  # IDLE
    
    def _handle_lane_correction_state(self, observation: Dict, ego_velocity: float, 
                                     leader: Optional[Dict], lane_offset: float) -> int:
        """
        Handle lane correction.
        
        Note: Highway-env doesn't have explicit lateral control actions.
        In real systems, this would apply steering torque.
        Here, we simulate by slightly reducing speed during correction.
        """
        
        # Check if correction is complete
        if abs(lane_offset) < self.lane_center_tolerance * 0.5:
            # Back to normal operation
            if leader:
                self._transition_to(LKAACCState.FOLLOWING, "Lane correction complete, leader present")
            else:
                self._transition_to(LKAACCState.CRUISING, "Lane correction complete")
            return 1  # IDLE
        
        # During correction, slightly reduce speed for safety
        # (In real system, this would be pure lateral control)
        if ego_velocity > self.target_velocity * 0.9:
            return 4  # SLOWER (gentle reduction)
        else:
            return 1  # IDLE
    
    def _handle_emergency_brake_state(self, observation: Dict, ego_velocity: float, leader: Optional[Dict]) -> int:
        """Same as ACC"""
        
        if not leader:
            self._transition_to(LKAACCState.CRUISING, "Emergency resolved - leader gone")
            return 1
        
        # Check if emergency is resolved
        ttc = self._calculate_ttc(ego_velocity, leader)
        distance = leader['relative_longitudinal']
        safe_distance = self.time_headway * ego_velocity
        
        if ttc > self.emergency_ttc * 1.5 and distance > safe_distance * 0.7:
            self._transition_to(LKAACCState.FOLLOWING, 
                              f"Emergency resolved - TTC={ttc:.2f}s, distance={distance:.1f}m")
            return 1
        
        if ego_velocity < 20.0:
            return 1  # IDLE (avoid full stop)
        
        return 4  # Continue braking
    
    def _get_lane_offset(self, observation: Dict) -> float:
        """
        Calculate lateral offset from lane center.
        
        Returns:
            Offset in meters (negative = right, positive = left)
        """
        ego_pos_y = observation['ego']['position'][1]
        lane_index = observation['ego']['lane_index']
        
        # Lane center Y position (assuming 4m lane width)
        lane_center_y = lane_index * 4.0
        
        offset = ego_pos_y - lane_center_y
        return offset
    
    def get_available_actions(self) -> List[int]:
        """
        LKA+ACC only controls speed (no lane changes).
        
        Returns:
            [IDLE, FASTER, SLOWER]
        """
        return [1, 3, 4]