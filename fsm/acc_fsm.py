from enum import Enum
from typing import Dict, List, Any, Optional
from src.fsm.base_fsm import BaseFSM
from src.config.config_manager import ADSConfig


class ACCState(Enum):
    """States for ACC-Only System"""
    CRUISING = "CRUISING"
    FOLLOWING = "FOLLOWING"
    EMERGENCY_BRAKE = "EMERGENCY_BRAKE"


class ACCFSM(BaseFSM):
    """
    Improved FSM for ACC-Only System (SAE Level 1)
    
    States:
        - CRUISING: Driving at target velocity, no vehicle ahead
        - FOLLOWING: Following a lead vehicle with safe distance
        - EMERGENCY_BRAKE: Critical situation, emergency braking
    
    Actions:
        - Only longitudinal control (FASTER/SLOWER/IDLE)
        - No lane changes
    """
    
    def __init__(self, ads_config: ADSConfig):
        super().__init__(ads_config)
        
        # Extract tactical parameters
        self.target_velocity = ads_config.tactical_params['target_velocity']
        self.time_headway = ads_config.tactical_params['time_headway']
        self.emergency_ttc = ads_config.tactical_params['emergency_ttc']
        self.min_gap = ads_config.tactical_params['min_gap']
        
        # NEW: Action smoothing - remember last action to reduce oscillation
        self.last_action = 1  # Start with IDLE
        self.action_hold_time = 0.0
        self.min_action_hold = 0.5  # Hold each action for at least 0.5 seconds
        
    def get_initial_state(self) -> Enum:
        """Initial state is CRUISING"""
        return ACCState.CRUISING
    
    def update(self, observation: Dict[str, Any], dt: float = 0.1) -> int:
        """
        Update FSM state and select action with improved smoothness.
        
        Args:
            observation: Noisy sensor observation
            dt: Time step (default 0.1s)
            
        Returns:
            action: Highway-env action code
                1 = IDLE (maintain speed)
                3 = FASTER (accelerate)
                4 = SLOWER (decelerate/brake)
        """
        self.time_in_state += dt
        self.action_hold_time += dt
        
        ego_velocity = observation['ego']['velocity']
        leader = self._find_leader(observation)
        
        # ====================================================================
        # PRIORITY 1: Emergency Check (overrides all other states)
        # ====================================================================
        if leader:
            ttc = self._calculate_ttc(ego_velocity, leader)
            if ttc < self.emergency_ttc:
                self._transition_to(ACCState.EMERGENCY_BRAKE, f"TTC={ttc:.2f}s < {self.emergency_ttc}s")
                self.last_action = 4
                self.action_hold_time = 0.0
                return 4  # SLOWER (emergency braking)
        
        # ====================================================================
        # PRIORITY 2: State-specific logic with improved smoothness
        # ====================================================================
        
        if self.current_state == ACCState.CRUISING:
            action = self._handle_cruising_state(observation, ego_velocity, leader)
        elif self.current_state == ACCState.FOLLOWING:
            action = self._handle_following_state(observation, ego_velocity, leader)
        elif self.current_state == ACCState.EMERGENCY_BRAKE:
            action = self._handle_emergency_brake_state(observation, ego_velocity, leader)
        else:
            action = 1  # Default fallback
        
        # NEW: Action smoothing - avoid rapid switching
        if action != self.last_action and self.action_hold_time < self.min_action_hold:
            # Keep previous action if we haven't held it long enough
            action = self.last_action
        else:
            # Update action history
            if action != self.last_action:
                self.action_hold_time = 0.0
            self.last_action = action
        
        return action
    
    def _handle_cruising_state(self, observation: Dict, ego_velocity: float, leader: Optional[Dict]) -> int:
        """Handle CRUISING state logic with improved smoothness"""
        
        if leader:
            # Vehicle detected ahead
            distance = leader['relative_longitudinal']
            safe_distance = max(self.min_gap, self.time_headway * ego_velocity)
            
            if distance < safe_distance * 0.9:  # Added 10% buffer to reduce oscillation
                # Too close - transition to FOLLOWING
                self._transition_to(ACCState.FOLLOWING, 
                                  f"Leader detected at {distance:.1f}m < safe distance {safe_distance:.1f}m")
                return 4  # SLOWER (reduce speed to establish safe distance)
        
        # No leader or safe distance maintained - cruise control
        velocity_error = self.target_velocity - ego_velocity
        
        # NEW: Proportional control with dead zone to reduce oscillation
        if velocity_error > 2.0:  # Significantly slower than target
            return 3  # FASTER
        elif velocity_error < -2.0:  # Significantly faster than target
            return 4  # SLOWER
        elif abs(velocity_error) > 1.0:  # Mild adjustment needed
            # Gentle correction based on direction
            return 3 if velocity_error > 0 else 4
        else:
            # Within acceptable range - maintain current speed
            return 1  # IDLE
    
    def _handle_following_state(self, observation: Dict, ego_velocity: float, leader: Optional[Dict]) -> int:
        """Handle FOLLOWING state logic with improved distance control"""
        
        if not leader:
            # Leader disappeared (lane change, exit, etc.)
            self._transition_to(ACCState.CRUISING, "Leader no longer detected")
            return 3  # FASTER (accelerate back to target velocity)
        
        # Leader still present - maintain safe distance with smooth control
        distance = leader['relative_longitudinal']
        safe_distance = max(self.min_gap, self.time_headway * ego_velocity)
        leader_velocity = leader.get('velocity', ego_velocity)
        
        # Calculate distance error and velocity difference
        distance_error = distance - safe_distance
        velocity_diff = ego_velocity - leader_velocity
        
        # NEW: Combined distance and velocity control (predictive)
        # If we're closing in fast, brake earlier
        predicted_distance = distance - velocity_diff * 2.0  # Look ahead 2 seconds
        
        if predicted_distance < safe_distance * 0.7:
            # Predicted to get too close - brake now
            return 4  # SLOWER
        elif distance < safe_distance * 0.8:
            # Currently too close - reduce speed
            return 4  # SLOWER
        elif distance > safe_distance * 1.3:
            # Gap is too large - close up gently
            if velocity_diff < -1.5:  # Leader is much faster, we can speed up safely
                return 3  # FASTER
            else:
                return 1  # IDLE (gap will close naturally)
        elif distance > safe_distance * 1.15:
            # Gap slightly large but acceptable
            if velocity_diff < 0:  # We're slower than leader
                return 3  # FASTER
            else:
                return 1  # IDLE
        else:
            # Distance is acceptable - match leader's velocity
            if velocity_diff > 2.0:  # We are significantly faster
                return 4  # SLOWER
            elif velocity_diff < -2.0:  # We are significantly slower
                return 3  # FASTER
            else:
                # Well matched - maintain
                return 1  # IDLE
    
    def _handle_emergency_brake_state(self, observation: Dict, ego_velocity: float, leader: Optional[Dict]) -> int:
        """Handle EMERGENCY_BRAKE state logic"""
        
        if not leader:
            # Leader disappeared - return to CRUISING
            self._transition_to(ACCState.CRUISING, "Emergency resolved - leader gone")
            return 1  # IDLE
        
        # Check if emergency is resolved
        ttc = self._calculate_ttc(ego_velocity, leader)
        distance = leader['relative_longitudinal']
        safe_distance = max(self.min_gap, self.time_headway * ego_velocity)
        
        # Exit emergency state when both TTC and distance are safe
        if ttc > self.emergency_ttc * 2.5 and distance > safe_distance * 1.2:
            # Situation improved significantly - back to FOLLOWING
            self._transition_to(ACCState.FOLLOWING, 
                              f"Emergency resolved - TTC={ttc:.2f}s, distance={distance:.1f}m")
            return 1  # IDLE
        elif ttc > self.emergency_ttc * 1.5:
            # Improving but still cautious - gentle deceleration
            self._transition_to(ACCState.FOLLOWING, f"Emergency improving - TTC={ttc:.2f}s")
            return 4  # SLOWER (gentle)
        
        # Still critical - continue braking
        return 4  # SLOWER
    
    def get_available_actions(self) -> List[int]:
        """
        ACC-Only can only perform longitudinal control.
        
        Returns:
            List of action codes: [IDLE, FASTER, SLOWER]
        """
        return [1, 3, 4]  # IDLE, FASTER, SLOWER
        