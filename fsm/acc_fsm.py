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

        self.comfortable_distance_ratio = ads_config.tactical_params.get('comfortable_distance_ratio', 1.2)
        self.warning_distance_ratio = ads_config.tactical_params.get('warning_distance_ratio', 0.85)
        self.critical_ttc = ads_config.tactical_params.get('critical_ttc', 1.5)
        
        # Action smoothing - remember last action to reduce oscillation
        self.last_action = 1  # Start with IDLE
        self.action_hold_time = 0.0
        self.min_action_hold = 1.5  # Hold each action for at least 0.5 seconds
        
    def get_initial_state(self) -> Enum:
        """Initial state is CRUISING"""
        return ACCState.CRUISING
    
    def update(self, observation: Dict[str, Any], dt: float = 1.0) -> int:
        """
        Update FSM state and select action with correct time step.
        """
        self.time_in_state += dt
        self.action_hold_time += dt

        ego_velocity = observation['ego']['velocity']
        leader = self._find_leader(observation)

        # ====================================================================
        # PRIORITY 1: Emergency Check (IMPROVED - avoid false triggers)
        # ====================================================================
        if leader:
            ttc = self._calculate_ttc(ego_velocity, leader)
            distance = leader['relative_longitudinal']
            safe_distance = max(self.min_gap, self.time_headway * ego_velocity)

            # ✓ FIXED: More nuanced emergency detection
            
            # Level 1: CRITICAL - Both distance AND TTC are dangerous
            if distance < self.min_gap * 0.8 and ttc < self.critical_ttc:
                # distance < 20m AND TTC < 1.2s
                if self.current_state != ACCState.EMERGENCY_BRAKE:
                    self._transition_to(ACCState.EMERGENCY_BRAKE,
                                    f"CRITICAL: distance={distance:.1f}m AND TTC={ttc:.2f}s")
                self.last_action = 4
                self.action_hold_time = 0.0
                return 4  # Maximum braking

            # Level 2: EMERGENCY - TTC critical with concerning distance
            elif ttc < self.emergency_ttc and distance < safe_distance * 0.7:
                # TTC < 2.0s AND distance < 0.7 × safe_distance
                if self.current_state != ACCState.EMERGENCY_BRAKE:
                    self._transition_to(ACCState.EMERGENCY_BRAKE,
                                    f"Emergency: TTC={ttc:.2f}s, distance={distance:.1f}m")
                self.last_action = 4
                self.action_hold_time = 0.0
                return 4  # Emergency braking
            
            # Level 3: WARNING - Close distance but manageable TTC
            elif distance < self.min_gap and ttc > self.critical_ttc:
                # distance < 25m but TTC > 1.2s - handle in FOLLOWING state
                if self.current_state not in [ACCState.FOLLOWING, ACCState.EMERGENCY_BRAKE]:
                    self._transition_to(ACCState.FOLLOWING, 
                                    f"Close distance: {distance:.1f}m")
                # Let state-specific logic handle it

        # ====================================================================
        # PRIORITY 2: State-specific logic
        # ====================================================================
        
        if self.current_state == ACCState.CRUISING:
            action = self._handle_cruising_state(observation, ego_velocity, leader)
        elif self.current_state == ACCState.FOLLOWING:
            action = self._handle_following_state(observation, ego_velocity, leader)
        elif self.current_state == ACCState.EMERGENCY_BRAKE:
            action = self._handle_emergency_brake_state(observation, ego_velocity, leader)
        else:
            action = 1

        # Action smoothing (same as before)
        if action != self.last_action and self.action_hold_time < self.min_action_hold:
            action = self.last_action
        else:
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
            
            if distance < safe_distance * self.warning_distance_ratio:  # Added 10% buffer to reduce oscillation
                # Too close - transition to FOLLOWING
                self._transition_to(ACCState.FOLLOWING, 
                                  f"Leader detected at {distance:.1f}m < safe distance {safe_distance:.1f}m")
                return 4  # SLOWER (reduce speed to establish safe distance)
        
        # No leader or safe distance maintained - cruise control
        velocity_error = self.target_velocity - ego_velocity
        
        # NEW: Proportional control with dead zone to reduce oscillation
        if velocity_error > 1.0:  # Significantly slower than target
            return 3  # FASTER
        elif velocity_error < -1.0:  # Significantly faster than target
            return 4  # SLOWER
        elif abs(velocity_error) > 0.5:  # Mild adjustment needed
            # Gentle correction based on direction
            return 3 if velocity_error > 0 else 4
        else:
            # Within acceptable range - maintain current speed
            return 1  # IDLE
    
    def _handle_following_state(self, observation: Dict, ego_velocity: float, leader: Optional[Dict]) -> int:
        """Handle FOLLOWING state logic with improved distance control and EXIT condition"""
        
        if not leader:
            # Leader disappeared (lane change, exit, etc.)
            self._transition_to(ACCState.CRUISING, "Leader no longer detected")
            return 3  # FASTER (accelerate back to target velocity)
        
        # Leader still present - maintain safe distance with smooth control
        distance = leader['relative_longitudinal']
        safe_distance = max(self.min_gap, self.time_headway * ego_velocity)
        leader_velocity = leader.get('velocity', ego_velocity)
        
        # Calculate distance error and velocity difference
        velocity_diff = ego_velocity - leader_velocity
        distance_error = distance - safe_distance
        
        # EXIT
        if distance > safe_distance * 1.5:
            # Too far - consider exiting FOLLOWING
            self._transition_to(ACCState.CRUISING, 
                              f"Safe distance exceeded: {distance:.1f}m > 1.5 × {safe_distance:.1f}m")
            return 3  # FASTER (accelerate to target)
        
        # Zone 1: Too close (< 0.7 * safe_distance)
        if distance < safe_distance * 0.7:
            return 4  # SLOWER
        # Zone 2: Slightly too close (0.7 - 0.9 * safe_distance)
        elif distance < safe_distance * 0.9:
            # Only break if we're significantly faster than leader
            if velocity_diff > 2.0:
                return 4  # SLOWER
            else:
                return 1  # IDLE (gap will increase naturally)
        # Zone 3: Ideal range (0.9 - 1.2 × safe_distance)
        elif distance < safe_distance * self.comfortable_distance_ratio:
            # Maintain by matching velocity
        # Priority: Match leader velocity
            if abs(velocity_diff) < 0.5:  # ✓ Within 0.5 m/s = 1.8 km/h
                # Excellent velocity matching
                return 1  # IDLE (maintain current state)
            
            elif velocity_diff > 1.5:  # We're faster by >1.5 m/s
                return 4  # SLOWER (reduce speed to match)
            
            elif velocity_diff > 0.5:  # We're slightly faster (0.5-1.5 m/s)
                # Close to matching, gentle correction
                return 4  # SLOWER (gentle deceleration)
            
            elif velocity_diff < -1.5:  # We're slower by >1.5 m/s
                return 3  # FASTER (increase speed to match)
            
            else:  # velocity_diff in (-1.5, -0.5) - we're slightly slower
                # Close to matching, gentle correction
                return 3  # FASTER (gentle acceleration)
        # Zone 4: Slightly far (1.2 - 1.5 × safe_distance)
        elif distance < safe_distance * 1.5:
            # Gently close gap
            if velocity_diff < -1.0:  # Leader is faster
                return 3  # FASTER
            else:
                return 1  # IDLE (gap will close naturally)
        # Zone 5: Too far (> 1.5 × safe_distance)
        else:
            # Accelerate to close gap
            if velocity_diff < -1.5:  # Leader much faster, we can speed up safely
                return 3  # FASTER
            else:
                return 1  # IDLE (gap will close gradually)
    
    def _handle_emergency_brake_state(self, observation: Dict, ego_velocity: float, leader: Optional[Dict]) -> int:
        """Handle EMERGENCY_BRAKE state logic with progressive recovery"""
        
        if not leader:
            # Leader disappeared - safe to return to cruising
            self._transition_to(ACCState.CRUISING, "Emergency resolved - leader gone")
            return 3  # FASTER (accelerate back to target)
        
        # Get current metrics
        ttc = self._calculate_ttc(ego_velocity, leader)
        distance = leader['relative_longitudinal']
        safe_distance = max(self.min_gap, self.time_headway * ego_velocity)
        velocity_diff = ego_velocity - leader.get('velocity', ego_velocity)
        
        # ====================================================================
        # Recovery Logic - Progressive exit from emergency state
        # ====================================================================
        
        # Level 1: Situation significantly improved - exit to FOLLOWING
        if distance > self.min_gap * 1.2 and ttc > self.critical_ttc * 1.5:
            # Distance > 30m AND TTC > 1.8s
            self._transition_to(ACCState.FOLLOWING, 
                            f"Emergency resolved - distance={distance:.1f}m, TTC={ttc:.2f}s")
            return 1  # IDLE (let it stabilize)
        
        # Level 2: TTC acceptable but distance still close
        elif ttc > self.emergency_ttc:  # TTC > 2.0s
            # TTC is OK, transition to FOLLOWING for normal control
            self._transition_to(ACCState.FOLLOWING, 
                            f"TTC improved to {ttc:.2f}s, switching to FOLLOWING")
            
            # Decide action based on distance
            if distance < safe_distance * 0.8:
                return 4  # SLOWER (still need to brake gently)
            else:
                return 1  # IDLE
        
        # Level 3: Very slow speed - avoid full stop
        elif ego_velocity < 15.0:
            # We're already very slow (< 54 km/h)
            
            if distance > self.min_gap * 0.9:  # distance > 22.5m
                # Distance is acceptable for current speed - ease off
                if velocity_diff > 1.0:
                    return 4  # SLOWER (gentle braking)
                else:
                    return 1  # IDLE (coast)
            else:
                # Still too close - continue gentle braking
                return 4  # SLOWER
        
        # Level 4: Moderate speed with improving situation
        elif ttc > self.critical_ttc:  # TTC > 1.2s
            # Not critical anymore, but still emergency
            if distance > self.min_gap * 0.85:  # distance > 21.25m
                # Transition to FOLLOWING for better control
                self._transition_to(ACCState.FOLLOWING, 
                                f"Situation improving - TTC={ttc:.2f}s, dist={distance:.1f}m")
                return 4  # SLOWER
            else:
                # Continue emergency braking
                return 4  # SLOWER
        
        # Still in critical situation - full braking
        return 4  # SLOWER (maximum braking)

    def get_available_actions(self) -> List[int]:
        """
        ACC-Only can only perform longitudinal control.
        
        Returns:
            List of action codes: [IDLE, FASTER, SLOWER]
        """
        return [1, 3, 4]  # IDLE, FASTER, SLOWER
        