# src/fsm/highway_pilot_fsm.py
from enum import Enum
from typing import Dict, Any, Optional, List
from src.fsm.lka_acc_fsm import LKAACCFSM, LKAACCState
from src.config.config_manager import ADSConfig


class HighwayPilotState(Enum):
    """States for Highway Pilot (SAE Level 2+)"""
    CRUISING = "CRUISING"
    FOLLOWING = "FOLLOWING"
    LANE_KEEPING = "LANE_KEEPING"
    LANE_CORRECTION = "LANE_CORRECTION"
    EMERGENCY_BRAKE = "EMERGENCY_BRAKE"
    
    # New states for lane change capability
    EVALUATING_OVERTAKE = "EVALUATING_OVERTAKE"      # 评估是否需要超车
    PREPARING_LANE_CHANGE_LEFT = "PREPARING_LC_LEFT"  # 准备左变道
    PREPARING_LANE_CHANGE_RIGHT = "PREPARING_LC_RIGHT"
    EXECUTING_LANE_CHANGE_LEFT = "EXECUTING_LC_LEFT"  # 执行变道
    EXECUTING_LANE_CHANGE_RIGHT = "EXECUTING_LC_RIGHT"
    ABORTING_LANE_CHANGE = "ABORTING_LANE_CHANGE"    # 中止变道


class HighwayPilotFSM(LKAACCFSM):
    """
    FSM for Highway Pilot (SAE Level 2+)
    
    Inherits from LKA+ACC and adds:
    - Automated lane change
    - Overtaking slower vehicles
    - Return to right lane after overtaking
    """
    
    def __init__(self, ads_config: ADSConfig):
        super().__init__(ads_config)
        
        # Lane change parameters (new)
        self.min_gap_for_lane_change = ads_config.tactical_params.get('min_gap_for_lane_change', 40.0)
        self.safe_gap_front = ads_config.tactical_params.get('safe_gap_front', 25.0)
        self.safe_gap_rear = ads_config.tactical_params.get('safe_gap_rear', 20.0)
        self.lane_change_duration = ads_config.tactical_params.get('lane_change_duration', 4.0)
        
        # Overtaking logic parameters
        self.velocity_delta_threshold = ads_config.tactical_params.get('velocity_delta_threshold', -5.0)
        self.min_following_time_before_overtake = ads_config.tactical_params.get('min_following_time_before_overtake', 10.0)
        
        # Safety parameters
        self.abort_if_ttc_during_lc = ads_config.tactical_params.get('abort_if_ttc_during_lc', 3.0)
        self.abort_if_gap_shrinks_by = ads_config.tactical_params.get('abort_if_gap_shrinks_by', 10.0)
        
        # Internal state tracking
        self.lane_change_start_time = 0.0
        self.initial_gap_at_lc_start = None
    
    def get_initial_state(self) -> Enum:
        return HighwayPilotState.CRUISING
    
    def update(self, observation: Dict[str, Any], dt: float = 0.1) -> int:
        """
        Enhanced update with lane change capability.
        
        Returns:
            action: Highway-env action code
                0 = LANE_LEFT
                1 = IDLE
                2 = LANE_RIGHT
                3 = FASTER
                4 = SLOWER
        """
        self.time_in_state += dt
        
        ego_velocity = observation['ego']['velocity']
        ego_lane = observation['ego']['lane_index']
        leader = self._find_leader(observation)

        # debugging logs
        step_count = int(self.time_in_state / dt)

        if self.current_state == HighwayPilotState.FOLLOWING and leader:
            distance = leader['relative_longitudinal']
            velocity_diff = ego_velocity - leader.get('velocity', ego_velocity)
            
            # output every 1 second
            if step_count % 10 == 0:
                print(f"  [FOLLOWING] time={self.time_in_state:.1f}s, "
                    f"vel_diff={velocity_diff:.2f} m/s, "
                    f"dist={distance:.1f}m, "
                    f"ego_vel={ego_velocity:.1f} m/s, "
                    f"leader_vel={leader.get('velocity', 0):.1f} m/s, "
                    f"threshold={self.velocity_delta_threshold}")
        
        # overtake logs
        if self.current_state == HighwayPilotState.EVALUATING_OVERTAKE:
            print(f"  [EVALUATING] Checking lanes for overtake...")
            if ego_lane > 0:
                left_gap = self._measure_gap_on_lane(observation, ego_lane - 1)
                print(f"    Left gap: {left_gap:.1f}m (need {self.min_gap_for_lane_change}m)")
            if ego_lane < 3:
                right_gap = self._measure_gap_on_lane(observation, ego_lane + 1)
                print(f"    Right gap: {right_gap:.1f}m")
        
        # lane change logs
        if self.current_state in [HighwayPilotState.EXECUTING_LANE_CHANGE_LEFT,
                                HighwayPilotState.EXECUTING_LANE_CHANGE_RIGHT]:
            print(f"  [LANE_CHANGE] Executing... time={self.time_in_state:.2f}s, lane={ego_lane}")        
        
        # ====================================================================
        # PRIORITY 1: Emergency Check (highest priority)
        # ====================================================================
        if leader:
            ttc = self._calculate_ttc(ego_velocity, leader)
            if ttc < self.emergency_ttc:
                self._transition_to(HighwayPilotState.EMERGENCY_BRAKE, 
                                  f"Emergency: TTC={ttc:.2f}s")
                return 4  # SLOWER
        
        # ====================================================================
        # PRIORITY 2: Lane Change Abort Check (during lane change)
        # ====================================================================
        if self._is_in_lane_change_state():
            should_abort, reason = self._check_lane_change_abort_conditions(observation, ego_velocity, leader)
            if should_abort:
                self._transition_to(HighwayPilotState.ABORTING_LANE_CHANGE, reason)
                return 4  # SLOWER (abort and decelerate)
        
        # ====================================================================
        # PRIORITY 3: State-specific behavior
        # ====================================================================
        
        if self.current_state == HighwayPilotState.CRUISING:
            return self._handle_cruising_state_hp(observation, ego_velocity, leader)
        
        elif self.current_state == HighwayPilotState.FOLLOWING:
            return self._handle_following_state_hp(observation, ego_velocity, leader)
        
        elif self.current_state == HighwayPilotState.EVALUATING_OVERTAKE:
            return self._handle_evaluating_overtake_state(observation, ego_velocity, leader, ego_lane)
        
        elif self.current_state == HighwayPilotState.PREPARING_LANE_CHANGE_LEFT:
            return self._handle_preparing_lane_change_left(observation, ego_velocity, ego_lane)
        
        elif self.current_state == HighwayPilotState.PREPARING_LANE_CHANGE_RIGHT:
            return self._handle_preparing_lane_change_right(observation, ego_velocity, ego_lane)
        
        elif self.current_state == HighwayPilotState.EXECUTING_LANE_CHANGE_LEFT:
            return self._handle_executing_lane_change_left(observation, ego_lane)
        
        elif self.current_state == HighwayPilotState.EXECUTING_LANE_CHANGE_RIGHT:
            return self._handle_executing_lane_change_right(observation, ego_lane)
        
        elif self.current_state == HighwayPilotState.ABORTING_LANE_CHANGE:
            return self._handle_aborting_lane_change(observation, ego_velocity, leader)
        
        elif self.current_state == HighwayPilotState.LANE_CORRECTION:
            return super()._handle_lane_correction_state(observation, ego_velocity, leader, 
                                                        self._get_lane_offset(observation))
        
        elif self.current_state == HighwayPilotState.EMERGENCY_BRAKE:
            return super()._handle_emergency_brake_state(observation, ego_velocity, leader)
        
        return 1  # IDLE
    
    def _handle_cruising_state_hp(self, observation: Dict, ego_velocity: float, leader: Optional[Dict]) -> int:
        """Enhanced cruising with lane change consideration"""
        
        if leader:
            distance = leader['relative_longitudinal']
            safe_distance = self.time_headway * ego_velocity
            
            if distance < safe_distance:
                self._transition_to(HighwayPilotState.FOLLOWING, 
                                  f"Leader detected at {distance:.1f}m")
                return 4  # SLOWER
        
        # Speed control (same as parent)
        if ego_velocity < self.target_velocity - 0.5:
            return 3  # FASTER
        elif ego_velocity > self.target_velocity + 0.5:
            return 4  # SLOWER
        else:
            return 1  # IDLE
    
    def _handle_following_state_hp(self, observation: Dict, ego_velocity: float, leader: Optional[Dict]) -> int:
        """Enhanced following with overtake evaluation"""
        
        if not leader:
            self._transition_to(HighwayPilotState.CRUISING, "Leader disappeared")
            return 3  # FASTER
        
        distance = leader['relative_longitudinal']
        safe_distance = self.time_headway * ego_velocity
        
        # Check if we should consider overtaking
        velocity_diff = ego_velocity - leader.get('velocity', ego_velocity)
        
        if (self.time_in_state > self.min_following_time_before_overtake and
            velocity_diff < self.velocity_delta_threshold and  # Leader is significantly slower
            distance > self.min_gap * 0.8):  # Maintain safe distance
            
            self._transition_to(HighwayPilotState.EVALUATING_OVERTAKE,
                              f"Leader slower by {-velocity_diff:.1f} m/s for {self.time_in_state:.1f}s")
            return 1  # IDLE (evaluate before action)
        
        # Normal following behavior (same as parent)
        if distance > safe_distance * 1.1:
            return 3  # FASTER
        elif distance < safe_distance * 0.9:
            return 4  # SLOWER
        else:
            if velocity_diff > 1.0:
                return 4  # SLOWER
            elif velocity_diff < -1.0:
                return 3  # FASTER
            else:
                return 1  # IDLE
    
    def _handle_evaluating_overtake_state(self, observation: Dict, ego_velocity: float, 
                                         leader: Optional[Dict], ego_lane: int) -> int:
        """Evaluate if overtaking is safe and beneficial"""
        
        if not leader:
            self._transition_to(HighwayPilotState.CRUISING, "Leader disappeared during evaluation")
            return 3  # FASTER
        
        # Check left lane (preferred for overtaking)
        if ego_lane > 0:  # Can go left
            left_gap = self._measure_gap_on_lane(observation, ego_lane - 1)
            
            if left_gap > self.min_gap_for_lane_change:
                self._transition_to(HighwayPilotState.PREPARING_LANE_CHANGE_LEFT,
                                  f"Left lane clear, gap={left_gap:.1f}m")
                return 1  # IDLE (prepare for lane change)
        
        # Check right lane as alternative
        if ego_lane < 3:  # Can go right (assuming 4 lanes)
            right_gap = self._measure_gap_on_lane(observation, ego_lane + 1)
            
            if right_gap > self.min_gap_for_lane_change:
                self._transition_to(HighwayPilotState.PREPARING_LANE_CHANGE_RIGHT,
                                  f"Right lane clear, gap={right_gap:.1f}m")
                return 1  # IDLE
        
        # No safe gap available - back to following
        self._transition_to(HighwayPilotState.FOLLOWING, "No safe gap for overtaking")
        return 4  # SLOWER (maintain safe distance)
    
    def _handle_preparing_lane_change_left(self, observation: Dict, ego_velocity: float, ego_lane: int) -> int:
        """Final safety check before left lane change"""
        
        left_gap = self._measure_gap_on_lane(observation, ego_lane - 1)
        
        if left_gap < self.safe_gap_front:
            # Gap closed - abort
            self._transition_to(HighwayPilotState.FOLLOWING, 
                              f"Left gap closed: {left_gap:.1f}m < {self.safe_gap_front:.1f}m")
            return 4  # SLOWER
        
        # All clear - execute lane change
        self.lane_change_start_time = self.get_current_time()
        self.initial_gap_at_lc_start = left_gap
        self._transition_to(HighwayPilotState.EXECUTING_LANE_CHANGE_LEFT, "Executing left lane change")
        
        return 0  # LANE_LEFT
    
    def _handle_preparing_lane_change_right(self, observation: Dict, ego_velocity: float, ego_lane: int) -> int:
        """Final safety check before right lane change"""
        
        right_gap = self._measure_gap_on_lane(observation, ego_lane + 1)
        
        if right_gap < self.safe_gap_front:
            self._transition_to(HighwayPilotState.FOLLOWING, 
                              f"Right gap closed: {right_gap:.1f}m")
            return 4  # SLOWER
        
        self.lane_change_start_time = self.get_current_time()
        self.initial_gap_at_lc_start = right_gap
        self._transition_to(HighwayPilotState.EXECUTING_LANE_CHANGE_RIGHT, "Executing right lane change")
        
        return 2  # LANE_RIGHT
    
    def _handle_executing_lane_change_left(self, observation: Dict, ego_lane: int) -> int:
        """Monitor lane change execution"""
        
        # Check if lane change completed
        expected_lane = ego_lane  # After lane change command, this should update next step
        
        if self.time_in_state > 0.5:  # Give time for lane change to register
            # Lane change should be complete
            self._transition_to(HighwayPilotState.CRUISING, "Lane change left completed")
            return 3  # FASTER (accelerate after lane change)
        
        return 1  # IDLE (maintain speed during lane change)
    
    def _handle_executing_lane_change_right(self, observation: Dict, ego_lane: int) -> int:
        """Monitor lane change execution"""
        
        if self.time_in_state > 0.5:
            self._transition_to(HighwayPilotState.CRUISING, "Lane change right completed")
            return 3  # FASTER
        
        return 1  # IDLE
    
    def _handle_aborting_lane_change(self, observation: Dict, ego_velocity: float, leader: Optional[Dict]) -> int:
        """Handle aborted lane change"""
        
        # Return to original lane (Highway-env might handle this automatically)
        # Slow down and return to FOLLOWING
        
        if self.time_in_state > 1.0:  # Abort maneuver duration
            self._transition_to(HighwayPilotState.FOLLOWING, "Lane change aborted, resuming following")
            return 1  # IDLE
        
        return 4  # SLOWER (decelerate during abort)
    
    def _check_lane_change_abort_conditions(self, observation: Dict, ego_velocity: float, 
                                           leader: Optional[Dict]) -> tuple[bool, str]:
        """
        Check if lane change should be aborted.
        
        Returns:
            (should_abort, reason)
        """
        # Check TTC during lane change
        if leader:
            ttc = self._calculate_ttc(ego_velocity, leader)
            if ttc < self.abort_if_ttc_during_lc:
                return (True, f"Abort: TTC={ttc:.2f}s during lane change")
        
        # Check if gap is shrinking dangerously
        if self.initial_gap_at_lc_start is not None:
            ego_lane = observation['ego']['lane_index']
            
            # Determine target lane
            if self.current_state == HighwayPilotState.EXECUTING_LANE_CHANGE_LEFT:
                target_lane = ego_lane - 1
            elif self.current_state == HighwayPilotState.EXECUTING_LANE_CHANGE_RIGHT:
                target_lane = ego_lane + 1
            else:
                return (False, "")
            
            current_gap = self._measure_gap_on_lane(observation, target_lane)
            gap_reduction = self.initial_gap_at_lc_start - current_gap
            
            if gap_reduction > self.abort_if_gap_shrinks_by:
                return (True, f"Abort: Gap shrunk by {gap_reduction:.1f}m")
        
        return (False, "")
    
    def _is_in_lane_change_state(self) -> bool:
        """Check if currently in a lane change state"""
        return self.current_state in [
            HighwayPilotState.EXECUTING_LANE_CHANGE_LEFT,
            HighwayPilotState.EXECUTING_LANE_CHANGE_RIGHT
        ]
    
    def get_available_actions(self) -> List[int]:
        """
        Highway Pilot has full action space.
        
        Returns:
            [LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER]
        """
        return [0, 1, 2, 3, 4]