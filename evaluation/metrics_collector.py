from typing import Dict, List, Any, Optional
import numpy as np

class MetricsCollector:
    """Collects and computes metrics during a simulation episode"""
    
    def __init__(self):
        # Initialize data lists
        self.timestamps: List[float] = []
        self.velocities: List[float] = []
        self.accelerations: List[float] = []
        self.jerks: List[float] = []  # NEW: Track jerk values
        self.ttcs: List[Optional[float]] = []
        self.fsm_states: List[str] = []
        self.ground_truths: List[Dict] = []  # NEW: Store ground truth for post-processing
        
        # Running metrics
        self.collisions = 0
        self.critical_situations = 0
        self.lane_changes = 0
        self.detection_failures = 0
        self.min_ttc = None

    def record_step(self,
                   timestamp: float,
                   ground_truth: Dict,
                   noisy_obs: Dict,
                   fsm_state: str,
                   action: int,
                   info: Dict):
        """Records data at each simulation step"""
        
        # Basic data recording
        self.timestamps.append(timestamp)
        self.velocities.append(ground_truth['ego']['velocity'])
        self.ttcs.append(info.get('ttc', None))
        self.fsm_states.append(fsm_state)
        self.ground_truths.append(ground_truth)  # NEW: Store full ground truth

        # Event statistics
        if info.get('collision', False) or info.get('crashed', False):
            self.collisions += 1
        if info.get('critical', False):
            self.critical_situations += 1
        if info.get('lane_change', False):
            self.lane_changes += 1
        if info.get('detection_failure', False):
            self.detection_failures += 1

        # Update minimum TTC
        current_ttc = info.get('ttc')
        if current_ttc is not None:
            if self.min_ttc is None or current_ttc < self.min_ttc:
                self.min_ttc = current_ttc

    def get_summary(self) -> Dict:
        """Returns a summary of collected metrics"""
        duration = self.timestamps[-1] if self.timestamps else 0
        velocities_array = np.array(self.velocities)
        accelerations_array = np.array(self.accelerations) if self.accelerations else np.array([0])
        jerks_array = np.array(self.jerks) if self.jerks else np.array([0])

        # Calculate average absolute jerk (better comfort metric)
        avg_jerk = float(np.mean(jerks_array)) if len(jerks_array) > 0 else 0.0
        
        # Calculate max jerk (worst case smoothness)
        max_jerk = float(np.max(jerks_array)) if len(jerks_array) > 0 else 0.0

        return {
            'safety': {
                'collisions': self.collisions,
                'min_ttc': self.min_ttc,
                'critical_situations': self.critical_situations
            },
            'efficiency': {
                'avg_velocity': float(np.mean(velocities_array)),
                'max_velocity': float(np.max(velocities_array)),
                'min_velocity': float(np.min(velocities_array)),
                'lane_changes': self.lane_changes,
                'duration': duration
            },
            'comfort': {
                'avg_acceleration': float(np.mean(np.abs(accelerations_array))),
                'max_acceleration': float(np.max(np.abs(accelerations_array))),
                'jerk': avg_jerk,
                'max_jerk': max_jerk,
                'std_acceleration': float(np.std(accelerations_array))
            },
            'sensor_impact': {
                'total_detection_failures': self.detection_failures
            }
        }