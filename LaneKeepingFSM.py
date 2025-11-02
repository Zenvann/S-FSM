from statemachine import StateMachine, State
import random

class LaneKeepingFSM(StateMachine):
    # Define States
    lane_keep = State('LaneKeep', initial=True)
    follow = State('Follow')
    cruise = State('Cruise')
    warning = State('Warning')
    brake = State('Brake')

    # Define Transitions
    detect_vehicle = lane_keep.to(follow)
    road_clear = follow.to(cruise)
    obstacle_detected = lane_keep.to(warning)
    emergency_break = follow.to(brake) | warning.to(brake)
    resume_driving = brake.to(lane_keep)
    obstacle_cleared = warning.to(lane_keep)

    def __init__(self):
        super().__init__()
        self.lateral_deviation = 0.0
        self.ttc = 2.0 # cite ttc
        self.tet = 2.0 # cite ttc???
        self.tit = 2.0 # cite ttc???
        self.front_vehicle_distance = 50.0
    
    def update_sensor_data(self, lateral_deviation, ttc, tet, tit, front_vehicle_distance)
        # Update sensor data
        self.lateral_deviation = lateral_deviation
        self.ttc = ttc
        self.tet = tet # cite ttc???
        self.tit = tit # cite ttc???
        self.front_vehicle_distance = front_vehicle_distance
        self.check_transitions()
    
    def check_transitions(self)
        # Check if transition happens

    def enter_lane_keep(self)

    def enter_follow(self)

    def enter_cruise(self)

    def enter_warning(self)

    def enter_brake(self)


if __name__ == "__main__":
    fsm = LaneKeepingFSM()