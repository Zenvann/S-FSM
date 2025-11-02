from statemachine import StateMachine, State
import numpy as np

class ADSStateMachine(StateMachine):
    # Define States
    keep_lane = State('KeepLane', initial=True)
    change_left = State('ChangeLeft')
    change_right = State('ChangeRight')
    accelerate = State('Accelerate')
    decelerate = State('Decelerate')
    
    # Define Tranasition
    lane_clear_left = keep_lane.to(change_left)
    lane_clear_right = keep_lane.to(change_right)
    lane_change_complete_left = change_left.to(keep_lane)
    lane_change_complete_right = change_right.to(keep_lane)
    front_vehicle_close = keep_lane.to(decelerate)
    front_vehicle_far = decelerate.to(keep_lane)
    no_obstacles = keep_lane.to(accelerate)
    obstacles_detected = accelerate.to(keep_lane)
    
    def __init__(self, variant_params, sensor_config):
        super().__init__()
        self.variant_params = variant_params
        self.current_action = "IDLE"
        self.sensor_model = SensorModel(sensor_config)  # 创建传感器模型实例
        self.step_count = 0  # 添加步数计数器
        
    def on_step(self, observation):
        """根据观测决定下一步动作"""
        self.step_count += 1
        processed_obs = self.sensor_model.get_observation(observation, self.step_count)
        ego_speed = np.linalg.norm(processed_obs["ego_velocity"]) if not np.isnan(processed_obs["ego_velocity"][0]) else 0
        front_vehicle_distance = self._get_front_vehicle_distance(processed_obs["surrounding_vehicles"])
        front_vehicle_distance = observation[4] if observation[4] > 0 else 100
        left_lane_clear = self._check_lane_clear(observation, 'left')
        right_lane_clear = self._check_lane_clear(observation, 'right')
        
        # 基于当前状态和观测值决定状态转换
        if self.current_state == self.keep_lane:
            if front_vehicle_distance < self.variant_params['safe_distance']:
                self.front_vehicle_close()
            elif front_vehicle_distance > self.variant_params['change_distance'] and left_lane_clear:
                self.lane_clear_left()
            elif front_vehicle_distance > self.variant_params['change_distance'] and right_lane_clear:
                self.lane_clear_right()
            elif front_vehicle_distance > self.variant_params['accelerate_distance']:
                self.no_obstacles()
                
        elif self.current_state == self.change_left:
            if self._is_lane_change_complete(observation, 'left'):
                self.lane_change_complete_left()
                
        elif self.current_state == self.change_right:
            if self._is_lane_change_complete(observation, 'right'):
                self.lane_change_complete_right()
                
        elif self.current_state == self.accelerate:
            if front_vehicle_distance < self.variant_params['decelerate_distance']:
                self.obstacles_detected()
                
        elif self.current_state == self.decelerate:
            if front_vehicle_distance > self.variant_params['resume_distance']:
                self.front_vehicle_far()
        
        # 添加随机不确定性
        self._apply_uncertainty()
        
        # 映射状态到动作
        self._map_state_to_action()
        
        return self.current_action
    
    def _check_lane_clear(self, observation, direction):
        """检查指定方向的车道是否畅通"""
        # 简化实现 - 实际应根据观测数据判断
        if direction == 'left':
            return np.random.random() > self.variant_params['perception_uncertainty']
        else:
            return np.random.random() > self.variant_params['perception_uncertainty']
    
    def _is_lane_change_complete(self, observation, direction):
        """检查换道是否完成"""
        # 简化实现
        return np.random.random() > 0.3  # 70%的概率完成换道
    
    def _apply_uncertainty(self):
        """应用随机不确定性"""
        if np.random.random() < self.variant_params['transition_uncertainty']:
            # 随机切换到另一个状态
            possible_states = [s for s in self.states if s != self.current_state]
            if possible_states:
                self.current_state = np.random.choice(possible_states)
    
    def _map_state_to_action(self):
        """将状态映射到highway_env动作"""
        action_map = {
            self.keep_lane: "LANE_RIGHT",  # 实际应根据情况选择
            self.change_left: "LANE_LEFT",
            self.change_right: "LANE_RIGHT",
            self.accelerate: "FASTER",
            self.decelerate: "SLOWER"
        }
        self.current_action = action_map.get(self.current_state, "IDLE")