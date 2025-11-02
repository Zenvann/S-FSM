import numpy as np
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class SensorModel:
    """
    传感器模型类，用于模拟不同类型的传感器及其特性
    """
    
    def __init__(self, sensor_config: Dict[str, Any]):
        """
        初始化传感器模型
        
        参数:
            sensor_config: 传感器配置字典，包含各种传感器的参数
        """
        self.config = sensor_config
        self.sensor_types = sensor_config.get("sensor_types", [])
        self.sensor_noise_params = sensor_config.get("noise_params", {})
        self.sensor_failure_params = sensor_config.get("failure_params", {})
        
        # 初始化传感器状态
        self.sensor_states = {}
        for sensor_type in self.sensor_types:
            self.sensor_states[sensor_type] = {
                "last_update": 0,
                "failure_mode": False,
                "failure_start_time": 0,
                "failure_duration": 0
            }
    
    def get_observation(self, env_observation: np.ndarray, step_count: int) -> Dict[str, Any]:
        """
        从环境观测中提取信息，并根据传感器配置添加噪声和故障
        
        参数:
            env_observation: 原始环境观测数据
            step_count: 当前仿真步数
            
        返回:
            处理后的观测数据字典
        """
        # 解析原始观测数据
        parsed_obs = self._parse_raw_observation(env_observation)
        
        # 应用传感器模型
        processed_obs = {}
        
        # 根据传感器类型处理数据
        if "position" in self.sensor_types:
            processed_obs["ego_position"] = self._apply_sensor_model(
                parsed_obs["ego_position"], "position", step_count
            )
        
        if "velocity" in self.sensor_types:
            processed_obs["ego_velocity"] = self._apply_sensor_model(
                parsed_obs["ego_velocity"], "velocity", step_count
            )
        
        if "lidar" in self.sensor_types:
            processed_obs["surrounding_vehicles"] = self._apply_sensor_model(
                parsed_obs["surrounding_vehicles"], "lidar", step_count
            )
        
        if "camera" in self.sensor_types:
            processed_obs["lane_info"] = self._apply_sensor_model(
                parsed_obs["lane_info"], "camera", step_count
            )
        
        if "radar" in self.sensor_types:
            processed_obs["relative_velocities"] = self._apply_sensor_model(
                parsed_obs["relative_velocities"], "radar", step_count
            )
        
        return processed_obs
    
    def _parse_raw_observation(self, obs: np.ndarray) -> Dict[str, Any]:
        """
        解析原始观测数据，提取有用信息
        
        参数:
            obs: 原始观测数组
            
        返回:
            解析后的观测数据字典
        """
        # 这里根据highway_env的观测结构进行解析
        # 注意: highway_env的观测结构可能因配置不同而变化
        # 以下是一个示例解析方式，可能需要根据实际观测结构调整
        
        parsed = {}
        
        # 假设观测数组结构:
        # [ego_x, ego_y, ego_vx, ego_vy, lane_id, 
        #  vehicle1_x, vehicle1_y, vehicle1_vx, vehicle1_vy, 
        #  vehicle2_x, ...]
        
        # 自车位置和速度
        parsed["ego_position"] = np.array([obs[0], obs[1]])
        parsed["ego_velocity"] = np.array([obs[2], obs[3]])
        
        # 车道信息
        parsed["lane_info"] = {
            "current_lane": int(obs[4]),
            "lane_width": 4.0  # 假设标准车道宽度
        }
        
        # 周围车辆信息
        n_vehicles = (len(obs) - 5) // 4
        surrounding_vehicles = []
        relative_velocities = []
        
        for i in range(n_vehicles):
            start_idx = 5 + i * 4
            if start_idx + 3 >= len(obs):
                break
                
            vehicle_pos = np.array([obs[start_idx], obs[start_idx + 1]])
            vehicle_vel = np.array([obs[start_idx + 2], obs[start_idx + 3]])
            
            # 计算相对位置和速度
            rel_pos = vehicle_pos - parsed["ego_position"]
            rel_vel = vehicle_vel - parsed["ego_velocity"]
            
            surrounding_vehicles.append({
                "position": vehicle_pos,
                "velocity": vehicle_vel,
                "relative_position": rel_pos,
                "distance": np.linalg.norm(rel_pos)
            })
            
            relative_velocities.append({
                "relative_velocity": rel_vel,
                "speed_difference": np.linalg.norm(rel_vel)
            })
        
        parsed["surrounding_vehicles"] = surrounding_vehicles
        parsed["relative_velocities"] = relative_velocities
        
        return parsed
    
    def _apply_sensor_model(self, data: Any, sensor_type: str, step_count: int) -> Any:
        """
        应用传感器模型，包括噪声注入和故障模拟
        
        参数:
            data: 原始数据
            sensor_type: 传感器类型
            step_count: 当前仿真步数
            
        返回:
            处理后的数据
        """
        # 检查传感器状态
        self._update_sensor_state(sensor_type, step_count)
        
        # 如果传感器处于故障模式，返回故障数据
        if self.sensor_states[sensor_type]["failure_mode"]:
            return self._simulate_sensor_failure(data, sensor_type)
        
        # 否则应用噪声模型
        return self._apply_noise_model(data, sensor_type)
    
    def _update_sensor_state(self, sensor_type: str, step_count: int):
        """
        更新传感器状态，包括故障检测
        
        参数:
            sensor_type: 传感器类型
            step_count: 当前仿真步数
        """
        state = self.sensor_states[sensor_type]
        
        # 检查是否需要启动故障
        if not state["failure_mode"]:
            failure_prob = self.sensor_failure_params.get(sensor_type, {}).get("failure_probability", 0.0)
            if np.random.random() < failure_prob:
                state["failure_mode"] = True
                state["failure_start_time"] = step_count
                state["failure_duration"] = self.sensor_failure_params.get(sensor_type, {}).get(
                    "failure_duration", 10
                )
                logger.warning(f"Sensor {sensor_type} failed at step {step_count}")
        
        # 检查是否需要恢复
        elif state["failure_mode"] and step_count - state["failure_start_time"] > state["failure_duration"]:
            state["failure_mode"] = False
            logger.info(f"Sensor {sensor_type} recovered at step {step_count}")
    
    def _apply_noise_model(self, data: Any, sensor_type: str) -> Any:
        """
        应用噪声模型到数据
        
        参数:
            data: 原始数据
            sensor_type: 传感器类型
            
        返回:
            添加噪声后的数据
        """
        noise_params = self.sensor_noise_params.get(sensor_type, {})
        
        if sensor_type == "position":
            # GPS位置噪声
            position_noise_std = noise_params.get("position_noise_std", 1.0)
            return data + np.random.normal(0, position_noise_std, data.shape)
        
        elif sensor_type == "velocity":
            # 速度传感器噪声
            velocity_noise_std = noise_params.get("velocity_noise_std", 0.5)
            return data + np.random.normal(0, velocity_noise_std, data.shape)
        
        elif sensor_type == "lidar":
            # LiDAR点云噪声
            distance_noise_std = noise_params.get("distance_noise_std", 0.1)
            angle_noise_std = noise_params.get("angle_noise_std", 0.01)
            
            noisy_vehicles = []
            for vehicle in data:
                # 添加距离噪声
                noisy_distance = vehicle["distance"] + np.random.normal(0, distance_noise_std)
                
                # 添加角度噪声
                angle = np.arctan2(vehicle["relative_position"][1], vehicle["relative_position"][0])
                noisy_angle = angle + np.random.normal(0, angle_noise_std)
                
                # 计算新的相对位置
                noisy_rel_pos = np.array([
                    noisy_distance * np.cos(noisy_angle),
                    noisy_distance * np.sin(noisy_angle)
                ])
                
                # 计算新的绝对位置
                noisy_pos = vehicle["position"] + (noisy_rel_pos - vehicle["relative_position"])
                
                noisy_vehicles.append({
                    **vehicle,
                    "position": noisy_pos,
                    "relative_position": noisy_rel_pos,
                    "distance": noisy_distance
                })
            
            return noisy_vehicles
        
        elif sensor_type == "camera":
            # 相机检测噪声
            lane_detection_noise = noise_params.get("lane_detection_noise", 0.1)
            
            # 添加车道检测噪声
            noisy_lane_info = data.copy()
            if np.random.random() < lane_detection_noise:
                # 错误检测车道
                lane_offset = np.random.choice([-1, 1])
                noisy_lane_info["current_lane"] = max(0, min(3, data["current_lane"] + lane_offset))
            
            return noisy_lane_info
        
        elif sensor_type == "radar":
            # 雷达速度测量噪声
            velocity_noise_std = noise_params.get("velocity_noise_std", 0.2)
            
            noisy_relative_velocities = []
            for rel_vel in data:
                noisy_rel_vel = rel_vel.copy()
                noisy_rel_vel["relative_velocity"] += np.random.normal(0, velocity_noise_std, 2)
                noisy_rel_vel["speed_difference"] = np.linalg.norm(noisy_rel_vel["relative_velocity"])
                noisy_relative_velocities.append(noisy_rel_vel)
            
            return noisy_relative_velocities
        
        # 默认返回原始数据
        return data
    
    def _simulate_sensor_failure(self, data: Any, sensor_type: str) -> Any:
        """
        模拟传感器故障
        
        参数:
            data: 原始数据
            sensor_type: 传感器类型
            
        返回:
            故障数据
        """
        failure_type = self.sensor_failure_params.get(sensor_type, {}).get("failure_type", "complete")
        
        if failure_type == "complete":
            # 完全故障 - 返回空或无数据
            if sensor_type in ["position", "velocity"]:
                return np.array([np.nan, np.nan])
            elif sensor_type in ["lidar", "radar"]:
                return []
            elif sensor_type == "camera":
                return {"current_lane": -1, "lane_width": 0.0}
        
        elif failure_type == "biased":
            # 偏置故障 - 返回有偏数据
            bias = self.sensor_failure_params.get(sensor_type, {}).get("bias", 10.0)
            
            if sensor_type == "position":
                return data + bias
            elif sensor_type == "velocity":
                return data + bias
            elif sensor_type == "lidar":
                # 为所有车辆添加距离偏置
                biased_vehicles = []
                for vehicle in data:
                    biased_vehicle = vehicle.copy()
                    biased_vehicle["distance"] += bias
                    biased_vehicle["relative_position"] = biased_vehicle["relative_position"] * (
                        1 + bias / max(1e-5, vehicle["distance"])
                    )
                    biased_vehicle["position"] = biased_vehicle["position"] + (
                        biased_vehicle["relative_position"] - vehicle["relative_position"]
                    )
                    biased_vehicles.append(biased_vehicle)
                return biased_vehicles
        
        elif failure_type == "noisy":
            # 高噪声故障 - 返回极高噪声数据
            if sensor_type == "position":
                return data + np.random.normal(0, 10.0, data.shape)
            elif sensor_type == "velocity":
                return data + np.random.normal(0, 5.0, data.shape)
            elif sensor_type == "lidar":
                # 为所有车辆添加高噪声
                noisy_vehicles = []
                for vehicle in data:
                    noisy_distance = vehicle["distance"] + np.random.normal(0, 5.0)
                    noisy_angle = np.arctan2(
                        vehicle["relative_position"][1], vehicle["relative_position"][0]
                    ) + np.random.normal(0, 0.5)
                    
                    noisy_rel_pos = np.array([
                        noisy_distance * np.cos(noisy_angle),
                        noisy_distance * np.sin(noisy_angle)
                    ])
                    
                    noisy_pos = vehicle["position"] + (noisy_rel_pos - vehicle["relative_position"])
                    
                    noisy_vehicles.append({
                        **vehicle,
                        "position": noisy_pos,
                        "relative_position": noisy_rel_pos,
                        "distance": noisy_distance
                    })
                return noisy_vehicles
        
        # 默认返回空数据
        if sensor_type in ["position", "velocity"]:
            return np.array([np.nan, np.nan])
        elif sensor_type in ["lidar", "radar"]:
            return []
        elif sensor_type == "camera":
            return {"current_lane": -1, "lane_width": 0.0}