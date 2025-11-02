import gymnasium as gym
import highway_env

class HighwayEnvironment:
    def __init__(self, scenario_name="highway-v0"):
        self.env = gym.make(scenario_name)
        self.observation = None
        self.info = {}
        
    def reset(self):
        """重置环境"""
        self.observation, self.info = self.env.reset()
        return self.observation
    
    def step(self, action):
        """执行动作并返回结果"""
        # 将动作字符串映射到环境动作索引
        action_map = {
            "IDLE": 0,
            "LANE_LEFT": 1,
            "LANE_RIGHT": 2,
            "FASTER": 3,
            "SLOWER": 4
        }
        action_idx = action_map.get(action, 0)
        
        # 执行动作
        self.observation, reward, terminated, truncated, self.info = self.env.step(action_idx)
        done = terminated or truncated
        
        return self.observation, reward, done, self.info
    
    def render(self):
        """渲染环境"""
        self.env.render()
    
    def close(self):
        """关闭环境"""
        self.env.close()