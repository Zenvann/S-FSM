import time
import json
import os
from datetime import datetime
from src.ads.state_machine import ADSStateMachine
from src.ads.variants import get_variant_params
from src.simulation.environment import HighwayEnvironment

class SimulationRunner:
    def __init__(self, config, variant_name, scenario_name):
        self.config = config
        self.variant_name = variant_name
        self.variant_params = get_variant_params(variant_name)
        self.env = HighwayEnvironment(scenario_name)
        sensor_config = config.get("sebsir_config", {})
        self.state_machine = ADSStateMachine(self.variant_params, sensor_config)
        
    def run(self, num_episodes=10):
        """运行指定次数的仿真"""
        results = []
        
        for episode in range(num_episodes):
            print(f"Running episode {episode+1}/{num_episodes} for variant {self.variant_name}")
            
            # 重置环境
            obs = self.env.reset()
            done = False
            episode_data = {
                "episode": episode,
                "variant": self.variant_name,
                "steps": [],
                "total_reward": 0,
                "collisions": 0,
                "completed": False
            }
            
            step_count = 0
            start_time = time.time()
            
            while not done and step_count < self.config.get("max_steps", 1000):
                # ADS make decision
                action = self.state_machine.on_step(obs)
                
                # 执行动作
                next_obs, reward, done, info = self.env.step(action)
                
                # 记录数据
                step_data = {
                    "step": step_count,
                    "state": str(self.state_machine.current_state),
                    "action": action,
                    "reward": reward,
                    "position": info.get("position", [0, 0]),
                    "speed": info.get("speed", 0),
                    "collision": info.get("collision", False)
                }
                episode_data["steps"].append(step_data)
                episode_data["total_reward"] += reward
                
                if info.get("collision", False):
                    episode_data["collisions"] += 1
                
                # Update
                obs = next_obs
                step_count += 1
                
                # 渲染（可选）
                if self.config.get("render", False):
                    self.env.render()
            
            episode_data["completed"] = not done
            episode_data["duration"] = time.time() - start_time
            results.append(episode_data)
            
            print(f"Episode completed: Steps={step_count}, Reward={episode_data['total_reward']}, "
                  f"Collisions={episode_data['collisions']}")
        
        return results
    
    def save_results(self, results, metrics):
        """保存仿真结果"""
        # 创建结果目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = f"results/{self.variant_name}_{timestamp}"
        os.makedirs(result_dir, exist_ok=True)
        
        # 保存原始数据
        with open(f"{result_dir}/simulation_data.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # 保存性能指标
        with open(f"{result_dir}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Results saved to {result_dir}")