import yaml
import argparse
from src.simulation.runner import SimulationRunner
from src.evaluation.metrics import calculate_metrics
from src.evaluation.visualizer import plot_results

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='ADS Simulation in Highway Environment')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to config file')
    parser.add_argument('--variant', type=str, required=True, help='ADS variant to simulate')
    parser.add_argument('--scenario', type=str, default='highway-v0', help='Scenario to run')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to run')
    args = parser.parse_args()
    
    # 加载配置文件
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建仿真运行器
    runner = SimulationRunner(config, args.variant, args.scenario)
    
    # 运行仿真
    results = runner.run(args.episodes)
    
    # 计算性能指标
    metrics = calculate_metrics(results)
    
    # 可视化结果
    plot_results(results, metrics, args.variant)
    
    # 保存结果
    runner.save_results(results, metrics)

if __name__ == "__main__":
    main()