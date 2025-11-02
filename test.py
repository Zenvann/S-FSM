import gymnasium as gym
import highway_env

# Test 1: Highway Szenario
env = gym.make("highway-v0", render_mode="human")
obs, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()  # Zufällige Aktion
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done or truncated:
        break
env.close()

# Test 2: Merge Szenario (für Spurwechsel)
env = gym.make("merge-v0", render_mode="human")
obs, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()  # Zufällige Aktion
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done or truncated:
        break
env.close()