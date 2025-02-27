import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
import numpy as np
import time

def make_env_with_delay(env_id, delay=0.001):
    """Create environment with simulated computational delay."""
    def _init():
        env = gym.make(env_id)
        # Simulate computational work by adding a small sleep
        original_step = env.step
        def step_with_delay(action):
            time.sleep(delay)  # Simulate computation
            return original_step(action)
        env.step = step_with_delay
        return env
    return _init

def sequential_run(env_id, num_envs, steps_per_env, delay=0.001):
    """Run environments sequentially and measure time."""
    total_start = time.time()
    
    # Create and run each environment one after another
    for env_idx in range(num_envs):
        env = make_env_with_delay(env_id, delay)()
        obs, _ = env.reset()
        
        for step in range(steps_per_env):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                obs, _ = env.reset()
        
        env.close()
    
    total_time = time.time() - total_start
    return total_time

def sync_vectorized_run(env_id, num_envs, steps_per_env, delay=0.001):
    """Run environments using SyncVectorEnv and measure time."""
    # Create environments
    envs = SyncVectorEnv([make_env_with_delay(env_id, delay) for _ in range(num_envs)])
    
    # Run environments
    total_start = time.time()
    
    obs, _ = envs.reset()
    for step in range(steps_per_env):
        actions = np.array([envs.single_action_space.sample() for _ in range(num_envs)])
        obs, rewards, terminateds, truncateds, infos = envs.step(actions)
        
        # Reset terminated environments
        dones = np.logical_or(terminateds, truncateds)
        if np.any(dones):
            # This is how you'd handle resets in vectorized envs
            pass
    
    envs.close()
    total_time = time.time() - total_start
    return total_time

def async_vectorized_run(env_id, num_envs, steps_per_env, delay=0.001):
    """Run environments using AsyncVectorEnv and measure time."""
    # Create environments
    envs = AsyncVectorEnv([make_env_with_delay(env_id, delay) for _ in range(num_envs)])
    
    # Run environments
    total_start = time.time()
    
    obs, _ = envs.reset()
    for step in range(steps_per_env):
        actions = np.array([envs.single_action_space.sample() for _ in range(num_envs)])
        obs, rewards, terminateds, truncateds, infos = envs.step(actions)
        
        # Reset terminated environments
        dones = np.logical_or(terminateds, truncateds)
        if np.any(dones):
            # This is how you'd handle resets in vectorized envs
            pass
    
    envs.close()
    total_time = time.time() - total_start
    return total_time

def run_benchmark():
    """Run benchmark comparing different methods with more intensive settings."""
    # Parameters
    env_id = "CartPole-v1"  # Using CartPole but with simulated computation
    num_envs = 10  # Increased number of environments
    steps_per_env = 100  # Increased steps per environment
    delay = 0.001  # delay to simulate computation
    
    print(f"Benchmark: {num_envs} environments, {steps_per_env} steps each, {delay*1000}ms simulated computation per step")
    print("-" * 70)
    
    # Sequential run
    # print()
    print("* Running sequential benchmark...")
    seq_time = sequential_run(env_id, num_envs, steps_per_env, delay)
    print(f"Sequential execution time: {seq_time:.4f} seconds")
    
    # Synchronous vectorized run
    # print()
    print("* Running synchronous vectorized benchmark...")
    sync_time = sync_vectorized_run(env_id, num_envs, steps_per_env, delay)
    print(f"Sync vectorized execution time: {sync_time:.4f} seconds")
    print(f"Speedup vs sequential: {seq_time/sync_time:.2f}x")
    
    # Asynchronous vectorized run
    # print()
    print("* Running asynchronous vectorized benchmark...")
    async_time = async_vectorized_run(env_id, num_envs, steps_per_env, delay)
    print(f"Async vectorized execution time: {async_time:.4f} seconds")
    print(f"Speedup vs sequential: {seq_time/async_time:.2f}x")
    print(f"Speedup vs sync vectorized: {sync_time/async_time:.2f}x")

if __name__ == "__main__":
    run_benchmark()
