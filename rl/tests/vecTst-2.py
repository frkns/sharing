import gymnasium as gym
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
import numpy as np
import time

def make_env_with_delay(env_id, delay=0.001):
    """Create an environment and add a simulated delay (in seconds) to its step() method."""
    def _init():
        env = gym.make(env_id)
        original_step = env.step
        def step_with_delay(action):
            time.sleep(delay)  # Simulated 1ms computation per step
            return original_step(action)
        env.step = step_with_delay
        return env
    return _init

def sequential_run(env_id, total_steps, delay=0.001):
    """Run a single environment sequentially for a total number of steps and measure time."""
    env = make_env_with_delay(env_id, delay)()
    obs, info = env.reset()
    start_time = time.time()
    
    for _ in range(total_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
    return time.time() - start_time

def vectorized_run(env_id, total_steps, num_envs, EnvClass, delay=0.001):
    """
    Run vectorized environments (synchronously or asynchronously) for total_steps steps
    (steps are divided evenly among num_envs) and measure time.
    
    EnvClass should be either SyncVectorEnv or AsyncVectorEnv.
    """
    steps_per_env = total_steps // num_envs
    # Create a list of environment constructors that add the delay to each env
    env_fns = [make_env_with_delay(env_id, delay) for _ in range(num_envs)]
    envs = EnvClass(env_fns)
    
    obs, info = envs.reset()
    start_time = time.time()
    
    for _ in range(steps_per_env):
        # Sample one action per environment
        actions = [envs.single_action_space.sample() for _ in range(num_envs)]
        obs, rewards, terminateds, truncateds, infos = envs.step(actions)
        # Check for episodes that ended and reset those environments
        dones = np.logical_or(terminateds, truncateds)
        if np.any(dones):
            new_obs, new_info = envs.reset()
            for i, done in enumerate(dones):
                if done:
                    obs[i] = new_obs[i]
    
    envs.close()
    return time.time() - start_time

def run_benchmark():
    env_id = "CartPole-v1"
    total_steps = 1000  # Total number of steps for the sequential run
    num_envs = 10          # Number of environments for vectorized runs
    delay = 0.0001         # s delay per step
    
    print(f"Benchmarking {env_id} with {delay*1000:.5f}ms simulated computation per step")
    print(f"Sequential run: {total_steps} steps")
    print(f"Vectorized run: {num_envs} envs, {total_steps // num_envs} steps each")
    print("-" * 70)
    
    # Sequential benchmark
    seq_time = sequential_run(env_id, total_steps, delay)
    seq_speed = total_steps / seq_time
    print(f"Sequential run: {seq_time:.4f} seconds ({seq_speed:.0f} steps/sec)")
    
    # Synchronous vectorized benchmark
    sync_time = vectorized_run(env_id, total_steps, num_envs, SyncVectorEnv, delay)
    sync_speed = total_steps / sync_time
    print(f"Sync vectorized run: {sync_time:.4f} seconds ({sync_speed:.0f} steps/sec)")
    print(f"Speedup vs sequential: {seq_time / sync_time:.2f}x")
    
    # Asynchronous vectorized benchmark
    async_time = vectorized_run(env_id, total_steps, num_envs, AsyncVectorEnv, delay)
    async_speed = total_steps / async_time
    print(f"Async vectorized run: {async_time:.4f} seconds ({async_speed:.0f} steps/sec)")
    print(f"Speedup vs sequential: {seq_time / async_time:.2f}x")
    print(f"Speedup vs sync vectorized: {sync_time / async_time:.2f}x")

if __name__ == "__main__":
    run_benchmark()

