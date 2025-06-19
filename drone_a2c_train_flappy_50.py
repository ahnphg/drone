#!/usr/bin/env python3
"""
Drone Flappy Bird - Fly through obstacles to reach 50m
"""

import argparse
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arrow

# ============================== DRONE MODEL ==============================
class FlappyDrone:
    def __init__(self, target_z=5.0, kp=6.0, ki=0.03, kd=4.0):
        self.mass = 1.0
        self.I = 0.2
        self.kd = 0.02
        self.g = 9.81
        self.state = np.array([0.0, target_z, 0.0, 0.0, 0.0, 0.0])
        self.target_z = target_z
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral_z = 0.0
        self.prev_error_z = 0.0

    def update(self, tau, dt=0.1):
        error = self.target_z - self.state[1]
        self.integral_z += error * dt
        derivative = (error - self.prev_error_z) / dt
        thrust = self.kp * error + self.ki * self.integral_z + self.kd * derivative
        thrust = np.clip(thrust, 20.0, 35.0)
        self.prev_error_z = error

        x, z, theta, x_dot, z_dot, theta_dot = self.state
        x_ddot = (thrust * np.sin(theta) / self.mass) - (self.kd / self.mass) * x_dot
        z_ddot = (thrust * np.cos(theta) / self.mass) - self.g - (self.kd / self.mass) * z_dot
        theta_ddot = tau / self.I

        self.state += np.array([
            x_dot * dt,
            z_dot * dt,
            theta_dot * dt,
            x_ddot * dt,
            z_ddot * dt,
            theta_ddot * dt
        ])
        return self.state.copy()

# ============================== ENVIRONMENT WITH OBSTACLES ==============================
class FlappyDroneEnv(gym.Env):
    def __init__(self, target_x=50.0):
        super().__init__()
        self.drone = FlappyDrone()
        self.target_x = target_x
        self.max_steps = 1500
        self.step_count = 0
        self.obstacles = []  # List of obstacles (x_pos, gap_center)
        self.obstacle_spacing = 10.0  # Distance between obstacles
        self.obstacle_width = 2.0
        self.gap_size = 4.0
        self.drone_radius = 0.5  # Collision detection radius
        
        # Action space
        self.action_space = spaces.Box(low=-0.7, high=0.7, shape=(1,), dtype=np.float32)
        
        # Observation space: drone state + next obstacle info
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, 0, -np.pi, -10, -10, -5, 0, -10, 0, 0]),
            high=np.array([np.inf, 20, np.pi, 10, 10, 5, np.inf, 10, np.inf, 20]),
            shape=(10,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        self.step_count = 0
        self.obstacles = []
        
        # Initialize drone with small randomness
        self.drone.state = np.array([
            np.random.uniform(-1.0, 1.0),
            5.0 + np.random.uniform(-0.3, 0.3),
            np.random.uniform(-0.1, 0.1),
            0.0, 0.0, 0.0
        ])
        
        # Generate first obstacle
        self._generate_next_obstacle()
        return self._get_obs(), {}

    def _generate_next_obstacle(self):
        # Generate obstacles at increasing distances
        if not self.obstacles:
            next_x = 15.0
        else:
            next_x = self.obstacles[-1][0] + self.obstacle_spacing
        
        # Gap center between 3.5 and 6.5 meters
        gap_center = np.random.uniform(3.5, 6.5)
        self.obstacles.append((next_x, gap_center))

    def _get_obs(self):
        x, z, theta, x_dot, z_dot, theta_dot = self.drone.state
        dx = self.target_x - x
        
        # Find next obstacle
        next_obstacle_x = 1000.0
        next_gap_center = 5.0
        for obs_x, gap_center in self.obstacles:
            if obs_x > x:
                next_obstacle_x = obs_x
                next_gap_center = gap_center
                break
        
        return np.array([
            x/self.target_x,
            (z - 5.0)/5.0,
            theta/np.pi,
            x_dot/5.0,
            z_dot/2.0,
            theta_dot/2.0,
            dx/self.target_x,
            np.sign(dx) * x_dot/5.0,
            (next_obstacle_x - x)/self.target_x,  # Distance to next obstacle
            (z - next_gap_center)/5.0  # Relative position to gap center
        ], dtype=np.float32)

    def _check_collision(self, x, z):
        # Check collision with obstacles
        for obs_x, gap_center in self.obstacles:
            if abs(x - obs_x) < self.obstacle_width/2 + self.drone_radius:
                gap_top = gap_center + self.gap_size/2
                gap_bottom = gap_center - self.gap_size/2
                if not (gap_bottom < z < gap_top):
                    return True
        return False

    def step(self, action):
        self.step_count += 1
        tau = action[0] * 2.5
        
        prev_state = self.drone.state.copy()
        state = self.drone.update(tau)
        x, z, theta = state[0], state[1], state[2]
        
        # Check for collisions
        collision = self._check_collision(x, z)
        
        # Remove passed obstacles
        self.obstacles = [(obs_x, gap) for obs_x, gap in self.obstacles if obs_x > x - 10]
        
        # Generate new obstacles if needed
        if not self.obstacles or self.obstacles[-1][0] < x + 20:
            self._generate_next_obstacle()
        
        obs = self._get_obs()
        dx = self.target_x - x
        vx = state[3]
        progress = (x - prev_state[0]) / self.target_x

        # Calculate rewards
        reward = 100.0 * progress  # Progress reward
        
        # Velocity reward
        if dx * vx > 0:
            reward += 30.0 * abs(vx)
        else:
            reward -= 15.0 * abs(vx)
            
        # Altitude reward
        altitude_error = abs(z - 5.0)
        reward -= 2.0 * altitude_error
        
        # Stability rewards
        reward -= 1.0 * abs(theta)
        reward -= 0.15 * abs(state[5])
        
        # Obstacle avoidance reward
        next_obstacle_dist = obs[8] * self.target_x
        if next_obstacle_dist < 10.0 and next_obstacle_dist > 0:
            gap_error = abs(z - self.obstacles[0][1])
            reward += 5.0 * (1 - gap_error/self.gap_size) * (10.0 - next_obstacle_dist)/10.0

        # Terminal conditions
        terminated = False
        term_reason = ""
        
        # Success condition
        if abs(dx) < 0.5:
            reward += 2000.0
            terminated = True
            term_reason = "TARGET_REACHED"
            
        # Collision condition
        elif collision:
            reward -= 1000.0
            terminated = True
            term_reason = "COLLISION"
            
        # Failure conditions
        elif z < 1.0 or z > 15.0:
            reward -= 1000.0
            terminated = True
            term_reason = "ALTITUDE_VIOLATION"
            
        elif abs(theta) > np.deg2rad(75):
            reward -= 500.0
            terminated = True
            term_reason = "ANGLE_VIOLATION"
            
        elif self.step_count >= self.max_steps:
            terminated = True
            term_reason = "MAX_STEPS"
            
        if terminated:
            print(f"[Terminated] {term_reason} | X: {x:.1f}/{self.target_x:.1f}m | Z: {z:.1f}m | Theta: {np.degrees(theta):.1f}°")
            
        return obs, reward, terminated, False, {}

# ============================== TRAINING ==============================
def train_flappy(target_x=50.0, save_dir="flappy_models", log_dir="flappy_logs"):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create env with normalization
    env = DummyVecEnv([lambda: Monitor(FlappyDroneEnv(target_x=target_x))])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=20_000,
        save_path=save_dir,
        name_prefix="flappy_drone"
    )
    
    eval_cb = EvalCallback(
        env,
        best_model_save_path=os.path.join(save_dir, "best"),
        eval_freq=10_000,
        deterministic=True,
        render=False
    )
    
    # Create model
    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=log_dir
    )
    
    # Train
    model.learn(
        total_timesteps=600_000,  # More timesteps for complex task
        callback=[checkpoint_cb, eval_cb],
        tb_log_name="a2c_flappy_drone"
    )
    
    # Save final model
    final_path = os.path.join(save_dir, "flappy_drone_final")
    model.save(final_path)
    env.save(os.path.join(save_dir, "flappy_drone_env.pkl"))
    
    return final_path

# ============================== VISUALIZATION ==============================
def visualize_flappy(model_path, target_x=50.0):
    # Load model
    model = A2C.load(model_path)
    env = FlappyDroneEnv(target_x=target_x)
    
    obs, _ = env.reset()
    traj = []
    states = []
    obstacles = []
    done = False
    
    # Run simulation
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, _, _ = env.step(action)
        x, z = env.drone.state[0], env.drone.state[1]
        traj.append([x, z])
        states.append(env.drone.state.copy())
        obstacles.append(env.obstacles.copy())
        done = terminated
    
    traj = np.array(traj)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_xlim(-2, target_x + 5)
    ax.set_ylim(0, 15)
    ax.set_xlabel("Horizontal Distance (m)")
    ax.set_ylabel("Altitude (m)")
    ax.set_title(f"Flappy Drone: Obstacle Avoidance to {target_x}m")
    ax.grid(True)
    
    # Plot obstacles
    for frame_obstacles in obstacles:
        for x, gap_center in frame_obstacles:
            if x > -2 and x < target_x + 5:
                # Bottom obstacle
                ax.add_patch(Rectangle(
                    (x - env.obstacle_width/2, 0),
                    env.obstacle_width,
                    gap_center - env.gap_size/2,
                    color='red', alpha=0.3
                ))
                # Top obstacle
                ax.add_patch(Rectangle(
                    (x - env.obstacle_width/2, gap_center + env.gap_size/2),
                    env.obstacle_width,
                    15 - (gap_center + env.gap_size/2),
                    color='red', alpha=0.3
                ))
    
    # Plot trajectory
    ax.plot(traj[:,0], traj[:,1], 'b-', lw=2, label='Flight Path')
    
    # Plot drone orientation markers
    for i in range(0, len(traj), 30):
        x, z = traj[i]
        theta = states[i][2]
        ax.add_patch(Arrow(
            x, z, 0.5*np.cos(theta), 0.5*np.sin(theta),
            width=0.2, color='blue', alpha=0.7
        ))
    
    # Mark start and target
    ax.plot(traj[0,0], traj[0,1], 'go', markersize=10, label='Start')
    ax.plot(target_x, 5.0, 'ro', markersize=10, label='Target')
    
    # Add legend
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("flappy_drone_trajectory.png", dpi=300)
    plt.show()

# ============================== MAIN ==============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'visualize'], required=True)
    args = parser.parse_args()

    target_distance = 50.0
    model_dir = "flappy_models"
    model_path = os.path.join(model_dir, "flappy_drone_final.zip")
    
    if args.mode == 'train':
        print("Training Flappy Drone with obstacles...")
        model_path = train_flappy(target_x=target_distance, 
                                 save_dir=model_dir,
                                 log_dir="flappy_logs")
        visualize_flappy(model_path, target_distance)
    else:
        if os.path.exists(model_path):
            visualize_flappy(model_path, target_distance)
        else:
            print(f"Model not found at {model_path}. Train first with --mode train")

if __name__ == '__main__':
    main()