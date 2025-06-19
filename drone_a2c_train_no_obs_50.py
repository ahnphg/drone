#!/usr/bin/env python3
"""
Drone A2C for direct 50m flight - No transfer learning needed
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
from matplotlib.patches import Arrow

# ============================== DRONE MODEL ==============================
class DirectFlightDrone:
    def __init__(self, target_z=5.0, kp=6.0, ki=0.03, kd=4.0):  # Tăng PID để giữ ổn định hơn
        self.mass = 1.0
        self.I = 0.2
        self.kd = 0.02  # Drag coefficient
        self.g = 9.81
        self.state = np.array([0.0, target_z, 0.0, 0.0, 0.0, 0.0])  # [x, z, theta, x_dot, z_dot, theta_dot]
        self.target_z = target_z
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral_z = 0.0
        self.prev_error_z = 0.0

    def update(self, tau, dt=0.1):
        # PID control for altitude
        error = self.target_z - self.state[1]
        self.integral_z += error * dt
        derivative = (error - self.prev_error_z) / dt
        thrust = self.kp * error + self.ki * self.integral_z + self.kd * derivative
        thrust = np.clip(thrust, 20.0, 35.0)  # Realistic thrust limits
        self.prev_error_z = error

        # Physics update
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

# ============================== ENVIRONMENT ==============================
class DirectDroneEnv(gym.Env):
    def __init__(self, target_x=50.0):
        super().__init__()
        self.drone = DirectFlightDrone()
        self.target_x = target_x
        self.max_steps = 1200  # Tăng số bước tối đa
        self.step_count = 0
        
        # Action space (torque control)
        self.action_space = spaces.Box(low=-0.7, high=0.7, shape=(1,), dtype=np.float32)  # Giảm phạm vi torque
        
        # Improved observation space
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, 0, -np.pi, -10, -10, -5, 0, -10]),
            high=np.array([np.inf, 20, np.pi, 10, 10, 5, np.inf, 10]),
            shape=(8,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        self.step_count = 0
        # Start with small randomness
        self.drone.state = np.array([
            np.random.uniform(-1.0, 1.0),  # x
            5.0 + np.random.uniform(-0.3, 0.3),  # z
            np.random.uniform(-0.1, 0.1),  # theta
            0.0,  # x_dot
            0.0,  # z_dot
            0.0  # theta_dot
        ])
        return self._get_obs(), {}

    def _get_obs(self):
        x, z, theta, x_dot, z_dot, theta_dot = self.drone.state
        dx = self.target_x - x
        # Normalized observations
        return np.array([
            x/self.target_x,  # Normalized x position
            (z - 5.0)/5.0,  # Normalized altitude error
            theta/np.pi,  # Normalized angle
            x_dot/5.0,  # Normalized x velocity
            z_dot/2.0,  # Normalized z velocity
            theta_dot/2.0,  # Normalized angular velocity
            dx/self.target_x,  # Normalized distance to target
            np.sign(dx) * x_dot/5.0  # Direction-corrected velocity
        ], dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        tau = action[0] * 2.5  # Giảm phạm vi torque
        
        prev_state = self.drone.state.copy()
        state = self.drone.update(tau)
        obs = self._get_obs()
        
        x, z, theta = state[0], state[1], state[2]
        dx = self.target_x - x
        vx = state[3]
        progress = (x - prev_state[0]) / self.target_x  # Normalized progress

        # Calculate rewards
        reward = 0
        
        # Progress reward (most important)
        reward += 100.0 * progress
        
        # Velocity reward
        if dx * vx > 0:  # Moving toward target
            reward += 30.0 * abs(vx)
        else:  # Moving away
            reward -= 15.0 * abs(vx)
            
        # Altitude reward
        altitude_error = abs(z - 5.0)
        reward -= 2.0 * altitude_error  # Tăng phạt lệch độ cao
        
        # Stability rewards
        reward -= 1.0 * abs(theta)  # Tăng phạt góc lớn
        reward -= 0.15 * abs(state[5])  # Tăng phạt vận tốc góc lớn
        if abs(theta) < np.deg2rad(10):
            reward += 2.0  # Thưởng nhỏ khi giữ góc nhỏ
        
        # Terminal conditions
        terminated = False
        term_reason = ""
        
        # Success condition
        if abs(dx) < 0.5:
            reward += 2000.0
            terminated = True
            term_reason = "TARGET_REACHED"
            
        # Failure conditions
        elif z < 2.0 or z > 15.0:
            reward -= 1000.0
            terminated = True
            term_reason = "ALTITUDE_VIOLATION"
        elif abs(theta) > np.deg2rad(75):  # Nới điều kiện góc lên 75 độ
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
def train_direct(target_x=50.0, save_dir="direct_models", log_dir="direct_logs"):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create env with normalization
    env = DummyVecEnv([lambda: Monitor(DirectDroneEnv(target_x=target_x))])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=20_000,
        save_path=save_dir,
        name_prefix=f"direct_drone"
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
        n_steps=512,  # Increased for better credit assignment
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=log_dir
    )
    
    # Train
    model.learn(
        total_timesteps=400_000,  # Increased for harder task
        callback=[checkpoint_cb, eval_cb],
        tb_log_name="a2c_direct_drone"
    )
    
    # Save final model
    final_path = os.path.join(save_dir, "direct_drone_final")
    model.save(final_path)
    env.save(os.path.join(save_dir, "direct_drone_env.pkl"))
    
    return final_path

# ============================== VISUALIZATION ==============================
def visualize_direct(model_path, target_x=50.0):
    # Load model
    model = A2C.load(model_path)
    env = DirectDroneEnv(target_x=target_x)
    
    obs, _ = env.reset()
    traj = []
    states = []
    done = False

    # Lưu trạng thái thực tế của drone
    real_states = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, _, _ = env.step(action)
        # Lưu trạng thái thực tế (không chuẩn hóa)
        real_states.append(env.drone.state.copy())
        done = terminated

    # Chuyển thành numpy array để dễ vẽ
    real_states = np.array(real_states)
    traj = real_states[:, [0, 1]]  # x, z

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.set_xlim(-5, target_x + 5)
    ax.set_ylim(0, 15)
    ax.set_xlabel("Horizontal Distance (m)")
    ax.set_ylabel("Altitude (m)")
    ax.set_title(f"Direct Drone Flight to {target_x}m")
    ax.grid(True)
    
    # Plot trajectory
    ax.plot(traj[:,0], traj[:,1], 'b-', lw=2, label='Flight Path')
    
    # Plot orientation markers
    for i in range(0, len(traj), 20):
        x, z = traj[i]
        theta = real_states[i][2]
        ax.add_patch(Arrow(x, z, 0.5*np.cos(theta), 0.5*np.sin(theta), 
                    width=0.2, color='red', alpha=0.7))
    
    ax.plot(traj[0,0], traj[0,1], 'go', markersize=10, label='Start')
    ax.plot(target_x, 5.0, 'ro', markersize=10, label='Target')
    ax.legend()
    plt.show()

# ============================== MAIN ==============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'visualize'], required=True)
    args = parser.parse_args()

    target_distance = 50.0  # Direct 50m flight
    model_dir = "direct_models"
    
    if args.mode == 'train':
        print(f"Training direct {target_distance}m flight...")
        model_path = train_direct(target_x=target_distance, 
                                save_dir=model_dir,
                                log_dir="direct_logs")
        visualize_direct(model_path, target_distance)
    else:
        model_path = os.path.join(model_dir, "direct_drone_final.zip")
        if os.path.exists(model_path):
            visualize_direct(model_path, target_distance)
        else:
            print(f"Model not found at {model_path}. Train first with --mode train")

if __name__ == '__main__':
    main()