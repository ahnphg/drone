"""
drone_transfer_learning_sac_100m_flappy_obstacles.py

– Đầu vào: quãng đường 100m
– Huấn luyện 1 đoạn 100m, với chướng ngại vật ngẫu nhiên kiểu Flappy Bird (các cặp khối trên và dưới với khoảng trống ở giữa) và tìm điểm kết thúc an toàn
– Hiển thị đường bay
"""

# python flappy_obs.py --mode <train|visualize> --distance 100

import argparse
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow, Rectangle

# Đặt encoding mặc định cho stdout
import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ================================= MÔ HÌNH DRONE =================================
class ImprovedDrone:
    def __init__(self, target_z=5.0, kp=5.0, ki=0.05, kd=2.0):  # Tăng kp từ 3.0 lên 5.0
        self.mass = 1.0
        self.I = 0.2
        self.kd = 0.02
        self.g = 9.81
        self.state = np.array([2.0, target_z, 0.0, 0.0, 0.0, 0.0])
        self.target_z = target_z
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral_z = 0.0
        self.prev_error_z = 0.0

    def update(self, tau, dt):
        error = self.target_z - self.state[1]
        self.integral_z += error * dt
        derivative = (error - self.prev_error_z) / dt
        thrust = self.kp * error + self.ki * self.integral_z + self.kd * derivative
        thrust = np.clip(thrust, 25.0, 40.0)
        self.prev_error_z = error

        x, z, theta, x_dot, z_dot, theta_dot = self.state
        # Giới hạn góc nghiêng
        MAX_ANGLE = np.deg2rad(70)
        if abs(theta) > MAX_ANGLE:
            theta_ddot = -np.sign(theta) * 5.0  # Phản hồi để giảm góc
        else:
            theta_ddot = tau / self.I

        x_ddot = (thrust * np.sin(theta) / self.mass) - (self.kd / self.mass) * x_dot
        z_ddot = (thrust * np.cos(theta) / self.mass) - self.g - (self.kd / self.mass) * z_dot

        self.state += np.array([
            x_dot * dt,
            z_dot * dt,
            theta_dot * dt,
            x_ddot * dt,
            z_ddot * dt,
            theta_ddot * dt
        ])
        return self.state

# ================================= HỖ TRỢ TÌM ĐIỂM KẾT THÚC AN TOÀN =================================
def find_safe_endpoint(env, x_target, z_target, step=0.5, max_offset=10.0):
    """Tìm điểm x an toàn gần x_target nếu x_target va chạm."""
    if not env._check_collision(x_target, z_target):
        return x_target
    for offset in np.arange(step, max_offset + step, step):
        for dir in (-1, 1):
            xt = x_target + dir * offset
            if xt >= 0 and not env._check_collision(xt, z_target):
                return xt
    raise RuntimeError(f"Không tìm được điểm an toàn gần x={x_target}")

# ================================= MÔI TRƯỜNG =================================
class FinalDroneEnv(gym.Env):
    def __init__(self, target_x=100.0):
        super().__init__()
        self.drone = ImprovedDrone()
        self.target_x = target_x
        self.initial_dx = None
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        self.obstacles = []
        self.target_z = 5.0

    def reset(self, seed=None, options=None):
        self.drone.state = np.array([
            np.random.uniform(-1.0, 1.0),
            5.0 + np.random.uniform(-0.5, 0.5),
            np.random.uniform(-0.2, 0.2),
            0.0, 0.0, 0.0
        ])
        self.obstacles = self._generate_random_obstacles()
        self.target_x = find_safe_endpoint(self, 100.0, self.target_z)
        self.initial_dx = self.target_x - self.drone.state[0]
        return self._get_obs(), {}

    def _generate_random_obstacles(self):
        """Sinh 6-10 cặp chướng ngại vật kiểu Flappy Bird trong phạm vi 0-100m."""
        num_obstacles = np.random.randint(6, 11)
        obstacles = []
        min_spacing = 10.0
        x_positions = []
        x_current = 10.0
        while len(x_positions) < num_obstacles and x_current < 95.0:
            x_positions.append(x_current)
            x_current += np.random.uniform(min_spacing, min_spacing + 5.0)
        for x in x_positions:
            gap_center = np.random.uniform(4.0, 6.0)
            gap_size = 2.0
            x_width = np.random.uniform(1.0, 2.0)
            x_start = x
            x_end = min(x_start + x_width, 99.0)
            obstacles.append({
                "x": (x_start, x_end),
                "z": (0.0, gap_center - gap_size / 2)
            })
            obstacles.append({
                "x": (x_start, x_end),
                "z": (gap_center + gap_size / 2, 10.0)
            })
        return obstacles

    def _get_obs(self):
        x, z, theta, x_dot, z_dot, theta_dot = self.drone.state
        dx = self.target_x - x
        # Tìm chướng ngại vật gần nhất phía trước
        next_obstacle_x = 1000.0  # Giá trị mặc định nếu không có chướng ngại
        next_gap_center = 5.0     # Giá trị mặc định
        for obs in self.obstacles:
            obs_x = (obs["x"][0] + obs["x"][1]) / 2  # Tâm x của chướng ngại vật
            gap_center = (obs["z"][0] + obs["z"][1]) / 2 if obs["z"][0] == 0.0 else 10.0 - (obs["z"][1] - obs["z"][0]) / 2
            if obs_x > x:
                next_obstacle_x = obs_x
                next_gap_center = gap_center
                break
        return np.array([
            x, z, theta, x_dot, z_dot, theta_dot, dx,
            next_obstacle_x - x,  # Khoảng cách đến chướng ngại vật
            z - next_gap_center   # Chênh lệch độ cao đến tâm khoảng trống
        ], dtype=np.float32)

    def _check_collision(self, x, z):
        for obs in self.obstacles:
            if obs["x"][0] <= x <= obs["x"][1] and obs["z"][0] <= z <= obs["z"][1]:
                return True
        return False

    def step(self, action):
        tau = action[0] * 10.0
        state = self.drone.update(tau, dt=0.1)
        obs = self._get_obs()
        x, z = state[0], state[1]
        dx, vx, theta = obs[6], state[3], state[2]

        reward = -abs(z - 5.0) * 10.0  # Phạt lệch độ cao
        progress = (self.initial_dx - dx) / self.initial_dx
        reward += 150.0 * progress
        if dx * vx > 0:
            reward += 25.0 * vx * (1 + progress)
        else:
            reward -= 10.0 * abs(vx)
        reward -= 20.0 * abs(z - 5.0)  # Tăng phạt lệch độ cao
        if abs(z - 5.0) < 0.2:  # Thưởng khi giữ z rất gần 5.0
            reward += 50.0
        reward += 20.0 * vx if dx * vx > 0 else -5.0 * abs(vx)
        reward -= 0.5 * abs(dx) + 0.5 * abs(z - 5.0)
        if abs(dx) < 5.0:
            reward += 100.0 * (5.0 - abs(dx))

        terminated = False
        term_reason = None
        if abs(dx) < 0.5:
            reward += 2000
            terminated = True
            term_reason = "ĐẠT_MỤC_TIÊU"
        if self._check_collision(x, z):
            reward -= 5000
            terminated = True
            term_reason = "VA_CHẠM"
        if (z < 3.0) or (z > 8.0) or (abs(theta) > np.pi/2):
            terminated = True
            term_reason = "NGUY_HIỂM"
        if terminated:
            print(f"[Kết thúc tập] lý do={term_reason} | X={x:.2f}, Z={z:.2f}")
        return obs, reward, terminated, False, {}

# ==================== HUẤN LUYỆN 1 ĐOẠN ====================
def train_chunk(target_x, start_x, cont, load_path, save_path, log_dir):
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)

    env = Monitor(FinalDroneEnv(target_x), log_dir)
    cb1 = CheckpointCallback(save_freq=20_000, save_path=save_dir, name_prefix=os.path.basename(save_path))
    eval_dir = os.path.join(save_dir, 'best')
    os.makedirs(eval_dir, exist_ok=True)
    cb2 = EvalCallback(env, eval_freq=1_000, best_model_save_path=eval_dir, verbose=1)

    if cont and load_path and os.path.exists(load_path + '.zip'):
        model = SAC.load(load_path, env=env, verbose=1)
    else:
        model = SAC('MlpPolicy', env, learning_rate=3e-4, gamma=0.99, tau=0.005, verbose=1, tensorboard_log=log_dir)

    model.learn(total_timesteps=1_000_000, callback=[cb1, cb2], reset_num_timesteps=not cont)
    model.save(save_path)

    best_model_file = os.path.join(eval_dir, 'best_model.zip')
    if not os.path.exists(best_model_file):
        best_model_file = save_path + '.zip'
    return best_model_file

# ==================== VẼ ĐƯỜNG BAY ====================
def visualize_model(model_path, target_x):
    model = SAC.load(model_path)
    env = FinalDroneEnv(target_x=target_x)
    obs, _ = env.reset()
    traj, states = [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, _, _ = env.step(action)
        traj.append([obs[0], obs[1]])
        states.append(obs)
        done = terminated

    traj = np.array(traj)
    states = np.array(states)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(-1, target_x + 1)
    ax.set_ylim(2, 8)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title(f"Quỹ đạo đến {target_x}m (Chướng ngại vật kiểu Flappy Bird)")
    ax.grid(True)
    for obs_rect in env.obstacles:
        x0, x1 = obs_rect["x"]
        z0, z1 = obs_rect["z"]
        ax.add_patch(Rectangle((x0, z0), x1-x0, z1-z0, color='green', alpha=0.5))
    ax.plot(traj[:,0], traj[:,1], 'b-', lw=2, label='Đường bay')
    for (x, z, theta) in zip(traj[:,0], traj[:,1], states[:,2]):
        ax.add_patch(Arrow(x, z, np.cos(theta)*0.3, np.sin(theta)*0.3, width=0.1, color='orange', alpha=0.8))
    ax.plot(traj[0,0], traj[0,1], 'go', markersize=10, label='Điểm bắt đầu')
    final_z = env.drone.target_z
    ax.plot(target_x, final_z, 'ro', markersize=10, label='Mục tiêu')
    ax.legend()
    plt.show()

def mode_visualize(distance):
    target_x = distance
    model_path = "./models/best/best_model.zip"
    if os.path.exists(model_path):
        print(f"Hiển thị quỹ đạo từ best model: 0m đến {target_x}m")
        visualize_model(model_path, target_x)
    else:
        print(f"Không tìm thấy best model: {model_path}")
        model_path = "./models/chunk_1_model.zip"
        if os.path.exists(model_path):
            print(f"Hiển thị quỹ đạo từ mô hình cuối cùng: 0m đến {target_x}m")
            visualize_model(model_path, target_x)
        else:
            print(f"Không tìm thấy mô hình cuối cùng: {model_path}")

def mode_train(distance):
    start_x = 0
    target_x = distance
    save_path = "./models/chunk_1_model"
    log_dir = "./logs/"

    print(f"Huấn luyện: {start_x}m đến {target_x}m")
    train_chunk(target_x, start_x, cont=False, load_path=None, save_path=save_path, log_dir=log_dir)

# ==================== MAIN ====================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'visualize'], required=True)
    parser.add_argument('--distance', type=int, choices=[100], required=True)
    args = parser.parse_args()

    if args.mode == 'train':
        mode_train(args.distance)
    else:
        mode_visualize(args.distance)