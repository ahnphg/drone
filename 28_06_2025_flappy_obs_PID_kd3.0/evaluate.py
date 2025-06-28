"""
evaluate.py

– Đánh giá mô hình SAC từ sac.py với 50 lần thử nghiệm
– Sử dụng môi trường 100m với chướng ngại vật ngẫu nhiên
– Lưu ảnh quỹ đạo mỗi lần thử vào ./evaluation_plots/
– Ghi log kết quả và các chỉ số hiệu suất vào evaluation_log.txt

Chạy cho mô hình tốt nhất: python evaluate.py --model_path ./models/best/best_model.zip

Chạy cho mô hình cuối: python evaluate.py --model_path ./models/chunk_1_model.zip

"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow, Rectangle
from stable_baselines3 import SAC
import logging

# Đặt mã hóa UTF-8 cho console
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation_log.txt', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ================================= MÔI TRƯỜNG =================================
class FinalDroneEnv:
    def __init__(self, target_x=100.0):
        self.drone = ImprovedDrone()
        self.target_x = target_x
        self.initial_dx = None
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
        next_obstacle_x = 1000.0
        next_gap_center = 5.0
        for obs in self.obstacles:
            obs_x = (obs["x"][0] + obs["x"][1]) / 2
            gap_center = (obs["z"][0] + obs["z"][1]) / 2 if obs["z"][0] == 0.0 else 10.0 - (obs["z"][1] - obs["z"][0]) / 2
            if obs_x > x:
                next_obstacle_x = obs_x
                next_gap_center = gap_center
                break
        return np.array([
            x, z, theta, x_dot, z_dot, theta_dot, dx,
            next_obstacle_x - x,
            z - next_gap_center
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

        reward = -abs(z - 5.0) * 10.0
        progress = (self.initial_dx - dx) / self.initial_dx
        reward += 150.0 * progress
        if dx * vx > 0:
            reward += 25.0 * vx * (1 + progress)
        else:
            reward -= 10.0 * abs(vx)
        reward -= 20.0 * abs(z - 5.0)
        if abs(z - 5.0) < 0.2:
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

# ================================= MÔ HÌNH DRONE =================================
class ImprovedDrone:
    def __init__(self, target_z=5.0, kp=5.0, ki=0.05, kd=3.0):
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
        MAX_ANGLE = np.deg2rad(70)
        if abs(theta) > MAX_ANGLE:
            theta_ddot = -np.sign(theta) * 5.0
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

def visualize_model(model_path, target_x, test_id, save_dir="./evaluation_plots"):
    """
    Vẽ và lưu quỹ đạo của một lần thử, tự phát hiện lý do kết thúc.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model = SAC.load(model_path)
    env = FinalDroneEnv(target_x=target_x)
    obs, _ = env.reset()
    traj, states = [], []
    total_reward = 0.0
    done = False
    steps = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, _, _ = env.step(action)
        traj.append([obs[0], obs[1]])
        states.append(obs)
        total_reward += reward
        done = terminated
        steps += 1
    
    traj = np.array(traj)
    states = np.array(states)
    
    # Trích xuất giá trị cuối từ final_obs
    final_obs = states[-1]
    final_x, final_z, final_theta = final_obs[0], final_obs[1], final_obs[2]
    
    # Phát hiện lý do kết thúc
    x, z, theta, x_dot, z_dot, theta_dot, dx, _, _ = final_obs  # Cập nhật cho 9 chiều
    if abs(dx) < 0.5:
        term_reason = "ĐẠT_MỤC_TIÊU"
    elif env._check_collision(x, z):
        term_reason = "VA_CHẠM"
    elif z < 3.0 or z > 8.0 or abs(theta) > np.pi/2:
        term_reason = "NGUY_HIỂM"
    else:
        term_reason = "UNKNOWN"
    
    # Vẽ quỹ đạo
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(-1, target_x + 1)
    ax.set_ylim(2, 8)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title(f"Quỹ đạo thử nghiệm {test_id} đến {target_x}m (Lý do: {term_reason})")
    ax.grid(True)
    
    for obs_rect in env.obstacles:
        x0, x1 = obs_rect["x"]
        z0, z1 = obs_rect["z"]
        ax.add_patch(Rectangle((x0, z0), x1-x0, z1-z0, color='gray', alpha=0.5))
    
    ax.plot(traj[:,0], traj[:,1], 'b-', lw=2, label='Đường bay')
    for (x, z, theta) in zip(traj[:,0], traj[:,1], states[:,2]):
        ax.add_patch(Arrow(x, z, np.cos(theta)*0.3, np.sin(theta)*0.3, width=0.1, color='orange', alpha=0.8))
    
    ax.plot(traj[0,0], traj[0,1], 'go', markersize=10, label='Điểm bắt đầu')
    ax.plot(target_x, env.drone.target_z, 'ro', markersize=10, label='Mục tiêu')
    ax.legend()
    
    # Thêm chú thích lý do kết thúc
    ax.text(0.02, 0.98, f"Lý do: {term_reason}", transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    save_path = os.path.join(save_dir, f"test_{test_id}.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    
    return term_reason, final_x, final_z, total_reward, steps, final_theta

def evaluate_model(model_path, num_tests=50, distance=100.0):
    """
    Đánh giá mô hình với num_tests lần thử, tính các chỉ số hiệu suất.
    """
    logger.info(f"Bắt đầu đánh giá mô hình: {model_path}")
    logger.info(f"Số lần thử: {num_tests}, Quãng đường: {distance}m")
    
    results = []
    for test_id in range(1, num_tests + 1):
        logger.info(f"--- Thử nghiệm {test_id} ---")
        term_reason, final_x, final_z, total_reward, steps, final_theta = visualize_model(
            model_path, distance, test_id
        )
        log_message = (
            f"Thử nghiệm {test_id}: Lý do kết thúc={term_reason}, "
            f"Vị trí cuối=(X={final_x:.2f}, Z={final_z:.2f}), "
            f"Góc cuối={np.degrees(final_theta):.1f}°, "
            f"Phần thưởng={total_reward:.2f}, Số bước={steps}"
        )
        logger.info(log_message)
        results.append({
            'test_id': test_id,
            'term_reason': term_reason,
            'final_x': final_x,
            'final_z': final_z,
            'final_theta': final_theta,
            'total_reward': total_reward,
            'steps': steps
        })
    
    # Tính các chỉ số
    success_count = sum(1 for r in results if r['term_reason'] == "ĐẠT_MỤC_TIÊU")
    collision_count = sum(1 for r in results if r['term_reason'] == "VA_CHẠM")
    danger_count = sum(1 for r in results if r['term_reason'] == "NGUY_HIỂM")
    total_tests = len(results)
    
    success_rate = (success_count / total_tests) * 100
    collision_rate = (collision_count / total_tests) * 100
    danger_rate = (danger_count / total_tests) * 100
    avg_reward = np.mean([r['total_reward'] for r in results])
    reward_std = np.std([r['total_reward'] for r in results])
    avg_steps = np.mean([r['steps'] for r in results])
    avg_theta = np.mean([r['final_theta'] for r in results])
    
    # Tính khoảng cách trung bình đến mục tiêu cho các lần không thành công
    failed_results = [r for r in results if r['term_reason'] != "ĐẠT_MỤC_TIÊU"]
    avg_distance = np.mean([
        np.sqrt((r['final_x'] - distance)**2 + (r['final_z'] - 5)**2)
        for r in failed_results
    ]) if failed_results else 0.0
    
    # Ghi các chỉ số vào log
    logger.info(f"--- Tóm tắt hiệu suất ---")
    logger.info(f"Tỷ lệ thành công: {success_count}/{total_tests} ({success_rate:.1f}%)")
    logger.info(f"Tỷ lệ va chạm: {collision_count}/{total_tests} ({collision_rate:.1f}%)")
    logger.info(f"Tỷ lệ thất bại do nguy hiểm: {danger_count}/{total_tests} ({danger_rate:.1f}%)")
    logger.info(f"Phần thưởng trung bình: {avg_reward:.2f}")
    logger.info(f"Độ lệch chuẩn phần thưởng: {reward_std:.2f}")
    logger.info(f"Số bước trung bình: {avg_steps:.2f}")
    logger.info(f"Góc nghiêng trung bình cuối: {np.degrees(avg_theta):.1f}°")
    logger.info(f"Khoảng cách trung bình đến mục tiêu (thất bại): {avg_distance:.2f} m")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="./models/chunk_1_model.zip",
                        help="Đường dẫn đến mô hình SAC đã huấn luyện")
    parser.add_argument('--num_tests', type=int, default=50,
                        help="Số lần thử nghiệm")
    parser.add_argument('--distance', type=float, default=100.0,
                        help="Quãng đường mục tiêu (m)")
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        logger.error(f"Không tìm thấy mô hình: {args.model_path}")
        exit(1)
    
    evaluate_model(args.model_path, args.num_tests, args.distance)