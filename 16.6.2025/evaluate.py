#!/usr/bin/env python3
"""
evaluate.py

– Đánh giá mô hình SAC từ sac.py với 50 lần thử nghiệm
– Sử dụng môi trường 50m với chướng ngại vật ngẫu nhiên
– Lưu ảnh quỹ đạo mỗi lần thử vào ./evaluation_plots/
– Ghi log kết quả và các chỉ số hiệu suất vào evaluation_log.txt
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow, Rectangle
from stable_baselines3 import SAC
import logging
from random_obs import FinalDroneEnv  # Nhập lớp FinalDroneEnv từ sac.py

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
    
    # Phát hiện lý do kết thúc
    final_obs = states[-1]
    x, z, theta, x_dot, z_dot, theta_dot, dx = final_obs
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
    ax.set_ylim(2, 7)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title(f"Quỹ đạo thử nghiệm {test_id} đến {target_x}m")
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
    
    save_path = os.path.join(save_dir, f"test_{test_id}.png")
    plt.savefig(save_path)
    plt.close(fig)
    
    final_x, final_z = traj[-1, 0], traj[-1, 1]
    return term_reason, final_x, final_z, total_reward, steps

def evaluate_model(model_path, num_tests=50, distance=50.0):
    """
    Đánh giá mô hình với num_tests lần thử, tính các chỉ số hiệu suất.
    """
    logger.info(f"Bắt đầu đánh giá mô hình: {model_path}")
    logger.info(f"Số lần thử: {num_tests}, Quãng đường: {distance}m")
    
    results = []
    for test_id in range(1, num_tests + 1):
        logger.info(f"--- Thử nghiệm {test_id} ---")
        term_reason, final_x, final_z, total_reward, steps = visualize_model(
            model_path, distance, test_id
        )
        log_message = (
            f"Thử nghiệm {test_id}: Lý do kết thúc={term_reason}, "
            f"Vị trí cuối=(X={final_x:.2f}, Z={final_z:.2f}), "
            f"Phần thưởng={total_reward:.2f}, Số bước={steps}"
        )
        logger.info(log_message)
        results.append({
            'test_id': test_id,
            'term_reason': term_reason,
            'final_x': final_x,
            'final_z': final_z,
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
    
    # Tính khoảng cách trung bình đến mục tiêu cho các lần không thành công
    failed_results = [r for r in results if r['term_reason'] != "ĐẠT_MỤC_TIÊU"]
    avg_distance = np.mean([
        np.sqrt((r['final_x'] - 50)**2 + (r['final_z'] - 5)**2)
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
    logger.info(f"Khoảng cách trung bình đến mục tiêu (thất bại): {avg_distance:.2f} m")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="./models/best/best_model.zip",
                        help="Đường dẫn đến mô hình SAC đã huấn luyện")
    parser.add_argument('--num_tests', type=int, default=50,
                        help="Số lần thử nghiệm")
    parser.add_argument('--distance', type=float, default=50.0,
                        help="Quãng đường mục tiêu (m)")
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        logger.error(f"Không tìm thấy mô hình: {args.model_path}")
        exit(1)
    
    evaluate_model(args.model_path, args.num_tests, args.distance)