# -*- coding: utf-8 -*-
"""
NTU RGB+D Skeleton 3D可视化工具
用于可视化.skeleton文件中的关节点和骨骼线
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import argparse


# NTU RGB+D 25个关节点定义
NTU_JOINT_NAMES = [
    'base',           # 0  - base of spine (脊柱基座)
    'mid_spine',      # 1  - spine (mid spine) (脊柱中部)
    'neck',           # 2  - neck (颈部)
    'head',           # 3  - head (头部)
    'left_shoulder',  # 4  - 左肩
    'left_elbow',     # 5  - 左肘
    'left_wrist',     # 6  - 左手腕
    'left_hand',      # 7  - 左手
    'right_shoulder', # 8  - 右肩
    'right_elbow',    # 9  - 右肘
    'right_wrist',    # 10 - 右手腕
    'right_hand',     # 11 - 右手
    'left_hip',       # 12 - 左髋
    'left_knee',      # 13 - 左膝
    'left_ankle',     # 14 - 左踝
    'left_foot',      # 15 - 左足
    'right_hip',      # 16 - 右髋
    'right_knee',     # 17 - 右膝
    'right_ankle',    # 18 - 右踝
    'right_foot',     # 19 - 右足
    'spine',          # 20 - 脊柱
    'left_hand_tip',  # 21 - 左手尖
    'left_thumb',     # 22 - 左拇指
    'right_hand_tip', # 23 - 右手尖
    'right_thumb',    # 24 - 右拇指
]

# NTU RGB+D 骨骼连接关系（基于visualize_skeleton.py中的定义）
# 格式: (关节1, 关节2)
NTU_SKELETON_CONNECTIONS = [
    # 躯干 body: 3-2-20-1-0
    (3, 2),
    (2, 20),
    (20, 1),
    (1, 0),
    
    # 手臂 arms: 23-11-10-9-8-20-4-5-6-7-21
    (23, 11),
    (11, 10),
    (10, 9),
    (9, 8),
    (8, 20),
    (20, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 21),
    
    # 右手 rightHand: 11-24
    (11, 24),
    
    # 左手 leftHand: 7-22
    (7, 22),
    
    # 腿部 legs: 19-18-17-16-0-12-13-14-15
    (19, 18),
    (18, 17),
    (17, 16),
    (16, 0),
    (0, 12),
    (12, 13),
    (13, 14),
    (14, 15),
]

# 按身体部位分组的骨骼连接（用于不同颜色显示）
SKELETON_GROUPS = {
    'body': [(3, 2), (2, 20), (20, 1), (1, 0)],
    'arms': [(23, 11), (11, 10), (10, 9), (9, 8), (8, 20), (20, 4), (4, 5), (5, 6), (6, 7), (7, 21)],
    'rightHand': [(11, 24)],
    'leftHand': [(7, 22)],
    'legs': [(19, 18), (18, 17), (17, 16), (16, 0), (0, 12), (12, 13), (13, 14), (14, 15)],
}

# 颜色配置
COLORS = {
    'joints': 'red',
    'body': 'blue',
    'arms': 'green',
    'rightHand': 'cyan',
    'leftHand': 'magenta',
    'legs': 'orange',
}


def read_skeleton_file(file_path):
    """
    读取.skeleton文件
    
    参数:
        file_path: .skeleton文件路径
    
    返回:
        skeleton_data: 包含所有帧数据的字典
    """
    with open(file_path, 'r') as f:
        skeleton_data = {}
        skeleton_data['num_frames'] = int(f.readline().strip())
        skeleton_data['frames'] = []
        
        for frame_idx in range(skeleton_data['num_frames']):
            frame_data = {}
            frame_data['num_bodies'] = int(f.readline().strip())
            frame_data['bodies'] = []
            
            for body_idx in range(frame_data['num_bodies']):
                body_data = {}
                body_line = f.readline().strip().split()
                body_data['body_id'] = float(body_line[0])
                
                num_joints = int(f.readline().strip())
                body_data['joints'] = []
                
                for joint_idx in range(num_joints):
                    joint_line = f.readline().strip().split()
                    joint_data = {
                        'x': float(joint_line[0]),
                        'y': float(joint_line[1]),
                        'z': float(joint_line[2]),
                    }
                    body_data['joints'].append(joint_data)
                
                frame_data['bodies'].append(body_data)
            
            skeleton_data['frames'].append(frame_data)
    
    return skeleton_data


def extract_xyz_coordinates(skeleton_data, max_body=2, num_joint=25):
    """
    提取x, y, z坐标
    
    参数:
        skeleton_data: read_skeleton_file返回的数据
        max_body: 最大人体数量
        num_joint: 关节点数量
    
    返回:
        data: shape (3, num_frames, num_joint, max_body)
    """
    num_frames = skeleton_data['num_frames']
    data = np.zeros((3, num_frames, num_joint, max_body))
    
    for frame_idx, frame in enumerate(skeleton_data['frames']):
        for body_idx, body in enumerate(frame['bodies']):
            if body_idx < max_body:
                for joint_idx, joint in enumerate(body['joints']):
                    if joint_idx < num_joint:
                        data[0, frame_idx, joint_idx, body_idx] = joint['x']
                        data[1, frame_idx, joint_idx, body_idx] = joint['y']
                        data[2, frame_idx, joint_idx, body_idx] = joint['z']
    
    return data


def visualize_single_frame_3d(ax, points, frame_idx, body_idx=0):
    """
    可视化单帧3D骨架
    
    参数:
        ax: matplotlib 3D轴
        points: 坐标数据 (3, num_frames, num_joint, max_body)
        frame_idx: 帧索引
        body_idx: 人体索引
    """
    ax.clear()
    
    x = points[0, frame_idx, :, body_idx]
    y = points[1, frame_idx, :, body_idx]
    z = points[2, frame_idx, :, body_idx]
    
    ax.scatter(x, y, z, c=COLORS['joints'], s=50, marker='o', label='Joints')
    
    for i in range(len(x)):
        ax.text(x[i], y[i], z[i], str(i), fontsize=8, color='blue')
    
    for group_name, connections in SKELETON_GROUPS.items():
        for (joint1, joint2) in connections:
            ax.plot([x[joint1], x[joint2]], 
                   [y[joint1], y[joint2]], 
                   [z[joint1], z[joint2]], 
                   c=COLORS[group_name], linewidth=2.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Frame: {frame_idx}')
    
    margin = 0.3
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    z_range = z.max() - z.min()
    
    ax.set_xlim(x.min() - margin * x_range, x.max() + margin * x_range)
    ax.set_ylim(y.min() - margin * y_range, y.max() + margin * y_range)
    ax.set_zlim(z.min() - margin * z_range, z.max() + margin * z_range)


def visualize_animation_3d(points, interval=50, body_idx=0, view_angle=(30, -60)):
    """
    3D动画可视化
    
    参数:
        points: 坐标数据 (3, num_frames, num_joint, max_body)
        interval: 帧间隔（毫秒）
        body_idx: 人体索引
        view_angle: 视角 (elevation, azimuth)
    """
    num_frames = points.shape[1]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    def update(frame_idx):
        visualize_single_frame_3d(ax, points, frame_idx, body_idx)
        ax.view_init(elev=view_angle[0], azim=view_angle[1])
    
    ani = FuncAnimation(fig, update, frames=num_frames, interval=interval, repeat=True)
    plt.show()
    
    return ani


def visualize_interactive_3d(points, body_idx=0):
    """
    交互式3D可视化（带键盘控制）
    
    参数:
        points: 坐标数据 (3, num_frames, num_joint, max_body)
        body_idx: 人体索引
    """
    num_frames = points.shape[1]
    current_frame = [0]
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    def update_plot():
        visualize_single_frame_3d(ax, points, current_frame[0], body_idx)
        ax.view_init(elev=30, azim=-60)
        fig.suptitle(f'Frame: {current_frame[0]}/{num_frames-1}\n'
                    f'Controls: ← → (navigate) | Space (play/pause) | Q (quit)', 
                    fontsize=10)
        plt.draw()
    
    def on_key(event):
        if event.key == 'right':
            current_frame[0] = min(current_frame[0] + 1, num_frames - 1)
            update_plot()
        elif event.key == 'left':
            current_frame[0] = max(current_frame[0] - 1, 0)
            update_plot()
        elif event.key == ' ':
            for i in range(current_frame[0], num_frames):
                current_frame[0] = i
                update_plot()
                plt.pause(0.05)
        elif event.key == 'q':
            plt.close(fig)
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    update_plot()
    plt.show()


def visualize_static_3d(points, frame_indices=None, body_idx=0, save_path=None):
    """
    静态3D可视化（显示多个关键帧）
    
    参数:
        points: 坐标数据 (3, num_frames, num_joint, max_body)
        frame_indices: 要显示的帧索引列表，如果为None则自动选择
        body_idx: 人体索引
        save_path: 保存路径，如果为None则显示
    """
    num_frames = points.shape[1]
    
    if frame_indices is None:
        frame_indices = [0, num_frames//4, num_frames//2, 3*num_frames//4, num_frames-1]
        frame_indices = [min(i, num_frames-1) for i in frame_indices]
    
    num_plots = len(frame_indices)
    cols = min(3, num_plots)
    rows = (num_plots + cols - 1) // cols
    
    fig = plt.figure(figsize=(5*cols, 4*rows))
    
    for idx, frame_idx in enumerate(frame_indices):
        ax = fig.add_subplot(rows, cols, idx+1, projection='3d')
        visualize_single_frame_3d(ax, points, frame_idx, body_idx)
        ax.view_init(elev=30, azim=-60)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    else:
        plt.show()


def print_skeleton_info(skeleton_data):
    """
    打印skeleton文件信息
    """
    print("\n" + "="*60)
    print("Skeleton文件信息")
    print("="*60)
    print(f"总帧数: {skeleton_data['num_frames']}")
    
    if skeleton_data['frames']:
        first_frame = skeleton_data['frames'][0]
        print(f"人体数量: {first_frame['num_bodies']}")
        
        for body_idx, body in enumerate(first_frame['bodies']):
            print(f"\n人体 {body_idx + 1}:")
            print(f"  Body ID: {body['body_id']}")
            print(f"  关节点数: {len(body['joints'])}")
            
            if body['joints']:
                joints = np.array([[j['x'], j['y'], j['z']] for j in body['joints']])
                print(f"  X范围: [{joints[:, 0].min():.3f}, {joints[:, 0].max():.3f}]")
                print(f"  Y范围: [{joints[:, 1].min():.3f}, {joints[:, 1].max():.3f}]")
                print(f"  Z范围: [{joints[:, 2].min():.3f}, {joints[:, 2].max():.3f}]")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='NTU RGB+D Skeleton 3D可视化工具')
    parser.add_argument('skeleton_file', type=str, help='Skeleton文件路径')
    parser.add_argument('--mode', type=str, default='interactive', 
                       choices=['animation', 'interactive', 'static'],
                       help='可视化模式: animation(动画), interactive(交互), static(静态)')
    parser.add_argument('--body', type=int, default=0, help='人体索引 (0或1)')
    parser.add_argument('--interval', type=int, default=50, help='动画帧间隔(毫秒)')
    parser.add_argument('--frames', type=int, nargs='+', default=None,
                       help='静态模式下要显示的帧索引')
    parser.add_argument('--save', type=str, default=None, help='保存图片路径')
    parser.add_argument('--info', action='store_true', help='打印文件信息')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.skeleton_file):
        print(f"错误: 文件不存在: {args.skeleton_file}")
        return
    
    print(f"\n正在读取文件: {args.skeleton_file}")
    skeleton_data = read_skeleton_file(args.skeleton_file)
    
    if args.info:
        print_skeleton_info(skeleton_data)
    
    print(f"\n提取坐标数据...")
    points = extract_xyz_coordinates(skeleton_data)
    
    print(f"数据形状: {points.shape}")
    print(f"  - 维度: {points.shape[0]} (X, Y, Z)")
    print(f"  - 帧数: {points.shape[1]}")
    print(f"  - 关节点: {points.shape[2]}")
    print(f"  - 人体数: {points.shape[3]}")
    
    print(f"\n启动{args.mode}模式可视化...")
    
    if args.mode == 'animation':
        visualize_animation_3d(points, interval=args.interval, body_idx=args.body)
    elif args.mode == 'interactive':
        visualize_interactive_3d(points, body_idx=args.body)
    elif args.mode == 'static':
        visualize_static_3d(points, frame_indices=args.frames, 
                           body_idx=args.body, save_path=args.save)


if __name__ == '__main__':
    main()
