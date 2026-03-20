# -*- coding: utf-8 -*-
"""
VIBE输出转换脚本
将VIBE模型输出的pkl文件转换为NTU RGB+D兼容的.skeleton格式
"""

import os
import sys
import joblib
import numpy as np
from pathlib import Path

# 添加lib目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))

# NTU RGB+D的25个关节点定义
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
    'right_wrist',    # 10  - 右手腕
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

# SMPL关节点名称（基于lib/data_utils/kp_utils.py中的get_spin_joint_names）
SMPL_JOINT_NAMES = [
    'OP Nose',        # 0  - 鼻子
    'OP Neck',        # 1  - 颈部
    'OP RShoulder',   # 2  - 右肩
    'OP RElbow',      # 3  - 右肘
    'OP RWrist',      # 4  - 右手腕
    'OP LShoulder',   # 5  - 左肩
    'OP LElbow',      # 6  - 左肘
    'OP LWrist',      # 7  - 左手腕
    'OP MidHip',      # 8  - 髋部中心
    'OP RHip',        # 9  - 右髋
    'OP RKnee',       # 10 - 右膝
    'OP RAnkle',      # 11 - 右踝
    'OP LHip',        # 12 - 左髋
    'OP LKnee',       # 13 - 左膝
    'OP LAnkle',      # 14 - 左踝
    'OP REye',        # 15 - 右眼
    'OP LEye',        # 16 - 左眼
    'OP REar',        # 17 - 右耳
    'OP LEar',        # 18 - 左耳
    'OP LBigToe',     # 19 - 左大脚趾
    'OP LSmallToe',   # 20 - 左小脚趾
    'OP LHeel',       # 21 - 左脚后跟
    'OP RBigToe',     # 22 - 右大脚趾
    'OP RSmallToe',   # 23 - 右小脚趾
    'OP RHeel',       # 24 - 右脚后跟
    'rankle',         # 25
    'rknee',          # 26
    'rhip',           # 27
    'lhip',           # 28
    'lknee',          # 29
    'lankle',         # 30
    'rwrist',         # 31
    'relbow',         # 32
    'rshoulder',      # 33
    'lshoulder',      # 34
    'lelbow',         # 35
    'lwrist',         # 36
    'neck',           # 37
    'headtop',        # 38 - 头顶
    'hip',            # 39 - 髋部
    'thorax',         # 40 - 胸部
    'Spine (H36M)',   # 41 - 脊柱
    'Jaw (H36M)',     # 42 - 下巴
    'Head (H36M)',    # 43 - 头部
    'nose',           # 44
    'leye',           # 45
    'reye',           # 46
    'lear',           # 47
    'rear',           # 48
]

# SMPL到NTU RGB+D的关节点映射关系
SMPL_TO_NTU_MAPPING = {
    # 头部
    0: 3,    # OP Nose -> head
    1: 2,    # OP Neck -> neck
    38: 3,   # headtop -> head
    
    # 躯干
    39: 0,   # hip -> base
    41: 1,   # Spine (H36M) -> mid_spine (脊柱中部)
    40: 20,  # thorax -> spine
    
    # 右上肢
    2: 8,    # OP RShoulder -> right_shoulder
    3: 9,    # OP RElbow -> right_elbow
    4: 10,   # OP RWrist -> right_wrist
    33: 8,   # rshoulder -> right_shoulder
    32: 9,   # relbow -> right_elbow
    31: 10,  # rwrist -> right_wrist
    
    # 左上肢
    5: 4,    # OP LShoulder -> left_shoulder
    6: 5,    # OP LElbow -> left_elbow
    7: 6,    # OP LWrist -> left_wrist
    34: 4,   # lshoulder -> left_shoulder
    35: 5,   # lelbow -> left_elbow
    36: 6,   # lwrist -> left_wrist
    
    # 右下肢
    9: 16,   # OP RHip -> right_hip
    10: 17,  # OP RKnee -> right_knee
    11: 18,  # OP RAnkle -> right_ankle
    27: 16,  # rhip -> right_hip
    26: 17,  # rknee -> right_knee
    25: 18,  # rankle -> right_ankle
    
    # 左下肢
    12: 12,  # OP LHip -> left_hip
    13: 13,  # OP LKnee -> left_knee
    14: 14,  # OP LAnkle -> left_ankle
    28: 12,  # lhip -> left_hip
    29: 13,  # lknee -> left_knee
    30: 14,  # lankle -> left_ankle
}


def normalize_skeleton_sequence(joints, verbose=True):
    """
    对骨架序列进行坐标统一化
    参考 CTR_GCN/data/ntu/seq_transformation.py 的 seq_translation 函数
    
    处理步骤:
    1. 找到第一个有效帧（非全零帧）
    2. 以该帧的关节点2（neck，索引2）作为新原点
    3. 将所有帧的坐标减去这个原点，实现坐标统一化
    4. 翻转Y轴坐标，使人物模型从脚到头沿Y轴正方向分布
    
    参数:
        joints: 关节点坐标 (num_frames, 25, 3)
        verbose: 是否打印详细信息
    
    返回:
        归一化后的关节点坐标 (num_frames, 25, 3)
    """
    num_frames = joints.shape[0]
    
    # 找到第一个有效帧（关节点不全为零）
    i = 0
    while i < num_frames:
        if np.any(joints[i] != 0):
            break
        i += 1
    
    if i >= num_frames:
        if verbose:
            print("  警告: 所有帧都无效，跳过坐标统一化")
        return joints
    
    # 以第一个有效帧的关节点2（neck）作为新原点
    # NTU关节点索引2 = neck (颈部)
    origin = np.copy(joints[i, 2, :])
    
    if verbose:
        print(f"  坐标统一化: 以帧{i}的neck关节点为原点")
        print(f"    原点坐标: ({origin[0]:.4f}, {origin[1]:.4f}, {origin[2]:.4f})")
    
    # 将所有帧的坐标减去原点
    for f in range(num_frames):
        joints[f] = joints[f] - origin
    
    # 翻转Y轴坐标（索引1），使人物模型从脚到头沿Y轴正方向分布
    # VIBE输出的Y轴方向与NTU RGB+D相反，需要翻转
    joints[:, :, 1] = -joints[:, :, 1]
    
    # 翻转Z轴坐标（索引2），使人物正面朝向Z轴正方向
    # VIBE输出的Z轴方向与NTU RGB+D相反，需要翻转
    joints[:, :, 2] = -joints[:, :, 2]
    
    if verbose:
        print(f"  Y轴翻转: 已将Y轴坐标取反，人物模型现在沿Y轴正方向分布")
        print(f"  Z轴翻转: 已将Z轴坐标取反，人物正面现在朝向Z轴正方向")
    
    return joints


def convert_vibe_to_ntu_skeleton(vibe_pkl_path, output_dir, action_id=1, setup_id=1, 
                                 camera_id=1, performer_id=1, repetition_id=1, verbose=True):
    """
    将VIBE输出的pkl文件转换为NTU RGB+D兼容的.skeleton格式
    
    参数:
        vibe_pkl_path: VIBE输出的pkl文件路径
        output_dir: 输出目录
        action_id: 动作ID (1-60)
        setup_id: 设置ID (1-17)
        camera_id: 相机ID (1-3)
        performer_id: 表演者ID (1-40)
        repetition_id: 重复ID (1-3)
        verbose: 是否打印详细信息
    
    返回:
        生成的.skeleton文件路径列表
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"正在加载VIBE输出文件: {vibe_pkl_path}")
        print(f"{'='*80}")
    
    # 加载VIBE输出
    vibe_results = joblib.load(vibe_pkl_path)
    
    if verbose:
        print(f"检测到 {len(vibe_results)} 个人物")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    skeleton_files = []
    
    # 为每个人物生成.skeleton文件
    for person_id, person_data in vibe_results.items():
        if verbose:
            print(f"\n处理人物 {person_id}...")
        
        # 获取3D关节点 (N, 49, 3)
        joints3d = person_data['joints3d']
        num_frames = joints3d.shape[0]
        
        if verbose:
            print(f"  帧数: {num_frames}")
            print(f"  VIBE关节点形状: {joints3d.shape}")
        
        # 转换为NTU RGB+D格式 (T, 25, 3)
        ntu_joints = np.zeros((num_frames, 25, 3), dtype=np.float32)
        
        # 映射关节点
        for smpl_idx, ntu_idx in SMPL_TO_NTU_MAPPING.items():
            if smpl_idx < joints3d.shape[1]:
                ntu_joints[:, ntu_idx, :] = joints3d[:, smpl_idx, :]
        
        # 备用：如果 Spine (H36M) (41) 不存在，使用基于距离的插值计算 mid_spine (1)
        if 41 >= joints3d.shape[1]:
            if 39 < joints3d.shape[1] and 40 < joints3d.shape[1]:
                base_pos = joints3d[:, 39, :]   # hip
                spine_pos = joints3d[:, 40, :]  # thorax
                # 计算两点之间的向量（不依赖坐标正负）
                vec = base_pos - spine_pos
                # mid_spine 在 spine 和 base 之间，距离 spine 约 1/3
                ntu_joints[:, 1, :] = spine_pos + vec * 0.33
        
        # 对于缺失的关节点，使用相邻关节点插值或估算
        # 左手相关关节点 (7, 21, 22)
        # left_hand (7) - 在left_wrist(6)和left_hand_tip(21)之间
        # SMPL索引: 7=OP LWrist(左手腕), 6=OP LElbow(左肘)
        if 7 < joints3d.shape[1]:
            wrist_pos = joints3d[:, 7, :]  # SMPL索引7 = OP LWrist (左手腕)
            elbow_pos = joints3d[:, 6, :] if 6 < joints3d.shape[1] else wrist_pos  # SMPL索引6 = OP LElbow (左肘)
            direction = wrist_pos - elbow_pos
            # hand 在 wrist 沿手臂方向延伸
            ntu_joints[:, 7, :] = wrist_pos + direction * 0.12
            # hand_tip 从 hand 继续沿手臂方向延伸
            ntu_joints[:, 21, :] = wrist_pos + direction * 0.25
            # thumb 也沿手臂方向延伸，距离稍短
            ntu_joints[:, 22, :] = wrist_pos + direction * 0.18
        
        # 右手相关关节点 (11, 23, 24)
        # right_hand (11) - 在right_wrist(10)和right_hand_tip(23)之间
        # SMPL索引: 4=OP RWrist(右手腕), 3=OP RElbow(右肘)
        if 4 < joints3d.shape[1]:
            wrist_pos = joints3d[:, 4, :]  # SMPL索引4 = OP RWrist (右手腕)
            elbow_pos = joints3d[:, 3, :] if 3 < joints3d.shape[1] else wrist_pos  # SMPL索引3 = OP RElbow (右肘)
            direction = wrist_pos - elbow_pos
            # hand 在 wrist 沿手臂方向延伸
            ntu_joints[:, 11, :] = wrist_pos + direction * 0.12
            # hand_tip 从 hand 继续沿手臂方向延伸
            ntu_joints[:, 23, :] = wrist_pos + direction * 0.25
            # thumb 也沿手臂方向延伸，距离稍短
            ntu_joints[:, 24, :] = wrist_pos + direction * 0.18
        
        # 左脚相关关节点 (15)
        # left_foot (15) - 使用SMPL的左脚趾关节点
        if 19 < joints3d.shape[1]:  # OP LBigToe
            ntu_joints[:, 15, :] = joints3d[:, 19, :]
        elif 20 < joints3d.shape[1]:  # OP LSmallToe
            ntu_joints[:, 15, :] = joints3d[:, 20, :]
        elif 14 < joints3d.shape[1]:  # 备用：从ankle延伸
            ankle_pos = joints3d[:, 14, :]
            knee_pos = joints3d[:, 13, :] if 13 < joints3d.shape[1] else ankle_pos
            direction = ankle_pos - knee_pos
            ntu_joints[:, 15, :] = ankle_pos + direction * 0.15
        
        # 右脚相关关节点 (19)
        # right_foot (19) - 使用SMPL的右脚趾关节点
        if 22 < joints3d.shape[1]:  # OP RBigToe
            ntu_joints[:, 19, :] = joints3d[:, 22, :]
        elif 23 < joints3d.shape[1]:  # OP RSmallToe
            ntu_joints[:, 19, :] = joints3d[:, 23, :]
        elif 11 < joints3d.shape[1]:  # 备用：从ankle延伸
            ankle_pos = joints3d[:, 11, :]
            knee_pos = joints3d[:, 10, :] if 10 < joints3d.shape[1] else ankle_pos
            direction = ankle_pos - knee_pos
            ntu_joints[:, 19, :] = ankle_pos + direction * 0.15
        
        if verbose:
            print(f"  NTU关节点形状: {ntu_joints.shape}")
        
        # 坐标统一化
        ntu_joints = normalize_skeleton_sequence(ntu_joints, verbose)
        
        # 生成NTU格式的.skeleton文件
        skeleton_filename = f"S{setup_id:03d}C{camera_id:03d}P{performer_id:03d}R{repetition_id:03d}A{action_id:03d}_person{person_id}.skeleton"
        skeleton_path = os.path.join(output_dir, skeleton_filename)
        
        write_ntu_skeleton_file(skeleton_path, ntu_joints, person_id, verbose)
        skeleton_files.append(skeleton_path)
        
        if verbose:
            print(f"  已保存: {skeleton_path}")
    
    return skeleton_files


def write_ntu_skeleton_file(skeleton_path, joints, body_id, verbose=True):
    """
    写入NTU RGB+D格式的.skeleton文件
    
    参数:
        skeleton_path: 输出文件路径
        joints: 关节点坐标 (num_frames, 25, 3)
        body_id: 人体ID
        verbose: 是否打印详细信息
    """
    num_frames = joints.shape[0]
    
    with open(skeleton_path, 'w') as f:
        # 第一行：总帧数
        f.write(f"{num_frames}\n")
        
        for frame_idx in range(num_frames):
            # 写入人数（1个人）
            f.write("1\n")
            
            # 写入bodyID和元数据
            # 格式：bodyID confidence depth_x depth_y depth_z frame_id
            body_id_float = float(body_id)
            confidence = 1.0
            depth_x = 0.0
            depth_y = 0.0
            depth_z = 0.0
            frame_id = frame_idx
            
            f.write(f"{body_id_float} {confidence} {depth_x} {depth_y} {depth_z} {frame_id}\n")
            
            # 写入关节数量（25个）
            f.write("25\n")
            
            # 写入25个关节点
            for joint_idx in range(25):
                x, y, z = joints[frame_idx, joint_idx, :]
                
                # NTU格式：x y z depth_x depth_y confidence depth_z confidence
                # 这里简化处理，depth_x和depth_y设为0，confidence设为1
                f.write(f"{x:.6f} {y:.6f} {z:.6f} 0 0 1 0\n")


def convert_multiple_vibe_to_ntu_skeleton(vibe_output_dir, output_dir, 
                                         action_id_start=1, setup_id=1, 
                                         camera_id=1, performer_id_start=1, 
                                         repetition_id=1, verbose=True):
    """
    批量转换VIBE输出目录中的所有pkl文件
    
    参数:
        vibe_output_dir: VIBE输出目录（包含多个子文件夹）
        output_dir: 输出目录
        action_id_start: 起始动作ID
        setup_id: 设置ID
        camera_id: 相机ID
        performer_id_start: 起始表演者ID
        repetition_id: 重复ID
        verbose: 是否打印详细信息
    
    返回:
        生成的所有.skeleton文件路径列表
    """
    vibe_output_dir = Path(vibe_output_dir)
    output_dir = Path(output_dir)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有vibe_output.pkl文件
    pkl_files = list(vibe_output_dir.glob("*/vibe_output.pkl"))
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"批量转换VIBE输出文件")
        print(f"{'='*80}")
        print(f"找到 {len(pkl_files)} 个VIBE输出文件")
    
    all_skeleton_files = []
    
    for idx, pkl_file in enumerate(pkl_files):
        # 使用文件夹名称作为标识
        folder_name = pkl_file.parent.name
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"处理文件 {idx+1}/{len(pkl_files)}: {folder_name}")
            print(f"{'='*80}")
        
        # 为每个文件生成不同的ID
        current_action_id = action_id_start + idx
        current_performer_id = performer_id_start + idx
        
        skeleton_files = convert_vibe_to_ntu_skeleton(
            str(pkl_file),
            str(output_dir),
            action_id=current_action_id,
            setup_id=setup_id,
            camera_id=camera_id,
            performer_id=current_performer_id,
            repetition_id=repetition_id,
            verbose=verbose
        )
        
        all_skeleton_files.extend(skeleton_files)
    
    return all_skeleton_files


def create_ntu_metadata_files(output_dir, num_files, verbose=True):
    """
    创建NTU RGB+D所需的元数据文件
    
    参数:
        output_dir: 输出目录
        num_files: 文件数量
        verbose: 是否打印详细信息
    """
    stat_dir = os.path.join(output_dir, 'statistics')
    os.makedirs(stat_dir, exist_ok=True)
    
    # 创建setup.txt
    setup_file = os.path.join(stat_dir, 'setup.txt')
    with open(setup_file, 'w') as f:
        for i in range(1, num_files + 1):
            f.write(f"{i}\n")
    
    # 创建camera.txt
    camera_file = os.path.join(stat_dir, 'camera.txt')
    with open(camera_file, 'w') as f:
        for i in range(1, num_files + 1):
            f.write(f"1\n")
    
    # 创建performer.txt
    performer_file = os.path.join(stat_dir, 'performer.txt')
    with open(performer_file, 'w') as f:
        for i in range(1, num_files + 1):
            f.write(f"{i}\n")
    
    # 创建replication.txt
    replication_file = os.path.join(stat_dir, 'replication.txt')
    with open(replication_file, 'w') as f:
        for i in range(1, num_files + 1):
            f.write(f"1\n")
    
    # 创建label.txt
    label_file = os.path.join(stat_dir, 'label.txt')
    with open(label_file, 'w') as f:
        for i in range(1, num_files + 1):
            f.write(f"1\n")
    
    # 创建skes_available_name.txt
    skes_name_file = os.path.join(stat_dir, 'skes_available_name.txt')
    skeleton_files = list(Path(output_dir).glob("*.skeleton"))
    skeleton_names = [f.stem for f in skeleton_files]
    with open(skes_name_file, 'w') as f:
        for name in sorted(skeleton_names):
            f.write(f"{name}\n")
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"已创建元数据文件到: {stat_dir}")
        print(f"{'='*80}")
        print(f"  - setup.txt: {num_files} 行")
        print(f"  - camera.txt: {num_files} 行")
        print(f"  - performer.txt: {num_files} 行")
        print(f"  - replication.txt: {num_files} 行")
        print(f"  - label.txt: {num_files} 行")
        print(f"  - skes_available_name.txt: {len(skeleton_names)} 个文件名")


def main():
    """
    主函数 - 提供交互式使用方式
    """
    print("\n" + "="*80)
    print("VIBE输出转换工具 - 将pkl文件转换为NTU RGB+D兼容的.skeleton格式")
    print("="*80)
    
    print("\n使用方式:")
    print("1. 转换单个pkl文件")
    print("2. 批量转换output目录下的所有pkl文件")
    print("3. 退出")
    
    choice = input("\n请选择操作 (1/2/3): ").strip()
    
    if choice == '1':
        # 转换单个文件
        vibe_pkl_path = input("请输入VIBE输出的pkl文件路径: ").strip()
        output_dir = input("请输入输出目录 (默认: ./converted_ntu_data): ").strip()
        
        if not output_dir:
            output_dir = './converted_ntu_data'
        
        action_id = int(input("请输入动作ID (1-60, 默认: 1): ").strip() or "1")
        setup_id = int(input("请输入设置ID (1-17, 默认: 1): ").strip() or "1")
        camera_id = int(input("请输入相机ID (1-3, 默认: 1): ").strip() or "1")
        performer_id = int(input("请输入表演者ID (1-40, 默认: 1): ").strip() or "1")
        repetition_id = int(input("请输入重复ID (1-3, 默认: 1): ").strip() or "1")
        
        convert_vibe_to_ntu_skeleton(
            vibe_pkl_path,
            output_dir,
            action_id=action_id,
            setup_id=setup_id,
            camera_id=camera_id,
            performer_id=performer_id,
            repetition_id=repetition_id
        )
        
        # 创建元数据文件
        skeleton_files = list(Path(output_dir).glob("*.skeleton"))
        create_ntu_metadata_files(output_dir, len(skeleton_files))
        
    elif choice == '2':
        # 批量转换
        vibe_output_dir = input("请输入VIBE输出目录 (默认: ./output): ").strip()
        output_dir = input("请输入输出目录 (默认: ./converted_ntu_data): ").strip()
        
        if not vibe_output_dir:
            vibe_output_dir = './output'
        if not output_dir:
            output_dir = './converted_ntu_data'
        
        convert_multiple_vibe_to_ntu_skeleton(vibe_output_dir, output_dir)
        
        # 创建元数据文件
        skeleton_files = list(Path(output_dir).glob("*.skeleton"))
        create_ntu_metadata_files(output_dir, len(skeleton_files))
        
    elif choice == '3':
        print("退出程序")
        return
    else:
        print("无效的选择")
        return
    
    print("\n" + "="*80)
    print("转换完成！")
    print("="*80)
    print("\n生成的文件结构:")
    print(f"{output_dir}/")
    print("  ├── *.skeleton          # NTU格式的骨架文件")
    print("  └── statistics/")
    print("      ├── setup.txt        # 设置ID")
    print("      ├── camera.txt       # 相机ID")
    print("      ├── performer.txt    # 表演者ID")
    print("      ├── replication.txt  # 重复ID")
    print("      ├── label.txt        # 动作标签")
    print("      └── skes_available_name.txt  # 可用的骨架文件名")
    
    print("\n下一步操作:")
    print("1. 将生成的.skeleton文件复制到 CTR_GCN/data/nturgbd_raw/nturgb+d_skeletons/")
    print("2. 将statistics目录复制到 CTR_GCN/data/ntu/")
    print("3. 运行 CTR_GCN/data/ntu/get_raw_skes_data.py 处理原始数据")
    print("4. 运行 CTR_GCN/data/ntu/get_raw_denoised_data.py 进行去噪")
    print("5. 运行 CTR_GCN/data/ntu/seq_transformation.py 进行对齐和划分")


if __name__ == '__main__':
    # 如果直接运行脚本，提供交互式界面
    if len(sys.argv) == 1:
        main()
    else:
        # 如果提供了命令行参数，使用非交互式模式
        import argparse
        
        parser = argparse.ArgumentParser(description='将VIBE输出转换为NTU RGB+D格式')
        parser.add_argument('--input', type=str, help='输入pkl文件或目录')
        parser.add_argument('--output', type=str, default='./converted_ntu_data', help='输出目录')
        parser.add_argument('--action_id', type=int, default=1, help='动作ID')
        parser.add_argument('--setup_id', type=int, default=1, help='设置ID')
        parser.add_argument('--camera_id', type=int, default=1, help='相机ID')
        parser.add_argument('--performer_id', type=int, default=1, help='表演者ID')
        parser.add_argument('--repetition_id', type=int, default=1, help='重复ID')
        parser.add_argument('--batch', action='store_true', help='批量转换模式')
        
        args = parser.parse_args()
        
        if args.batch:
            convert_multiple_vibe_to_ntu_skeleton(
                args.input,
                args.output,
                action_id_start=args.action_id,
                setup_id=args.setup_id,
                camera_id=args.camera_id,
                performer_id_start=args.performer_id,
                repetition_id=args.repetition_id
            )
        else:
            convert_vibe_to_ntu_skeleton(
                args.input,
                args.output,
                action_id=args.action_id,
                setup_id=args.setup_id,
                camera_id=args.camera_id,
                performer_id=args.performer_id,
                repetition_id=args.repetition_id
            )
        
        # 创建元数据文件
        skeleton_files = list(Path(args.output).glob("*.skeleton"))
        create_ntu_metadata_files(args.output, len(skeleton_files))
