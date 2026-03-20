## 导入第三方库
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import argparse
 

## 读取关节数据
def read_skeleton(file):
    with open(file, 'r') as f: # 打开file(.skeleton)文件
        skeleton_sequence = {} # 初始化skeleton_sequence
        skeleton_sequence['numFrame'] = int(f.readline()) # 读取.skeleton文件第一行，即帧数
        skeleton_sequence['frameInfo'] = []
        
        for t in range(skeleton_sequence['numFrame']): # 遍历每一帧
            frame_info = {} # 初始化frame_info
            frame_info['numBody'] = int(f.readline()) # 再次调用.readline函数，读取.skeleton文件的下一行，即body数
            frame_info['bodyInfo'] = []
            
            for m in range(frame_info['numBody']): # 遍历每一个body
                body_info = {} # 初始化body_info
                body_info_key = [ # key: 数字表示的意义，即对应的key
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v) # 字典类型; key: value(float类型)
                    for k, v in zip(body_info_key, f.readline().split()) # 读取下一行数据，根据key打包数据，遍历返回key, value
                }
                
                body_info['numJoint'] = int(f.readline()) # 读取下一行数据，即关节数
                body_info['jointInfo'] = []
                
                for v in range(body_info['numJoint']): # 遍历25个关节的数据
                    joint_info_key = [ # Key: 数字表示的意义，即对应的key
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v) # 字典类型; key: value(float类型)
                        for k, v in zip(joint_info_key, f.readline().split()) # 读取下一行数据，根据key打包数据，遍历返回key, value
                    }
                    body_info['jointInfo'].append(joint_info) # 保存关节数据
                
                frame_info['bodyInfo'].append(body_info) # 保存body数据
            skeleton_sequence['frameInfo'].append(frame_info) # 保存当前帧的数据
    return skeleton_sequence
 
 
## 读取关节的x，y，z三个坐标
def read_xyz(file, max_body=2, num_joint=25):
    seq_info = read_skeleton(file) # 调用read_skeleton()函数读取.skeleton文件的数据
    
    data = np.zeros((3, seq_info['numFrame'], num_joint, max_body)) # 初始化数据； 3 × 帧数 × 25 × max_body  
    for n, f in enumerate(seq_info['frameInfo']): # 遍历每一帧的数据
        for m, b in enumerate(f['bodyInfo']): # 遍历每一个body的数据
            for j, v in enumerate(b['jointInfo']): # 遍历每一个关节的数据
                if m < max_body and j < num_joint:
                    data[:, n, j, m] = [v['x'], v['y'], v['z']] # 保存 x,y,z三个坐标的数据
                else:
                    pass
    return data
 
## 3D展示    
def Print3D(num_frame, point, arms, rightHand, leftHand, legs, body, speed=1.0):

    # 求坐标最大值
    xmax = np.max(point[0, :, :, :])
    xmin = np.min(point[0, :, :, :]) 
    ymax = np.max(point[1, :, :, :])
    ymin = np.min(point[1, :, :, :])
    zmax = np.max(point[2, :, :, :])
    zmin = np.min(point[2, :, :, :])    
    
    current_frame = [0]
    is_playing = [False]
    current_speed = [speed]
    
    fig = plt.figure(figsize=(12, 9))
    plot3D = fig.add_subplot(111, projection='3d')
    
    def update_plot():
        plot3D.clear()
        plot3D.view_init(120, -90)
        
        Expan_Multiple = 1.4
        
        num_joint = point.shape[2]
        num_body = point.shape[3]
        i = current_frame[0]
        
        for body_idx in range(num_body):
            plot3D.scatter(point[0, i, :, body_idx]*Expan_Multiple, 
                          point[1, i, :, body_idx]*Expan_Multiple, 
                          point[2, i, :, body_idx], c='red', s=40.0)
            
            for j in range(num_joint):
                x = point[0, i, j, body_idx] * Expan_Multiple
                y = point[1, i, j, body_idx] * Expan_Multiple
                z = point[2, i, j, body_idx]
                plot3D.text(x, y, z, str(j), fontsize=8, color='blue')

        plot3D.plot(point[0, i, arms, 0]*Expan_Multiple, point[1, i, arms, 0]*Expan_Multiple, point[2, i, arms, 0], c='green', lw=2.0)
        plot3D.plot(point[0, i, rightHand, 0]*Expan_Multiple, point[1, i, rightHand, 0]*Expan_Multiple, point[2, i, rightHand, 0], c='green', lw=2.0)
        plot3D.plot(point[0, i, leftHand, 0]*Expan_Multiple, point[1, i, leftHand, 0]*Expan_Multiple, point[2, i, leftHand, 0], c='green', lw=2.0)
        plot3D.plot(point[0, i, legs, 0]*Expan_Multiple, point[1, i, legs, 0]*Expan_Multiple, point[2, i, legs, 0], c='green', lw=2.0)
        plot3D.plot(point[0, i, body, 0]*Expan_Multiple, point[1, i, body, 0]*Expan_Multiple, point[2, i, body, 0], c='green', lw=2.0)

        plot3D.plot(point[0, i, arms, 1]*Expan_Multiple, point[1, i, arms, 1]*Expan_Multiple, point[2, i, arms, 1], c='green', lw=2.0)
        plot3D.plot(point[0, i, rightHand, 1]*Expan_Multiple, point[1, i, rightHand, 1]*Expan_Multiple, point[2, i, rightHand, 1], c='green', lw=2.0)
        plot3D.plot(point[0, i, leftHand, 1]*Expan_Multiple, point[1, i, leftHand, 1]*Expan_Multiple, point[2, i, leftHand, 1], c='green', lw=2.0)
        plot3D.plot(point[0, i, legs, 1]*Expan_Multiple, point[1, i, legs, 1]*Expan_Multiple, point[2, i, legs, 1], c='green', lw=2.0)
        plot3D.plot(point[0, i, body, 1]*Expan_Multiple, point[1, i, body, 1]*Expan_Multiple, point[2, i, body, 1], c='green', lw=2.0)
 
        plot3D.set_xlabel('X')
        plot3D.set_ylabel('Y')
        plot3D.set_zlabel('Z')
        plot3D.set_xlim3d(xmin-0.5, xmax+0.5)
        plot3D.set_ylim3d(ymin-0.3, ymax+0.3)
        plot3D.set_zlim3d(zmin-0.3, zmax+0.3)
        
        fig.suptitle(f'Frame: {current_frame[0]}/{num_frame-1} | Speed: {current_speed[0]:.1f}x\n'
                    f'Controls: ← → (navigate) | ↑ ↓ (speed) | Space (play/pause) | Q (quit)', 
                    fontsize=10)
        plt.draw()
    
    def on_key(event):
        if event.key == 'right':
            current_frame[0] = min(current_frame[0] + 1, num_frame - 1)
            update_plot()
        elif event.key == 'left':
            current_frame[0] = max(current_frame[0] - 1, 0)
            update_plot()
        elif event.key == 'up':
            current_speed[0] = min(current_speed[0] + 0.5, 10.0)
            update_plot()
        elif event.key == 'down':
            current_speed[0] = max(current_speed[0] - 0.5, 0.5)
            update_plot()
        elif event.key == ' ':
            is_playing[0] = not is_playing[0]
            if is_playing[0]:
                play_animation()
        elif event.key == 'q':
            plt.close(fig)
    
    def play_animation():
        base_interval = 0.05
        while is_playing[0] and current_frame[0] < num_frame - 1:
            interval = base_interval / current_speed[0]
            current_frame[0] += 1
            update_plot()
            plt.pause(interval)
        if current_frame[0] >= num_frame - 1:
            is_playing[0] = False
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    update_plot()
    plt.show() 
    
 
## main函数
def main():
    parser = argparse.ArgumentParser(description='NTU RGB+D Skeleton 3D可视化工具')
    parser.add_argument('skeleton_file', type=str, nargs='?', 
                        default='S001C001P003R001A001_person1.skeleton',
                        help='Skeleton文件路径')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='播放速度系数，大于1加速，小于1减速 (默认: 1.0)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.skeleton_file):
        print(f"错误: 文件不存在: {args.skeleton_file}")
        return
    
    print(f"正在读取文件: {args.skeleton_file}")
    point = read_xyz(args.skeleton_file)
    print('Read Data Done!')
    print(f'播放速度: {args.speed}x')
 
    num_frame = point.shape[1]
    print(point.shape)
 
    # 相邻关节标号 
    arms = [23, 11, 10, 9, 8, 20, 4, 5, 6, 7, 21]
    rightHand = [11, 24]
    leftHand = [7, 22]
    legs = [19, 18, 17, 16, 0, 12, 13, 14, 15]
    body = [3, 2, 20, 1, 0]
    
    Print3D(num_frame, point, arms, rightHand, leftHand, legs, body, speed=args.speed)

if __name__ == '__main__':
    main()