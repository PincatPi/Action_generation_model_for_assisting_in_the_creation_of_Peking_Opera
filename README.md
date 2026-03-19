# 京剧创作辅助的动作生成模型

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.1+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

本项目是一个基于人体姿态估计和骨架动作识别的京剧动作生成辅助系统，结合了 **VIBE** (Video Inference for Human Body Pose and Shape Estimation) 和 **CTR-GCN** (Channel-wise Topology Refinement Graph Convolution) 两项前沿技术，用于从视频中提取人体动作并转换为可用于京剧创作的动作数据。

## 项目结构

```
.
├── VIBE/                           # 人体姿态估计模块 (CVPR 2020)
│   ├── demo.py                     # 视频推理演示脚本
│   ├── train.py                    # 训练脚本
│   ├── lib/models/vibe.py          # VIBE 模型实现
│   ├── lib/models/smpl.py          # SMPL 身体模型
│   └── lib/smplify/                # Temporal SMPLify 实现
│
├── CTR_GCN/                        # 骨架动作识别模块 (ICCV 2021)
│   ├── main.py                     # 主训练/测试脚本
│   ├── ensemble.py                 # 多模态集成脚本
│   ├── model/ctrgcn.py             # CTR-GCN 模型实现
│   ├── feeders/                    # 数据加载器
│   └── graph/                      # 图拓扑定义
│
├── conda_envs/                     # Conda 环境配置 (已忽略)
├── vibe_win_install/               # Windows 安装文件 (已忽略)
└── README.md                       # 本文件
```

## 功能特性

### VIBE 模块
- 从单目视频中估计人体 3D 姿态和形状
- 输出 SMPL 模型参数（姿态、形状、相机参数）
- 支持时序一致性优化
- 可用于提取视频中的人体骨骼关键点

### CTR-GCN 模块
- 基于通道拓扑细化的图卷积网络
- 支持 NTU RGB+D 60/120 和 NW-UCLA 数据集
- 支持关节、骨骼、关节运动、骨骼运动四种模态
- 集成多模态融合策略

## 环境要求

### 基础环境
- Python >= 3.7 (VIBE), Python >= 3.6 (CTR-GCN)
- PyTorch >= 1.1.0
- CUDA (用于 GPU 加速)

### 安装步骤

#### 1. 克隆仓库
```bash
git clone https://github.com/PincatPi/Action_generation_model_for_assisting_in_the_creation_of_Peking_Opera.git
cd Action_generation_model_for_assisting_in_the_creation_of_Peking_Opera
```

#### 2. 安装 CTR-GCN 依赖
```bash
cd CTR_GCN
pip install -r requirements.txt
cd ..
```

#### 3. 安装 VIBE 依赖
```bash
cd VIBE
pip install -r requirements.txt
```

#### 4. 下载 SMPL 模型 (VIBE 需要)
从 [SMPL 官网](https://smpl.is.tue.mpg.de/) 下载 SMPL 模型文件，并放置在相应目录。

## 使用方法

### CTR-GCN 训练

#### NTU RGB+D 120 (跨被试)
```bash
cd CTR_GCN
python main.py --config config/nturgbd120-cross-subject/default.yaml \
    --work-dir work_dir/ntu120/csub/ctrgcn \
    --device 0
```

#### 测试模型
```bash
python main.py --config <work_dir>/config.yaml \
    --work-dir <work_dir> \
    --phase test \
    --save-score True \
    --weights <work_dir>/xxx.pt \
    --device 0
```

#### 多模态集成
```bash
python ensemble.py --datasets ntu120/xsub \
    --joint-dir work_dir/... \
    --bone-dir work_dir/... \
    --joint-motion-dir work_dir/... \
    --bone-motion-dir work_dir/...
```

### VIBE 推理

#### 运行视频 Demo
```bash
cd VIBE
python demo.py --vid_file sample_video.mp4 \
    --output_folder output/ \
    --display
```

#### 训练
```bash
python train.py --cfg configs/config.yaml
```

## 数据集

### NTU RGB+D 60/120
- 用于骨架动作识别
- 申请地址: https://rose1.ntu.edu.sg/dataset/actionRecognition

### NW-UCLA
- 跨视角动作识别数据集

### 3DPW, AMASS, MPI-INF-3DHP
- 用于 VIBE 训练的 3D 姿态估计数据集

## 项目应用

本项目可应用于：
- **京剧动作采集**: 从视频中提取专业演员的动作数据
- **动作生成**: 基于骨架数据生成新的京剧动作序列
- **动作教学**: 为学习者提供标准化的动作参考
- **数字化保存**: 将传统京剧动作以数字形式保存

## 致谢

本项目基于以下开源项目：

- [VIBE](https://github.com/mkocabiyik/VIBE) - Video Inference for Human Body Pose and Shape Estimation (CVPR 2020)
- [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN) - Channel-wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition (ICCV 2021)

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 联系方式

如有问题或建议，请通过 GitHub Issues 联系我们。

---

**注意**: 本项目仅供研究和学习使用。使用本项目时请遵守相关法律法规，尊重知识产权。