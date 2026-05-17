# 源代码清单

## 一、基础框架信息

| 项目         | 内容                                   |
| ------------ | -------------------------------------- |
| **框架名称** | LeRobot                                |
| **官方仓库** | https://github.com/huggingface/lerobot |
| **许可证**   | Apache-2.0                             |

---

## 二、本项目目录结构

lerobot/ 

├── src/lerobot/ 

│   ├── policies/

│   │   ├── cnn_baseline/          # CNN 基准策略（新增） 

│   │   ├── pinn_v1/               # PINN 方案 v1（新增） 

│   │   ├── pinn_v2/               # PINN 方案 v2（新增） 

│   │   └── pinn_v3/               # PINN 方案 v3（新增） 

│   └── utils/ 

│       └── visualization_utils.py   # 可视化工具（新增） 

├── tools/ 

│   ├── yolo_range/ 

│   │   ├── calibrate_focal_px_once.py  # 相机焦距标定（新增） 

│   │   └── run_overlay_live.py         # YOLO 实时检测（新增） 

│   ├── calib_fx.py              # 相机标定脚本（新增） 

│   ├── test_cam.py              # 摄像头测试（新增） 

│   ├── focal_px.txt             # 标定参数存储（新增） 

│   └── yolov8n.pt               # YOLOv8 预训练权重（新增） 

└── [其他框架原有文件]

---

## 三、新增/修改文件清单

### 3.1 策略模型文件（`src/lerobot/policies/`）

| 序号 | 文件路径                             | 类型 | 修改说明                            |
| :--: | ------------------------------------ | :--: | ----------------------------------- |
|  1   | `src/lerobot/policies/cnn_baseline/` | 新增 | CNN 基准策略实现，用于对比实验      |
|  2   | `src/lerobot/policies/pinn_v1/`      | 新增 | PINN 方案 v1：仅运动学约束          |
|  3   | `src/lerobot/policies/pinn_v2/`      | 新增 | PINN 方案 v2：运动学 + 动力学约束   |
|  4   | `src/lerobot/policies/pinn_v3/`      | 新增 | PINN 方案 v3：完整模型 + 自循环结构 |

### 3.2 工具脚本文件（`tools/`）

| 序号 | 文件路径                                      | 类型 | 修改说明                   |
| :--: | --------------------------------------------- | :--: | -------------------------- |
|  5   | `tools/calib_fx.py`                           | 新增 | 相机焦距标定主脚本         |
|  6   | `tools/focal_px.txt`                          | 新增 | 存储标定得到的焦距参数     |
|  7   | `tools/test_cam.py`                           | 新增 | 摄像头连接与测试脚本       |
|  8   | `tools/yolov8n.pt`                            | 新增 | YOLOv8 nano 预训练模型权重 |
|  9   | `tools/yolo_range/calibrate_focal_px_once.py` | 新增 | 单次焦距标定工具           |
|  10  | `tools/yolo_range/run_overlay_live.py`        | 新增 | YOLO 实时检测与可视化      |

### 3.3 工具函数文件（`src/lerobot/utils/`）

| 序号 | 文件路径                                   | 类型 | 修改说明                     |
| :--: | ------------------------------------------ | :--: | ---------------------------- |
|  11  | `src/lerobot/utils/visualization_utils.py` | 新增 | 训练过程可视化与结果绘图工具 |

---

## 四、文件统计

| 类别             |   数量    | 说明                             |
| ---------------- | :-------: | -------------------------------- |
| **新增策略模块** |   4 个    | cnn_baseline, pinn_v1/v2/v3      |
| **新增工具脚本** |   6 个    | 相机标定、YOLO检测、测试工具     |
| **新增工具函数** |   1 个    | 可视化工具                       |
| **新增模型权重** |   1 个    | YOLOv8n.pt                       |
| **新增配置文件** |   1 个    | focal_px.txt                     |
| **合计**         | **13 个** | 均为新增文件，无框架原生文件修改 |

---

## 五、核心功能说明

### 5.1 策略模型对比方案

| 方案         | 路径                     | 技术特点                       |
| ------------ | ------------------------ | ------------------------------ |
| CNN-Baseline | `policies/cnn_baseline/` | 纯数据驱动，ResNet-18 骨干网络 |
| PINN-v1      | `policies/pinn_v1/`      | 增加运动学约束损失             |
| PINN-v2      | `policies/pinn_v2/`      | 增加动力学约束损失             |
| PINN-v3      | `policies/pinn_v3/`      | 完整 PINN + 自循环 GRU 结构    |

### 5.2 视觉感知模块

- **相机标定**：`tools/calib_fx.py` + `calibrate_focal_px_once.py` 实现焦距自动标定
- **目标检测**：`yolov8n.pt` 提供预训练权重，`run_overlay_live.py` 实现实时检测
- **摄像头测试**：`test_cam.py` 用于硬件连接验证

### 5.3 可视化工具

- `visualization_utils.py`：提供训练曲线绘制、抓取结果可视化、对比实验图表生成

---

## 六、快速开始

### 6.1 环境准备

```bash
# 1. 克隆官方框架
git clone https://github.com/huggingface/lerobot.git

# 2. 将本项目的文件复制到对应目录
# - policies/ 下的四个文件夹 → lerobot/src/lerobot/policies/
# - tools/ 下的文件 → lerobot/tools/
# - visualization_utils.py → lerobot/src/lerobot/utils/

# 3. 安装依赖
pip install -e .
```

### 6.2 硬件连接与配置

```bash
# 4. 查找机械臂端口
python -m lerobot.find_port

# 5. 查找可用摄像头
python -m lerobot.find_cameras opencv
```

### 6.3 遥操作控制

```bash
# 6. 启动主从遥操作（Leader-Follower模式）
conda activate lerobot && cd ~/lerobot
python -m lerobot.teleoperate \
    --robot.type=so101_follower \
    --robot.port='COM6' \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, up: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port='COM5' \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true
```

### 6.4 数据采集

```bash
# 7. 录制训练数据集
python -m lerobot.record \
    --robot.type=so101_follower \
    --robot.port='COM6' \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, up: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port='COM5' \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true \
    --dataset.repo_id=seeed_123/so101_test \
    --dataset.num_episodes=50 \
    --dataset.single_task="Grab the black cube" \
    --dataset.push_to_hub=false \
    --dataset.episode_time_s=70 \
    --dataset.reset_time_s=20
```

### 6.5 模型训练

```bash
# 8. 训练 ACT 策略
python -m lerobot.scripts.train \
    --dataset.repo_id=seeed_123/so101_test \
    --policy.type=act \
    --output_dir=outputs/train/act_so101_test \
    --job_name=act_so101_test \
    --policy.device=cuda \
    --wandb.enable=false \
    --policy.push_to_hub=false \
    --steps=300000

# 9. 训练 CNN 基准模型
python -m lerobot.scripts.train \
    --dataset.repo_id=seeed_123/so101_test \
    --policy.path=src/lerobot/policies/cnn_baseline \
    --output_dir=outputs/train/cnn_baseline \
    --job_name=cnn_baseline \
    --policy.device=cuda \
    --wandb.enable=false \
    --policy.push_to_hub=false \
    --steps=300000

# 10. 训练 PINN-v3 模型
python -m lerobot.scripts.train \
    --dataset.repo_id=seeed_123/so101_test \
    --policy.path=src/lerobot/policies/pinn_v3 \
    --output_dir=outputs/train/pinn_v3 \
    --job_name=pinn_v3 \
    --policy.device=cuda \
    --wandb.enable=false \
    --policy.push_to_hub=false \
    --steps=300000
```

### 6.6 模型评估

```bash
# 11. 评估训练好的策略
conda activate lerobot && cd C:\Users\wangruihan\~\lerobot
python -m lerobot.record \
    --robot.type=so101_follower \
    --robot.port='COM6' \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, up: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
    --robot.id=my_awesome_follower_arm \
    --display_data=false \
    --dataset.repo_id=seeed/eval_test123 \
    --dataset.single_task="Put lego brick into the transparent box" \
    --policy.path=your path\~\lerobot\outputs\train\act_so101_test\checkpoints\last\pretrained_model
```

## 七、注意事项

1. **策略模块独立性**：四个策略文件夹（cnn_baseline, pinn_v1/v2/v3）相互独立，可直接切换对比
2. **相机标定**：首次使用需运行 `calib_fx.py` 获取焦距参数，结果保存在 `focal_px.txt`
3. **YOLO 权重**：`yolov8n.pt` 为官方预训练权重，可根据需要替换为其他版本
4. **无框架侵入性修改**：本项目所有文件均为新增，未修改 LeRobot 框架原生代码
5. **摄像头名称一致性**：评估时 `--robot.cameras` 中的名称必须与数据采集时严格一致，否则会出现 infinity 错误
6. **端口配置**：Windows 系统使用 `COMx` 格式，Linux 系统使用 `/dev/ttyACM0` 格式