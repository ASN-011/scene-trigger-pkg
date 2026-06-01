# Automatic Interaction Triggering for Human-Robot Interaction Using RGB Vision

> Master's Thesis | FAU Erlangen-Nürnberg | FAPS Chair  
> Author: Athul Sukesh Nair | Supervisor: Prof. Sebastian Reitelshöfer

---

## Overview

This package implements an **automatic interaction triggering system** for social robots using RGB camera vision. It replaces the manual joystick-based trigger in the existing FORSocialRobots marketplace architecture with a fully automatic detection pipeline.

The system detects when a person is intentionally looking at the robot and automatically triggers the downstream social interaction pipeline.

---

## System Architecture

```
Camera Feed (RGB)
       │
       ▼
┌─────────────┐
│  YOLOv8n    │  ── Person presence detection
└─────────────┘
       │
       ▼
┌─────────────┐
│    SAM2     │  ── Persistent cross-frame person tracking (unique IDs)
└─────────────┘
       │
       ▼
┌─────────────┐
│ RetinaFace  │  ── Face detection on tracked person
└─────────────┘
       │
       ▼
┌─────────────┐
│  L2CS-Net   │  ── Gaze direction estimation (pitch + yaw angles)
└─────────────┘
       │
       ▼
┌──────────────────────┐
│  Time-based Trigger  │  ── Same person looking ≥ 1.0s → trigger
└──────────────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│     FORSocialRobots Marketplace          │
│  DeepFace → Scene Analysis → Bidding     │
│  → Agent Selection → Video Playback      │
└──────────────────────────────────────────┘
```

---

## Package Structure

```
scene_trigger_pkg/
├── scene_trigger_pkg/
│   ├── __init__.py
│   ├── camera_publisher.py      # Publishes RGB frames from webcam
│   ├── scene_trigger_node.py    # Main trigger pipeline (YOLO + SAM2 + RetinaFace + L2CS)
│   └── dashboard_node.py        # Live web dashboard for pipeline monitoring
├── models/                      # AI model weights (not tracked in git)
│   ├── yolov8n.pt
│   ├── sam2_tiny.pth
│   └── l2cs_fixed.pkl
├── package.xml
├── setup.py
└── setup.cfg
```

---

## Dependencies

### ROS2
- ROS2 Jazzy (Ubuntu 24.04)

### Python Packages
```bash
pip install ultralytics==8.0.196          # YOLOv8n
pip install torch torchvision             # PyTorch
pip install opencv-python==4.8.1.78       # OpenCV (must be <4.9)
pip install numpy==1.26.4                 # NumPy (must be <2.0)
pip install deepface                      # DeepFace
pip install retina-face                   # RetinaFace
```

### Model Weights
Download and place in `scene_trigger_pkg/models/`:
- `yolov8n.pt` — YOLOv8 nano from [Ultralytics](https://github.com/ultralytics/ultralytics)
- `sam2_tiny.pth` — SAM2 tiny from [Meta AI](https://github.com/facebookresearch/segment-anything-2)
- `l2cs_fixed.pkl` — L2CS-Net from [L2CS-Net](https://github.com/Ahmednull/L2CS-Net)

---

## Installation

```bash
# Clone into your ROS2 workspace
cd ~/your_ws/src
git clone git@github.com:ASN-011/scene-trigger-pkg.git

# Build
cd ~/your_ws
colcon build --packages-select scene_trigger_pkg --symlink-install
source install/setup.bash
```

---

## Parameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `PERSISTENCE` | 1.0s | Lavit Nicora et al. 2024 - ideal observation window for interaction |
| `TIMER` | 0.5s | Hardware-independent timer tick (not frame-count based) |
| `COOLDOWN` | 20s | Prevents repeated triggering after interaction |
| `GAZE_THRESHOLD` | 30° | Standard gaze-within-ROI threshold from literature |

> **Note:** Time-based persistence (not frame-count) ensures consistent behavior across different hardware speeds (3fps laptop vs 30fps GPU robot).

---

## Usage

### Launch (as part of full FORSocialRobots system)

**Terminal 1 - Backend:**
```bash
cd ~/FORSocialRobots_ws
source /opt/ros/jazzy/setup.bash && source install/setup.bash
ros2 launch social_marketplace_bringup social_launch_automatic.py
```

**Terminal 2 - UI Stack:**
```bash
cd ~/FORSocialRobots_ws
source /opt/ros/jazzy/setup.bash && source install/setup.bash
ros2 launch social_marketplace_ui ui_stack.launch.py
```

### Standalone (trigger node only)

```bash
# Terminal 1 - Camera
ros2 run scene_trigger_pkg camera_publisher

# Terminal 2 - Trigger Node
ros2 run scene_trigger_pkg scene_trigger_node
```

---

## Results

End-to-end pipeline performance (tested on OMEN laptop, CPU-only):

| Component | Performance |
|-----------|-------------|
| Camera fps | ~3 fps |
| YOLO person detection | ~85-90% confidence |
| RetinaFace face detection | ~95% confidence |
| L2CS-Net gaze estimation | ±15° to ±30° when looking at camera |
| Trigger latency | ~1.0s (persistence window) |
| SAM2 ID stability | Stable when subject is stationary |

---

## Key Design Decisions

1. **SAM2 for tracking** — Assigns persistent unique IDs per person across frames, ensuring the same person maintains eye contact (not multiple different people)
2. **Time-based persistence** — Uses a 0.5s timer tick instead of frame counting, making the system hardware-independent
3. **Three-stage detection** — YOLO (presence) → RetinaFace (face) → L2CS-Net (gaze) reduces false positives
4. **20s cooldown** — Prevents the system from re-triggering immediately after an interaction

---

## References

- Lavit Nicora et al. (2024) — Gaze-based interaction triggering
- Cheng et al. (2026) — L2CS-Net gaze estimation (77.6% accuracy on Pepper)
- Admoni & Scassellati (2017) — Gaze in human-robot interaction survey
- Reitelshöfer et al. — FORSocialRobots marketplace architecture

---

## License

This project is part of a Master's thesis at FAU Erlangen-Nürnberg (FAPS Chair).  
Funded by BFS – Bayerische Forschungsstiftung – FORSocialRobots (AZ-1594-23).
