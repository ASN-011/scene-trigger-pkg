# scene_trigger_pkg

Automatic interaction triggering for the FORSocialRobots marketplace architecture, using RGB camera vision instead of a manual trigger.

This is part of my Master's thesis at the FAPS chair, FAU Erlangen-Nürnberg, supervised by Prof. Sebastian Reitelshöfer.

## What it does

The existing marketplace architecture needs something to tell it when an interaction should start. In the reference setup this was a manual joystick button pressed by whoever was running the demo. This package replaces that with a camera-based pipeline that detects when a person is looking at the robot with enough sustained attention to mean they actually want to interact, and calls the trigger service automatically.

## Pipeline

A captured frame goes through five stages:

1. YOLOv8n detects person bounding boxes
2. SAM2 tracks each detected person across frames and keeps a stable ID per person
3. RetinaFace finds and quality-checks a face for the tracked person
4. L2CS-Net estimates gaze (pitch and yaw)
5. Persistence logic checks whether gaze has stayed within the threshold for long enough, then calls the trigger service

## Package layout

```
scene_trigger_pkg/
├── scene_trigger_pkg/
│   ├── camera_publisher.py     # publishes RGB frames from the webcam
│   ├── scene_trigger_node.py   # the actual pipeline (YOLO, SAM2, RetinaFace, L2CS-Net, trigger logic)
│   └── dashboard_node.py       # web dashboard for watching the pipeline live
├── package.xml
├── setup.py
└── setup.cfg
```

Model weights (`yolov8n.pt`, SAM2 checkpoint, L2CS-Net checkpoint) aren't tracked in git — see below for where to get them.

## Dependencies

ROS2 Jazzy on Ubuntu 24.04.

```bash
pip install ultralytics==8.0.196
pip install torch torchvision
pip install opencv-python==4.8.1.78   # must stay below 4.9
pip install numpy==1.26.4             # must stay below 2.0
pip install deepface
pip install retina-face
```

Model weights, download and drop into `scene_trigger_pkg/models/`:
- `yolov8n.pt` from [Ultralytics](https://github.com/ultralytics/ultralytics)
- SAM2 tiny checkpoint from [Meta's SAM2 repo](https://github.com/facebookresearch/segment-anything-2)
- L2CS-Net checkpoint from [L2CS-Net](https://github.com/Ahmednull/L2CS-Net)

## Building

```bash
cd ~/your_ws/src
git clone git@github.com:ASN-011/scene-trigger-pkg.git
cd ~/your_ws
colcon build --packages-select scene_trigger_pkg --symlink-install
source install/setup.bash
```

## Parameters

| Parameter | Value | Why |
|---|---|---|
| `PERSISTENCE_TIME_SEC` | 1.0 s | Falls within the 0.5–2 s window Belardinelli (2024) identifies as generally optimal for this kind of interaction |
| `TIMER_INTERVAL_SEC` | 0.5 s | Time-based rather than frame-count-based, so behaviour stays the same across different hardware speeds (this laptop runs the camera at ~3 fps, a GPU-equipped robot would run much faster) |
| `COOLDOWN_TIME_SEC` | 90.0 s | Matches the observed length of a full interaction cycle, so the system doesn't re-trigger mid-interaction |
| `GAZE_THRESHOLD_DEG` | 30° | Chosen as a baseline and tested against 20°/40° in the threshold sweep in my thesis (Section 4.2.2) — not pulled from a specific paper, this one's empirical |
| `GAZE_BUFFER_SIZE` | 3 | Also empirical, tied to the majority-to-unanimous vote fix described in the thesis |

## Running it

As part of the full system:

```bash
# Terminal 1
cd ~/FORSocialRobots_ws
source /opt/ros/jazzy/setup.bash && source install/setup.bash
ros2 launch social_marketplace_bringup social_launch_automatic.py

# Terminal 2
source /opt/ros/jazzy/setup.bash && source install/setup.bash
ros2 launch social_marketplace_ui ui_stack.launch.py
```

Just the trigger pipeline on its own:

```bash
ros2 run scene_trigger_pkg camera_publisher
ros2 run scene_trigger_pkg scene_trigger_node
```

## What I found during testing

Running on this laptop (CPU only, no dedicated GPU for inference):

- Camera: ~3 fps
- YOLO person detection: usually 85-90% confidence
- RetinaFace: usually ~95% confidence
- L2CS-Net: accurate to roughly ±15-30° when someone's actually looking at the camera
- Trigger latency: dominated by the 1.0s persistence window, not computation
- SAM2 keeps a stable ID as long as the person isn't moving much; it loses and resets IDs more easily when they are, which is the main source of latency variance at wider thresholds (see the thesis for the full breakdown)

## A couple of design choices worth knowing about

**Why SAM2 for tracking:** it assigns a persistent ID per person, which is what the persistence logic checks against (so it's the same person looking, not several different people adding up to the threshold between them).

**Why time-based persistence instead of counting frames:** a frame count means something different depending on how fast the camera runs. Checking elapsed time keeps the same 1.0s requirement regardless of fps.

**Why YOLO → RetinaFace → L2CS-Net as separate stages:** each stage filters out cases the next one shouldn't have to deal with, which keeps false positives down.

## On the automated feedback side

There's a separate, additive node (`automated_feedback_node.py`) that uses a mood reading already available in the scene parameters to decide whether feedback can be auto-submitted instead of waiting on the manual questionnaire. It's not included in this repo yet — ask me if you need it, it's a smaller, more prototype-stage piece of the thesis.

## References

- Deng et al. (2020) — RetinaFace
- Abdelrahman et al. (2022) — L2CS-Net
- Ravi et al. (2025) — SAM 2
- Belardinelli (2024) — gaze-based intention estimation in HRI
- Lavit Nicora et al. (2024) — gaze-based interaction triggering (persistence over multiple frames)
- Reitelshöfer et al. — FORSocialRobots marketplace architecture

## License / funding

Part of a Master's thesis at FAU Erlangen-Nürnberg, FAPS chair. Funded by BFS — Bayerische Forschungsstiftung, FORSocialRobots (AZ-1594-23).
