# scene_trigger_node.py
#
# Replaces the manual joystick trigger in the FORSocialRobots
# marketplace with automatic RGB-camera-based detection.
#
# Pipeline: camera -> YOLO (person) -> SAM2 (tracking) ->
# RetinaFace (face) -> L2CS-Net (gaze) -> persistence check
# -> call /trigger_scene_analysis
#
# Persistence is time-based rather than frame-count-based so
# behaviour stays consistent across different hardware speeds.
# Gaze offset is calibrated once per deployment setup
# (ros2 run scene_trigger_pkg calibrate_gaze, saved to
# config/gaze_offset.yaml).
#
# Part of my Master's thesis at FAU Erlangen-Nürnberg, FAPS chair.
# Author: Athul Sukesh Nair. Supervisor: Prof. Sebastian Reitelshöfer.

import json
import os
import pathlib
import yaml
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from ultralytics import YOLO
from face_detection import RetinaFace
from l2cs import getArch
from sam2.sam2_image_predictor import SAM2ImagePredictor
from social_marketplace_interfaces.srv import TriggerSceneAnalysis

# Path to L2CS-Net model weights
MODEL_PATH = pathlib.Path(__file__).parent / 'models' / 'l2cs_fixed.pkl'

# ── Configurable Parameters ──────────────────
# Baseline value, tested against 20°/40° in the threshold sweep
# (see thesis Section 4.2.2) rather than taken from a paper
GAZE_THRESHOLD_DEG = 30.0

# Falls within the 0.5-2s window Belardinelli (2024) identifies
# as generally optimal; Lavit Nicora et al. (2024) separately
# establish that multiple consecutive frames, not a single one,
# are needed for reliable gaze-based triggering
PERSISTENCE_TIME_SEC = 1.0

# How often the timer checks the latest frame
TIMER_INTERVAL_SEC = 0.5

# Matches the observed length of a full interaction cycle,
# so the system doesn't re-trigger mid-interaction
COOLDOWN_TIME_SEC = 90.0

# Empirical, tied to the majority-to-unanimous vote fix
# described in the thesis (Section 4.1.3)
GAZE_BUFFER_SIZE = 3

# ── Face detection quality thresholds ────────
# Reject unreliable gaze readings from partial/profile faces.
# These commonly occur at extreme gaze angles, where L2CS-Net's
# training distribution (mostly frontal faces) generalizes less
# reliably
FACE_CONFIDENCE_MIN   = 0.9
FACE_MIN_SIZE_PX       = 60
FACE_MIN_ASPECT_RATIO  = 0.6

# ── Gaze Offset Calibration ──────────────────
# Loaded once at startup from config/gaze_offset.yaml, generated
# by calibrate_gaze. Falls back to (0, 0) if no file is found.
_OFFSET_FILE = os.path.expanduser(
    '~/FORSocialRobots_ws/src/scene_trigger_pkg/config/gaze_offset.yaml'
)
GAZE_PITCH_OFFSET = 0.0
GAZE_YAW_OFFSET   = 0.0

try:
    with open(_OFFSET_FILE) as _f:
        _cfg = yaml.safe_load(_f)
        GAZE_PITCH_OFFSET = _cfg['gaze_offset']['pitch_offset_deg']
        GAZE_YAW_OFFSET   = _cfg['gaze_offset']['yaw_offset_deg']
    print(
        f'[scene_trigger_node] Gaze calibration loaded: '
        f'pitch_offset={GAZE_PITCH_OFFSET}° '
        f'yaw_offset={GAZE_YAW_OFFSET}°'
    )
except Exception:
    print(
        '[scene_trigger_node] No calibration file found. '
        'Using default offset (0, 0). '
        'Run calibrate_gaze to improve accuracy.'
    )


class SceneTriggerNode(Node):

    def __init__(self):
        super().__init__('scene_trigger_node')

        self.get_logger().info('Scene Trigger Node started.')
        self.get_logger().info(
            'YOLO + SAM2 + RetinaFace + L2CS-Net, '
            'time-based persistence and person tracking.'
        )
        self.get_logger().info(
            f'Persistence threshold: {PERSISTENCE_TIME_SEC}s | '
            f'Timer interval: {TIMER_INTERVAL_SEC}s | '
            f'Gaze threshold: {GAZE_THRESHOLD_DEG} degrees'
        )
        self.get_logger().info(
            f'Gaze offset: pitch={GAZE_PITCH_OFFSET}° '
            f'yaw={GAZE_YAW_OFFSET}°'
        )

        # ── YOLO model ──────────────────────────
        self.get_logger().info('Loading YOLO model...')
        self.yolo = YOLO('yolov8n.pt')
        self.get_logger().info('YOLO model loaded.')

        # ── SAM2 predictor ──────────────────────
        self.get_logger().info('Loading SAM2 model...')
        self.sam2_predictor = SAM2ImagePredictor.from_pretrained(
            'facebook/sam2-hiera-tiny',
            device='cuda'
        )
        self.get_logger().info('SAM2 model loaded.')

        # ── RetinaFace detector ──────────────────
        self.get_logger().info('Loading RetinaFace detector...')
        self.face_detector = RetinaFace()
        self.get_logger().info('RetinaFace loaded.')

        # ── L2CS-Net gaze estimator ──────────────
        self.get_logger().info('Loading L2CS-Net gaze model...')
        self.gaze_model = getArch('ResNet50', 90)
        self.gaze_model.load_state_dict(
            torch.load(str(MODEL_PATH), map_location='cpu')
        )
        self.gaze_model.eval()
        self.softmax = nn.Softmax(dim=1)
        self.idx_tensor = torch.FloatTensor(list(range(90)))
        self.gaze_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])
        self.get_logger().info('L2CS-Net loaded.')

        self.bridge = CvBridge()

        self.latest_frame    = None
        self.last_confidence = 0.0

        self.current_person_id = None
        self.gaze_buffer = []
        self.looking_duration = 0.0

        self.in_cooldown        = False
        self.cooldown_remaining = 0.0

        self.client = self.create_client(
            TriggerSceneAnalysis,
            '/trigger_scene_analysis'
        )
        self.get_logger().info(
            'Service client created for /trigger_scene_analysis.'
        )

        self.dashboard_pub = self.create_publisher(
            String,
            '/dashboard_update',
            10
        )

        self.image_subscriber = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10
        )
        self.get_logger().info(
            'Subscribed to /image_raw. Waiting for frames.'
        )

        self.timer = self.create_timer(
            TIMER_INTERVAL_SEC,
            self.timer_callback
        )
        self.get_logger().info(
            f'Timer started at {TIMER_INTERVAL_SEC}s interval.'
        )

    def image_callback(self, msg):
        """Saves the latest camera frame."""
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(
                msg, 'bgr8'
            )
        except Exception as e:
            self.get_logger().error(
                f'Image conversion failed: {e}'
            )

    def timer_callback(self):
        """Called every TIMER_INTERVAL_SEC seconds."""

        if self.in_cooldown:
            self.cooldown_remaining -= TIMER_INTERVAL_SEC
            if self.cooldown_remaining <= 0.0:
                self.in_cooldown        = False
                self.cooldown_remaining = 0.0
                self.get_logger().info(
                    'Cooldown finished. Ready for new trigger.'
                )
            self.publish_dashboard(
                person_detected=False,
                face_detected=False,
                looking=False,
                pitch=0.0,
                yaw=0.0
            )
            return

        if self.latest_frame is None:
            return

        frame = self.latest_frame.copy()

        results    = self.yolo(frame, verbose=False)
        person_box = None

        for result in results:
            for box in result.boxes:
                if self.yolo.names[int(box.cls[0])] == 'person':
                    self.last_confidence = float(box.conf[0]) * 100
                    person_box = box.xyxy[0].cpu().numpy()
                    break
            if person_box is not None:
                break

        if person_box is None:
            if self.looking_duration > 0.0:
                self.get_logger().info(
                    'Person no longer detected. '
                    'Resetting looking duration.'
                )
            self.looking_duration  = 0.0
            self.gaze_buffer       = []
            self.current_person_id = None
            self.publish_dashboard(
                person_detected=False,
                face_detected=False,
                looking=False,
                pitch=0.0,
                yaw=0.0
            )
            return

        person_id = self.get_person_id_with_sam2(frame, person_box)

        if person_id is None:
            self.get_logger().warn(
                'SAM2 segmentation failed. Skipping this frame.'
            )
            return

        if self.current_person_id is not None:
            if person_id != self.current_person_id:
                self.get_logger().info(
                    f'Different person detected! '
                    f'Old ID: {self.current_person_id}, '
                    f'New ID: {person_id}. '
                    f'Resetting duration.'
                )
                self.looking_duration  = 0.0
                self.gaze_buffer       = []
                self.current_person_id = person_id
                return

        if self.current_person_id is None:
            self.current_person_id = person_id
            self.get_logger().info(
                f'New person detected with SAM2 ID: {person_id}'
            )

        gaze_result = self.estimate_gaze(frame)

        if gaze_result is None:
            self.get_logger().info(
                'Person detected but face not found. '
                'Resetting duration.'
            )
            self.looking_duration = 0.0
            self.gaze_buffer      = []
            self.publish_dashboard(
                person_detected=True,
                face_detected=False,
                looking=False,
                pitch=0.0,
                yaw=0.0
            )
            return

        pitch_deg, yaw_deg = gaze_result

        looking = self.is_looking_at_camera(pitch_deg, yaw_deg)
        self.gaze_buffer.append(looking)
        if len(self.gaze_buffer) > GAZE_BUFFER_SIZE:
            self.gaze_buffer.pop(0)

        smoothed_looking = (
            self.gaze_buffer.count(True) >= len(self.gaze_buffer)
        )

        if smoothed_looking:
            self.looking_duration += TIMER_INTERVAL_SEC
        else:
            if self.looking_duration > 0.0:
                self.get_logger().info(
                    'Person looked away. Resetting duration.'
                )
            self.looking_duration = 0.0

        # Distance from the calibrated center, just for logging
        pitch_dist = abs(pitch_deg - GAZE_PITCH_OFFSET)
        yaw_dist   = abs(yaw_deg   - GAZE_YAW_OFFSET)

        self.get_logger().info(
            f'Person ID {person_id}. '
            f'Pitch: {pitch_deg:.1f}° (Δ{pitch_dist:.1f}°)  '
            f'Yaw: {yaw_deg:.1f}° (Δ{yaw_dist:.1f}°)  '
            f'Looking: {smoothed_looking} | '
            f'Duration: {self.looking_duration:.1f}s'
            f'/{PERSISTENCE_TIME_SEC}s'
        )

        self.publish_dashboard(
            person_detected=True,
            face_detected=True,
            looking=smoothed_looking,
            pitch=pitch_deg,
            yaw=yaw_deg
        )

        if self.looking_duration >= PERSISTENCE_TIME_SEC:
            self.get_logger().info(
                f'Person ID {person_id} confirmed looking at robot '
                f'for {self.looking_duration:.1f}s. '
                f'Calling trigger service.'
            )
            self.call_trigger_service()
            self.looking_duration  = 0.0
            self.gaze_buffer       = []
            self.current_person_id = None
            self.in_cooldown        = True
            self.cooldown_remaining = COOLDOWN_TIME_SEC

    def get_person_id_with_sam2(self, frame, person_box):
        """Segments the person with SAM2 and derives a stable ID
        from the mask's center of mass, quantized to a 100px grid
        so small movements don't change the ID."""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.sam2_predictor.set_image(frame_rgb)

            x1, y1, x2, y2 = person_box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            masks, scores, _ = self.sam2_predictor.predict(
                point_coords=np.array([[center_x, center_y]]),
                point_labels=np.array([1]),
                multimask_output=False
            )

            mask = masks[0]
            y_coords, x_coords = np.where(mask)

            if len(x_coords) == 0:
                return None

            center_of_mass_x = int(np.mean(x_coords))
            center_of_mass_y = int(np.mean(y_coords))

            stable_x  = round(center_of_mass_x / 100) * 100
            stable_y  = round(center_of_mass_y / 100) * 100
            person_id = hash((stable_x, stable_y)) % 1000000

            return person_id

        except Exception as e:
            self.get_logger().error(f'SAM2 error: {e}')
            return None

    def estimate_gaze(self, frame):
        """Detects a face with RetinaFace and estimates gaze with
        L2CS-Net. Returns (pitch_deg, yaw_deg), or None if no face
        passes the quality checks (confidence, size, aspect ratio)."""
        faces = self.face_detector(frame)

        if faces is None or len(faces) == 0:
            return None

        box, landmark, score = faces[0]

        if score < FACE_CONFIDENCE_MIN:
            return None

        x_min = max(int(box[0]), 0)
        y_min = max(int(box[1]), 0)
        x_max = int(box[2])
        y_max = int(box[3])

        face_width  = x_max - x_min
        face_height = y_max - y_min
        if face_width < FACE_MIN_SIZE_PX or face_height < FACE_MIN_SIZE_PX:
            return None
        aspect_ratio = face_width / face_height
        if aspect_ratio < FACE_MIN_ASPECT_RATIO:
            return None

        face_img = frame[y_min:y_max, x_min:x_max]
        if face_img.size == 0:
            return None

        face_rgb     = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (224, 224))
        img_tensor   = self.gaze_transform(face_resized).unsqueeze(0)

        with torch.no_grad():
            gaze_pitch, gaze_yaw = self.gaze_model(img_tensor)

        pitch = float(
            torch.sum(
                self.softmax(gaze_pitch) * self.idx_tensor
            ) * 4 - 180
        )
        yaw = float(
            torch.sum(
                self.softmax(gaze_yaw) * self.idx_tensor
            ) * 4 - 180
        )

        return pitch, yaw

    def is_looking_at_camera(self, pitch_deg, yaw_deg):
        """True if gaze falls within GAZE_THRESHOLD_DEG of the
        calibrated offset (or of 0,0 if no calibration was found)."""
        return (
            abs(yaw_deg   - GAZE_YAW_OFFSET)   < GAZE_THRESHOLD_DEG and
            abs(pitch_deg - GAZE_PITCH_OFFSET) < GAZE_THRESHOLD_DEG
        )

    def publish_dashboard(self, person_detected, face_detected,
                          looking, pitch, yaw,
                          trigger_fired=False, scene_id=0):
        """Publishes current state to dashboard node."""
        data = {
            "person_detected":    person_detected,
            "yolo_confidence":    round(self.last_confidence, 1),
            "face_detected":      face_detected,
            "looking_at_camera":  looking,
            "pitch_deg":          round(pitch, 1),
            "yaw_deg":            round(yaw, 1),
            "persistence_counter": round(self.looking_duration, 1),
            "persistence_threshold": PERSISTENCE_TIME_SEC,
            "cooldown_active":    self.in_cooldown,
            "cooldown_counter":   round(self.cooldown_remaining, 1),
            "trigger_fired":      trigger_fired,
            "scene_id":           scene_id,
            "person_id":          self.current_person_id,
            "pitch_offset":       GAZE_PITCH_OFFSET,
            "yaw_offset":         GAZE_YAW_OFFSET,
        }
        msg      = String()
        msg.data = json.dumps(data)
        self.dashboard_pub.publish(msg)

    def call_trigger_service(self):
        """Sends trigger request to scene_analysis."""
        if not self.client.service_is_ready():
            self.get_logger().warn(
                '/trigger_scene_analysis service not available.'
            )
            return

        request         = TriggerSceneAnalysis.Request()
        request.trigger = True

        future = self.client.call_async(request)
        future.add_done_callback(self.response_callback)

        self.get_logger().info(
            'Trigger request sent to scene_analysis node.'
        )

    def response_callback(self, future):
        """Handles trigger service response."""
        try:
            response = future.result()
            self.get_logger().info(
                f'Trigger successful. '
                f'Scene ID: {response.scene_id}'
            )
            self.publish_dashboard(
                person_detected=True,
                face_detected=True,
                looking=True,
                pitch=0.0,
                yaw=0.0,
                trigger_fired=True,
                scene_id=response.scene_id
            )
        except Exception as e:
            self.get_logger().error(
                f'Trigger service call failed: {e}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = SceneTriggerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
