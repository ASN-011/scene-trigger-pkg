# ─────────────────────────────────────────────
# scene_trigger_node.py
#
# THESIS: Automatic Interaction Triggering
#         for HRI using RGB Vision
#
# Author: Athul Sukesh Nair
# FAU Erlangen-Nürnberg, FAPS Chair
# Supervisor: Prof. Sebastian Reitelshöfer
#
# Description:
# This node replaces the manual joystick trigger
# in the FORSocialRobots social marketplace system
# with automatic RGB camera based detection.
#
# Pipeline:
# Camera -> YOLO person detection
#        -> SAM2 person segmentation & tracking
#        -> RetinaFace face detection
#        -> L2CS-Net gaze estimation
#        -> Time-based persistence check
#        -> Call /trigger_scene_analysis service
#
# The downstream system (DeepFace, scene analysis,
# bidding platform, agents) works unchanged.
#
# Literature basis:
# - Lavit Nicora et al. (2024): 5 frames at 25fps
#   scaled to time = ~1.0 second threshold
# - Belardinelli (2024): optimal window 0.5-2.0s
# - Lu et al. (2024): 3.0s engagement threshold
# - Cheng et al. (2026): L2CS-Net gaze threshold
#   abs(yaw) < 30 and abs(pitch) < 30 degrees
#
# Note on time-based approach:
# Per supervisor feedback, persistence is measured
# in time (seconds) rather than frame count to
# ensure consistent behavior across different
# hardware with varying frame rates.
#
# SAM2 person tracking:
# Per supervisor feedback, SAM2 tracks persons
# with unique IDs to ensure the SAME person is
# looking across all time checks.
# ─────────────────────────────────────────────

import json
import pathlib
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
# Gaze threshold from Cheng et al. (2026)
GAZE_THRESHOLD_DEG = 30.0

# Time-based persistence threshold in seconds
# Based on Lavit Nicora et al. (2024) and
# Belardinelli (2024): optimal window 0.5-2.0s
# Must look at robot for this long to trigger
PERSISTENCE_TIME_SEC = 1.0

# Timer interval in seconds
# How often the timer checks the latest frame
# Professor recommendation: configurable interval
TIMER_INTERVAL_SEC = 0.5

# Cooldown time in seconds after trigger fires
# Allows DeepFace analysis on CPU to complete
COOLDOWN_TIME_SEC = 60.0

# Gaze smoothing buffer size in frames
# Based on Cheng et al. (2026) and
# Lavit Nicora et al. (2024): 3-point window
GAZE_BUFFER_SIZE = 3


class SceneTriggerNode(Node):

    def __init__(self):
        super().__init__('scene_trigger_node')

        # ── Logger ──────────────────────────────
        self.get_logger().info('Scene Trigger Node started.')
        self.get_logger().info(
            'Stage 4+: YOLO + SAM2 + RetinaFace + L2CS-Net '
            'with time-based persistence and person tracking.'
        )
        self.get_logger().info(
            f'Persistence threshold: {PERSISTENCE_TIME_SEC}s | '
            f'Timer interval: {TIMER_INTERVAL_SEC}s | '
            f'Gaze threshold: {GAZE_THRESHOLD_DEG} degrees'
        )

        # ── YOLO model ──────────────────────────
        self.get_logger().info('Loading YOLO model...')
        self.yolo = YOLO('yolov8n.pt')
        self.get_logger().info('YOLO model loaded.')

        # ── SAM2 predictor ──────────────────────
        self.get_logger().info('Loading SAM2 model...')
        self.sam2_predictor = SAM2ImagePredictor.from_pretrained(
            'facebook/sam2-hiera-tiny'
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

        # ── cv_bridge ───────────────────────────
        self.bridge = CvBridge()

        # ── Latest frame storage ─────────────────
        # Image subscriber saves latest frame
        # Timer reads it at fixed intervals
        # This decouples frame rate from check rate
        self.latest_frame = None
        self.last_confidence = 0.0

        # ── SAM2 person tracking ─────────────────
        # Tracks the unique ID of the current person
        # If ID changes → different person → reset
        self.current_person_id = None

        # ── Gaze smoothing buffer ────────────────
        # Based on Cheng et al. (2026) and
        # Lavit Nicora et al. (2024)
        self.gaze_buffer = []

        # ── Time-based persistence ───────────────
        # Measures how long person has been
        # continuously looking at the robot
        # Professor feedback: use time not frames
        self.looking_duration = 0.0

        # ── Cooldown ─────────────────────────────
        self.in_cooldown = False
        self.cooldown_remaining = 0.0

        # ── Service client ────────────────────────
        self.client = self.create_client(
            TriggerSceneAnalysis,
            '/trigger_scene_analysis'
        )
        self.get_logger().info(
            'Service client created for /trigger_scene_analysis.'
        )

        # ── Dashboard publisher ──────────────────
        self.dashboard_pub = self.create_publisher(
            String,
            '/dashboard_update',
            10
        )

        # ── Camera subscriber ────────────────────
        # Only saves latest frame — does not process
        # Processing happens in timer callback
        self.image_subscriber = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10
        )
        self.get_logger().info(
            'Subscribed to /image_raw. Waiting for frames.'
        )

        # ── Timer ────────────────────────────────
        # Ticks every TIMER_INTERVAL_SEC seconds
        # Checks latest frame for gaze
        # Hardware independent — same behavior
        # on slow laptop or fast Jetson robot
        self.timer = self.create_timer(
            TIMER_INTERVAL_SEC,
            self.timer_callback
        )
        self.get_logger().info(
            f'Timer started at {TIMER_INTERVAL_SEC}s interval.'
        )

    def image_callback(self, msg):
        """
        Saves the latest camera frame.
        Does NOT process — just stores for timer.
        This decouples processing from frame rate.
        """
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(
                msg, 'bgr8'
            )
        except Exception as e:
            self.get_logger().error(
                f'Image conversion failed: {e}'
            )

    def timer_callback(self):
        """
        Called every TIMER_INTERVAL_SEC seconds.
        Checks latest frame for person and gaze.
        Time-based approach ensures consistent
        behavior regardless of hardware frame rate.
        """

        # Step 1: Handle cooldown
        if self.in_cooldown:
            self.cooldown_remaining -= TIMER_INTERVAL_SEC
            if self.cooldown_remaining <= 0.0:
                self.in_cooldown = False
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

        # Step 2: Check if frame is available
        if self.latest_frame is None:
            return

        frame = self.latest_frame.copy()

        # Step 3: YOLO person detection
        results = self.yolo(frame, verbose=False)
        person_box = None

        for result in results:
            for box in result.boxes:
                if self.yolo.names[int(box.cls[0])] == 'person':
                    self.last_confidence = float(box.conf[0]) * 100
                    # Get bounding box coordinates
                    person_box = box.xyxy[0].cpu().numpy()
                    break
            if person_box is not None:
                break

        # Step 4: No person — reset and return
        if person_box is None:
            if self.looking_duration > 0.0:
                self.get_logger().info(
                    'Person no longer detected. '
                    'Resetting looking duration.'
                )
            self.looking_duration = 0.0
            self.gaze_buffer = []
            self.current_person_id = None
            self.publish_dashboard(
                person_detected=False,
                face_detected=False,
                looking=False,
                pitch=0.0,
                yaw=0.0
            )
            return

        # Step 5: SAM2 person segmentation & tracking
        person_id = self.get_person_id_with_sam2(frame, person_box)

        if person_id is None:
            self.get_logger().warn(
                'SAM2 segmentation failed. Skipping this frame.'
            )
            return

        # Step 6: Check if same person as before
        if self.current_person_id is not None:
            if person_id != self.current_person_id:
                self.get_logger().info(
                    f'Different person detected! '
                    f'Old ID: {self.current_person_id}, '
                    f'New ID: {person_id}. '
                    f'Resetting duration.'
                )
                self.looking_duration = 0.0
                self.gaze_buffer = []
                self.current_person_id = person_id
                return

        # First person detected
        if self.current_person_id is None:
            self.current_person_id = person_id
            self.get_logger().info(
                f'New person detected with SAM2 ID: {person_id}'
            )

        # Step 7: Gaze estimation
        gaze_result = self.estimate_gaze(frame)

        if gaze_result is None:
            self.get_logger().info(
                'Person detected but face not found. '
                'Resetting duration.'
            )
            self.looking_duration = 0.0
            self.gaze_buffer = []
            self.publish_dashboard(
                person_detected=True,
                face_detected=False,
                looking=False,
                pitch=0.0,
                yaw=0.0
            )
            return

        pitch_deg, yaw_deg = gaze_result

        # Step 8: 3-frame smoothing buffer
        looking = self.is_looking_at_camera(pitch_deg, yaw_deg)
        self.gaze_buffer.append(looking)
        if len(self.gaze_buffer) > GAZE_BUFFER_SIZE:
            self.gaze_buffer.pop(0)

        smoothed_looking = (
            self.gaze_buffer.count(True) >= len(self.gaze_buffer) / 2
        )

        # Step 9: Update time-based duration
        if smoothed_looking:
            self.looking_duration += TIMER_INTERVAL_SEC
        else:
            if self.looking_duration > 0.0:
                self.get_logger().info(
                    'Person looked away. Resetting duration.'
                )
            self.looking_duration = 0.0

        self.get_logger().info(
            f'Person ID {person_id}. '
            f'Pitch: {pitch_deg:.1f} deg  '
            f'Yaw: {yaw_deg:.1f} deg  '
            f'Looking: {smoothed_looking} | '
            f'Duration: {self.looking_duration:.1f}s'
            f'/{PERSISTENCE_TIME_SEC}s'
        )

        # Step 10: Publish dashboard
        self.publish_dashboard(
            person_detected=True,
            face_detected=True,
            looking=smoothed_looking,
            pitch=pitch_deg,
            yaw=yaw_deg
        )

        # Step 11: Check time threshold
        if self.looking_duration >= PERSISTENCE_TIME_SEC:
            self.get_logger().info(
                f'Person ID {person_id} confirmed looking at robot '
                f'for {self.looking_duration:.1f}s. '
                f'Calling trigger service.'
            )
            self.call_trigger_service()
            self.looking_duration = 0.0
            self.gaze_buffer = []
            self.current_person_id = None
            self.in_cooldown = True
            self.cooldown_remaining = COOLDOWN_TIME_SEC

    def get_person_id_with_sam2(self, frame, person_box):
        """
        Uses SAM2 to segment the person and get unique ID.
        
        Instead of hashing the mask (unstable), we use
        the center of mass of the segmented region which
        is much more stable for the same person.
        
        Args:
            frame: BGR image
            person_box: [x1, y1, x2, y2] from YOLO
        
        Returns:
            Unique person ID based on position or None
        """
        try:
            # Convert BGR to RGB for SAM2
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Set image for SAM2
            self.sam2_predictor.set_image(frame_rgb)
            
            # Get center point of bounding box as prompt
            x1, y1, x2, y2 = person_box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # Predict mask using point prompt
            masks, scores, _ = self.sam2_predictor.predict(
                point_coords=np.array([[center_x, center_y]]),
                point_labels=np.array([1]),  # 1 = foreground
                multimask_output=False
            )
            
            # Get the mask with highest score
            mask = masks[0]
            
            # Calculate center of mass of mask
            # This is much more stable than hashing
            y_coords, x_coords = np.where(mask)
            
            if len(x_coords) == 0:
                return None
            
            center_of_mass_x = int(np.mean(x_coords))
            center_of_mass_y = int(np.mean(y_coords))
            
            # Create stable ID from bounding box position
            # Round to nearest 50 pixels to allow small movements
            stable_x = round(center_of_mass_x / 50) * 50
            stable_y = round(center_of_mass_y / 50) * 50
            
            # Generate ID from stable position
            person_id = hash((stable_x, stable_y)) % 1000000
            
            return person_id
            
        except Exception as e:
            self.get_logger().error(
                f'SAM2 segmentation error: {e}'
            )
            return None

    def estimate_gaze(self, frame):
        """
        Detects face using RetinaFace and estimates
        gaze direction using L2CS-Net.
        Returns (pitch_deg, yaw_deg) or None.
        """
        faces = self.face_detector(frame)

        if faces is None or len(faces) == 0:
            return None

        box, landmark, score = faces[0]
        x_min = max(int(box[0]), 0)
        y_min = max(int(box[1]), 0)
        x_max = int(box[2])
        y_max = int(box[3])

        face_img = frame[y_min:y_max, x_min:x_max]
        if face_img.size == 0:
            return None

        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (224, 224))
        img_tensor = self.gaze_transform(face_resized).unsqueeze(0)

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
        """
        Returns True if gaze is within threshold.
        Based on Cheng et al. (2026): 30 degrees.
        """
        return (
            abs(yaw_deg) < GAZE_THRESHOLD_DEG and
            abs(pitch_deg) < GAZE_THRESHOLD_DEG
        )

    def publish_dashboard(self, person_detected, face_detected,
                          looking, pitch, yaw,
                          trigger_fired=False, scene_id=0):
        """Publishes current state to dashboard node."""
        data = {
            "person_detected": person_detected,
            "yolo_confidence": round(self.last_confidence, 1),
            "face_detected": face_detected,
            "looking_at_camera": looking,
            "pitch_deg": round(pitch, 1),
            "yaw_deg": round(yaw, 1),
            "persistence_counter": round(self.looking_duration, 1),
            "persistence_threshold": PERSISTENCE_TIME_SEC,
            "cooldown_active": self.in_cooldown,
            "cooldown_counter": round(self.cooldown_remaining, 1),
            "trigger_fired": trigger_fired,
            "scene_id": scene_id,
            "person_id": self.current_person_id,
        }
        msg = String()
        msg.data = json.dumps(data)
        self.dashboard_pub.publish(msg)

    def call_trigger_service(self):
        """Sends trigger request to scene_analysis."""
        if not self.client.service_is_ready():
            self.get_logger().warn(
                '/trigger_scene_analysis service not available. '
                'Ensure scene_analysis node is running.'
            )
            return

        request = TriggerSceneAnalysis.Request()
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
