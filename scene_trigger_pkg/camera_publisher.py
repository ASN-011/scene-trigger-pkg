# ─────────────────────────────────────────────
# camera_publisher.py
#
# THESIS: Automatic Interaction Triggering
#         for HRI using RGB Vision
#
# Author: Athul Sukesh Nair
# FAU Erlangen-Nürnberg, FAPS Chair
# Supervisor: Prof. Sebastian Reitelshöfer
#
# Description:
# Reads frames from the laptop webcam and publishes
# them as ROS2 Image messages on the /image_raw topic.
#
# This node was written as a replacement for the
# standard ros-jazzy-usb_cam package which had
# compatibility issues with the HP True Vision FHD
# Camera on Ubuntu 24.04 with ROS2 Jazzy.
#
# The /image_raw topic is consumed by:
# - scene_trigger_node.py (YOLO person detection)
# - deepface_node (face analysis)
# ─────────────────────────────────────────────

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class CameraPublisher(Node):

    def __init__(self):
        super().__init__('camera_publisher')

        # ── Publisher ────────────────────────────
        # Publishes on /image_raw at queue size 10.
        # Both scene_trigger_node and deepface_node
        # subscribe to this topic.
        self.publisher = self.create_publisher(
            Image,
            '/image_raw',
            10
        )

        # ── cv_bridge ───────────────────────────
        # Converts OpenCV images to ROS2 Image messages.
        self.bridge = CvBridge()

        # ── Camera ──────────────────────────────
        # Opens /dev/video0 — HP True Vision FHD Camera.
        # VideoCapture(0) directly reads the webcam
        # without relying on usb_cam driver.
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            self.get_logger().error(
                'Failed to open camera on /dev/video0.'
            )
            return

        self.get_logger().info(
            'Camera opened successfully on /dev/video0.'
        )

        # ── Timer ───────────────────────────────
        # Publishes a frame every 0.1 seconds (~10fps).
        # Actual published rate is approximately 3fps
        # due to camera hardware limitations.
        self.timer = self.create_timer(0.1, self.publish_frame)

        self.get_logger().info(
            'Publishing camera frames on /image_raw.'
        )

    def publish_frame(self):
        """
        Timer callback that reads one frame from the
        webcam and publishes it as a ROS2 Image message.
        """
        ret, frame = self.cap.read()

        if ret:
            msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
            self.publisher.publish(msg)
            self.get_logger().info(
                'Publishing on /image_raw.',
                once=True
            )
        else:
            self.get_logger().warn(
                'Failed to read frame from camera.'
            )

    def destroy_node(self):
        """
        Releases the camera resource before shutting down.
        """
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

