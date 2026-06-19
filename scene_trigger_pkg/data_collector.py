#!/usr/bin/env python3
# ─────────────────────────────────────────────
# data_collector.py
#
# THESIS: Automatic Interaction Triggering
#         for HRI using RGB Vision
#
# Author: Athul Sukesh Nair
# FAU Erlangen-Nürnberg, FAPS Chair
# Supervisor: Prof. Sebastian Reitelshöfer
#
# Description:
# Automatically collects evaluation data from
# the trigger pipeline and saves to CSV + JSON
# for thesis graphs and analysis.
#
# Run: ros2 run scene_trigger_pkg data_collector
# Output: ~/thesis_evaluation/evaluation_data.csv
# ─────────────────────────────────────────────

import csv
import json
import os
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from social_marketplace_interfaces.msg import SceneParameters


# ── Output directory ─────────────────────────
OUTPUT_DIR = os.path.expanduser("~/thesis_evaluation")
CSV_FILE   = os.path.join(OUTPUT_DIR, "evaluation_data.csv")
JSON_FILE  = os.path.join(OUTPUT_DIR, "evaluation_summary.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── CSV columns ──────────────────────────────
CSV_COLUMNS = [
    "test_id",
    "timestamp",
    "person_detected",
    "yolo_confidence",
    "face_detected",
    "gaze_looking",
    "pitch_deg",
    "yaw_deg",
    "person_id",
    "trigger_fired",
    "scene_id",
    "trigger_success",
    "time_to_trigger_sec",
    "age",
    "professional_level",
    "emotional_state",
    "winner_agent",
]


class DataCollectorNode(Node):

    def __init__(self):
        super().__init__('data_collector')

        # ── State tracking ───────────────────
        self.test_id            = 0
        self.person_first_seen  = None
        self.current_data       = self._empty_record()
        self.records            = []
        self.session_start      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ── Latest scene/agent data ──────────
        self.latest_age               = None
        self.latest_professional_level = None
        self.latest_emotional_state   = None
        self.latest_winner_agent      = None

        # ── Subscribe to dashboard updates ───
        self.create_subscription(
            String,
            '/dashboard_update',
            self.dashboard_callback,
            10
        )

        # ── Subscribe to scene parameters ────
        self.create_subscription(
            SceneParameters,
            '/scene_parameters',
            self.scene_parameters_callback,
            10
        )

        # ── Subscribe to current agent ───────
        self.create_subscription(
            String,
            '/current_feedback_agent',
            self.agent_callback,
            10
        )

        # ── Initialize CSV ───────────────────
        self._init_csv()

        self.get_logger().info("=" * 50)
        self.get_logger().info("DATA COLLECTOR STARTED")
        self.get_logger().info(f"Output: {CSV_FILE}")
        self.get_logger().info("=" * 50)
        self.get_logger().info("Waiting for trigger events...")

    def _empty_record(self):
        return {col: None for col in CSV_COLUMNS}

    def _init_csv(self):
        if not os.path.exists(CSV_FILE):
            with open(CSV_FILE, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writeheader()
            self.get_logger().info(f"Created CSV: {CSV_FILE}")
        else:
            self.get_logger().info(f"Appending to CSV: {CSV_FILE}")

    def scene_parameters_callback(self, msg):
        """Receives scene parameters from scene_analysis."""
        try:
            names  = msg.parameter_names
            values = msg.parameter_values
            params = dict(zip(names, values))
            self.latest_age                = round(params.get('age', 0), 2)
            self.latest_professional_level = round(params.get('professional_level', 0), 2)
            self.latest_emotional_state    = round(params.get('emotional_state', 0), 2)
            self.get_logger().info(
                f"Scene params: age={self.latest_age}, "
                f"prof={self.latest_professional_level}, "
                f"emotion={self.latest_emotional_state}"
            )
        except Exception as e:
            self.get_logger().error(f"Scene params error: {e}")

    def agent_callback(self, msg):
        """Receives winner agent from feedback server."""
        try:
            data = json.loads(msg.data)
            self.latest_winner_agent = data.get('agent_name', 'N/A')
            self.get_logger().info(f"Winner agent: {self.latest_winner_agent}")

            # Update current record if trigger already fired
            if self.current_data.get('trigger_fired'):
                self.current_data['winner_agent'] = self.latest_winner_agent
        except Exception as e:
            self.get_logger().error(f"Agent callback error: {e}")

    def dashboard_callback(self, msg):
        """Process dashboard updates from scene_trigger_node."""
        try:
            data = json.loads(msg.data)
            now  = time.time()

            # ── Track when person first appears ──
            if data.get('person_detected') and self.person_first_seen is None:
                self.person_first_seen = now
                self.test_id += 1
                self.current_data = self._empty_record()
                self.current_data['test_id']         = self.test_id
                self.current_data['timestamp']       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.current_data['person_detected'] = True
                self.get_logger().info(f"[TEST {self.test_id}] Person detected - tracking started")

            # ── Update current data ───────────────
            if data.get('person_detected'):
                self.current_data['yolo_confidence'] = data.get('yolo_confidence')
                self.current_data['face_detected']   = data.get('face_detected')
                self.current_data['gaze_looking']    = data.get('looking_at_camera')
                self.current_data['pitch_deg']       = data.get('pitch_deg')
                self.current_data['yaw_deg']         = data.get('yaw_deg')
                self.current_data['person_id']       = data.get('person_id')

            # ── Trigger fired ─────────────────────
            if data.get('trigger_fired') and self.person_first_seen:
                time_to_trigger = now - self.person_first_seen
                scene_id        = data.get('scene_id', -1)
                success         = scene_id > 0

                self.current_data['trigger_fired']       = True
                self.current_data['scene_id']            = scene_id
                self.current_data['trigger_success']     = success
                self.current_data['time_to_trigger_sec'] = round(time_to_trigger, 2)

                # Add scene parameters
                self.current_data['age']                = self.latest_age
                self.current_data['professional_level'] = self.latest_professional_level
                self.current_data['emotional_state']    = self.latest_emotional_state
                self.current_data['winner_agent']       = self.latest_winner_agent

                # Save record
                self._save_record(self.current_data)

                status = "✅ SUCCESS" if success else "❌ FAILED"
                self.get_logger().info(
                    f"[TEST {self.test_id}] Trigger {status} | "
                    f"Scene ID: {scene_id} | "
                    f"Time: {time_to_trigger:.2f}s | "
                    f"Agent: {self.latest_winner_agent} | "
                    f"Age: {self.latest_age}"
                )

                # Reset for next test
                self.person_first_seen = None

            # ── Person left without triggering ────
            if not data.get('person_detected') and self.person_first_seen:
                if not self.current_data.get('trigger_fired'):
                    self.current_data['trigger_fired']       = False
                    self.current_data['trigger_success']     = False
                    self.current_data['time_to_trigger_sec'] = round(now - self.person_first_seen, 2)
                    self._save_record(self.current_data)
                    self.get_logger().info(
                        f"[TEST {self.test_id}] Person left without triggering"
                    )
                self.person_first_seen = None

        except Exception as e:
            self.get_logger().error(f"Error processing data: {e}")

    def _save_record(self, record):
        """Save record to CSV and update summary."""
        self.records.append(record)

        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writerow(record)

        self._update_summary()

    def _update_summary(self):
        """Calculate and save summary statistics."""
        total      = len(self.records)
        successful = sum(1 for r in self.records if r.get('trigger_success'))
        failed     = total - successful

        # Average time to trigger
        times = [r['time_to_trigger_sec'] for r in self.records
                 if r.get('trigger_success') and r.get('time_to_trigger_sec')]
        avg_time = round(sum(times) / len(times), 2) if times else 0

        # Average YOLO confidence
        confidences = [r['yolo_confidence'] for r in self.records
                      if r.get('yolo_confidence')]
        avg_conf = round(sum(confidences) / len(confidences), 1) if confidences else 0

        # Pitch/yaw when looking
        pitches = [r['pitch_deg'] for r in self.records
                  if r.get('gaze_looking') and r.get('pitch_deg')]
        yaws    = [r['yaw_deg'] for r in self.records
                  if r.get('gaze_looking') and r.get('yaw_deg')]

        # Agent selection counts
        agents = {}
        for r in self.records:
            agent = r.get('winner_agent')
            if agent and agent != 'N/A':
                agents[agent] = agents.get(agent, 0) + 1

        # Age statistics
        ages = [r['age'] for r in self.records
               if r.get('age') is not None]
        avg_age = round(sum(ages) / len(ages), 2) if ages else 0

        summary = {
            "session_start":           self.session_start,
            "last_updated":            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_tests":             total,
            "successful_triggers":     successful,
            "failed_triggers":         failed,
            "success_rate_percent":    round((successful / total * 100), 1) if total > 0 else 0,
            "avg_time_to_trigger":     avg_time,
            "min_time_to_trigger":     round(min(times), 2) if times else 0,
            "max_time_to_trigger":     round(max(times), 2) if times else 0,
            "avg_yolo_confidence":     avg_conf,
            "avg_pitch_when_looking":  round(sum(pitches)/len(pitches), 1) if pitches else 0,
            "avg_yaw_when_looking":    round(sum(yaws)/len(yaws), 1) if yaws else 0,
            "avg_detected_age":        avg_age,
            "agent_selection_counts":  agents,
        }

        with open(JSON_FILE, 'w') as f:
            json.dump(summary, f, indent=2)

        self.get_logger().info(
            f"📊 SUMMARY: {successful}/{total} triggers successful "
            f"({summary['success_rate_percent']}%) | "
            f"Avg time: {avg_time}s | "
            f"Avg YOLO: {avg_conf}%"
        )


def main(args=None):
    rclpy.init(args=args)
    node = DataCollectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Data collection stopped.")
        node.get_logger().info(f"Data saved to: {CSV_FILE}")
        node.get_logger().info(f"Summary saved to: {JSON_FILE}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
