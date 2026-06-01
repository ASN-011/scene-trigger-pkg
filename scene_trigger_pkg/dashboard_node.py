# ─────────────────────────────────────────────
# dashboard_node.py
#
# THESIS: Automatic Interaction Triggering
#         for HRI using RGB Vision
#
# Author: Athul Sukesh Nair
# FAU Erlangen-Nürnberg, FAPS Chair
# Supervisor: Prof. Sebastian Reitelshöfer
#
# Description:
# Web dashboard that visualizes the complete
# trigger pipeline in real time.
# Open browser at http://localhost:5000
# ─────────────────────────────────────────────

import json
import threading
import time
from datetime import datetime
from flask import Flask, Response, render_template_string
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


# ── HTML Dashboard Template ──────────────────
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>HRI Trigger Dashboard</title>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background: #0a1628;
            color: #ffffff;
            padding: 20px;
        }
        h1 {
            font-size: 22px;
            color: #1a8fe3;
            margin-bottom: 4px;
        }
        .subtitle {
            font-size: 12px;
            color: #8899aa;
            margin-bottom: 20px;
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 16px;
            margin-bottom: 16px;
        }
        .card {
            background: #0d2a4a;
            border: 1px solid #1a3a5a;
            border-radius: 8px;
            padding: 16px;
        }
        .card h2 {
            font-size: 11px;
            color: #8899aa;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }
        .status-dot {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .green { background: #00c9a7; }
        .red { background: #e74c3c; }
        .yellow { background: #f5a623; }
        .blue { background: #1a8fe3; }
        .value {
            font-size: 32px;
            font-weight: bold;
            color: #1a8fe3;
        }
        .value.green-text { color: #00c9a7; }
        .value.yellow-text { color: #f5a623; }
        .label {
            font-size: 11px;
            color: #8899aa;
            margin-top: 4px;
        }
        .progress-bar {
            background: #1a3a5a;
            border-radius: 4px;
            height: 12px;
            margin-top: 8px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            background: #1a8fe3;
            border-radius: 4px;
            transition: width 0.3s;
        }
        .gaze-box {
            display: flex;
            justify-content: space-around;
            margin-top: 8px;
        }
        .gaze-item {
            text-align: center;
        }
        .gaze-value {
            font-size: 24px;
            font-weight: bold;
        }
        .log-box {
            background: #071a35;
            border-radius: 4px;
            padding: 10px;
            max-height: 180px;
            overflow-y: auto;
            font-family: monospace;
            font-size: 11px;
        }
        .log-entry {
            padding: 3px 0;
            border-bottom: 1px solid #1a3a5a;
            color: #8899aa;
        }
        .log-entry.trigger {
            color: #00c9a7;
            font-weight: bold;
        }
        .log-entry.warn {
            color: #f5a623;
        }
        .pipeline {
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
            margin-top: 8px;
        }
        .pipe-step {
            background: #1a3a5a;
            border-radius: 4px;
            padding: 6px 10px;
            font-size: 11px;
            color: #8899aa;
            border: 1px solid #1a3a5a;
        }
        .pipe-step.active {
            background: #0d3b6e;
            border-color: #00c9a7;
            color: #00c9a7;
        }
        .pipe-arrow {
            color: #1a3a5a;
            font-size: 14px;
        }
        .footer {
            text-align: center;
            color: #8899aa;
            font-size: 11px;
            margin-top: 16px;
            padding-top: 12px;
            border-top: 1px solid #1a3a5a;
        }
        .scene-params {
            display: flex;
            justify-content: space-around;
            margin-top: 8px;
        }
        .param-item {
            text-align: center;
        }
        .param-value {
            font-size: 22px;
            font-weight: bold;
            color: #00c9a7;
        }
        .param-label {
            font-size: 10px;
            color: #8899aa;
        }
    </style>
</head>
<body>
    <h1>HRI Automatic Interaction Trigger — Live Dashboard</h1>
    <p class="subtitle">
        Athul Sukesh Nair &nbsp;·&nbsp;
        FAU Erlangen-Nürnberg &nbsp;·&nbsp;
        FAPS Chair &nbsp;·&nbsp;
        Supervisor: Prof. Sebastian Reitelshöfer &nbsp;·&nbsp;
        Last update: {{ data.timestamp }}
    </p>

    <!-- Pipeline Status -->
    <div class="card" style="margin-bottom: 16px;">
        <h2>Pipeline Status</h2>
        <div class="pipeline">
            <div class="pipe-step {{ 'active' if data.person_detected else '' }}">
                Camera + YOLO
            </div>
            <div class="pipe-arrow">→</div>
            <div class="pipe-step {{ 'active' if data.face_detected else '' }}">
                RetinaFace
            </div>
            <div class="pipe-arrow">→</div>
            <div class="pipe-step {{ 'active' if data.looking_at_camera else '' }}">
                L2CS-Net Gaze
            </div>
            <div class="pipe-arrow">→</div>
            <div class="pipe-step {{ 'active' if data.persistence_counter > 0 else '' }}">
                Persistence Counter
            </div>
            <div class="pipe-arrow">→</div>
            <div class="pipe-step {{ 'active' if data.trigger_fired else '' }}">
                Trigger Service
            </div>
            <div class="pipe-arrow">→</div>
            <div class="pipe-step {{ 'active' if data.scene_id > 0 else '' }}">
                Marketplace
            </div>
        </div>
    </div>

    <!-- Main Grid -->
    <div class="grid">

        <!-- Person Detection -->
        <div class="card">
            <h2>Person Detection (YOLO)</h2>
            {% if data.person_detected %}
                <span class="status-dot green"></span>
                <span style="color: #00c9a7; font-weight: bold;">Person Detected</span>
                <div class="value green-text">{{ data.yolo_confidence }}%</div>
                <div class="label">Confidence</div>
            {% else %}
                <span class="status-dot red"></span>
                <span style="color: #e74c3c;">No Person</span>
                <div class="value" style="color: #e74c3c;">--</div>
                <div class="label">Waiting...</div>
            {% endif %}
        </div>

        <!-- Gaze Detection -->
        <div class="card">
            <h2>Gaze Detection (L2CS-Net)</h2>
            {% if data.face_detected %}
                <span class="status-dot {{ 'green' if data.looking_at_camera else 'yellow' }}"></span>
                <span style="color: {{ '#00c9a7' if data.looking_at_camera else '#f5a623' }}; font-weight: bold;">
                    {{ 'Looking at Robot' if data.looking_at_camera else 'Looking Away' }}
                </span>
                <div class="gaze-box" style="margin-top: 10px;">
                    <div class="gaze-item">
                        <div class="gaze-value" style="color: #1a8fe3;">{{ data.pitch_deg }}</div>
                        <div class="label">Pitch (deg)</div>
                    </div>
                    <div class="gaze-item">
                        <div class="gaze-value" style="color: #1a8fe3;">{{ data.yaw_deg }}</div>
                        <div class="label">Yaw (deg)</div>
                    </div>
                </div>
            {% else %}
                <span class="status-dot red"></span>
                <span style="color: #e74c3c;">No Face Detected</span>
                <div class="gaze-box" style="margin-top: 10px;">
                    <div class="gaze-item">
                        <div class="gaze-value" style="color: #8899aa;">--</div>
                        <div class="label">Pitch (deg)</div>
                    </div>
                    <div class="gaze-item">
                        <div class="gaze-value" style="color: #8899aa;">--</div>
                        <div class="label">Yaw (deg)</div>
                    </div>
                </div>
            {% endif %}
        </div>

        <!-- Persistence Counter -->
        <div class="card">
            <h2>Persistence Counter</h2>
            {% if data.cooldown_active %}
                <span class="status-dot yellow"></span>
                <span style="color: #f5a623; font-weight: bold;">Cooldown Active</span>
                <div class="value yellow-text">{{ data.cooldown_counter }}</div>
                <div class="label">frames remaining</div>
            {% else %}
                <div class="value">{{ data.persistence_counter }}/{{ data.persistence_threshold }}</div>
                <div class="label">
                    ~{{ (data.persistence_counter / 3.0) | round(1) }}s
                    / ~{{ (data.persistence_threshold / 3.0) | round(1) }}s
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {{ (data.persistence_counter / data.persistence_threshold * 100) | int }}%;"></div>
                </div>
            {% endif %}
        </div>

    </div>

    <!-- Scene Parameters + Trigger Log -->
    <div class="grid" style="grid-template-columns: 1fr 2fr;">

        <!-- Scene Parameters -->
        <div class="card">
            <h2>Last Scene Parameters (DeepFace)</h2>
            {% if data.scene_id > 0 %}
                <div style="margin-bottom: 8px;">
                    <span class="status-dot green"></span>
                    <span style="color: #00c9a7;">Scene ID: {{ data.scene_id }}</span>
                </div>
                <div class="scene-params">
                    <div class="param-item">
                        <div class="param-value">{{ data.age }}</div>
                        <div class="param-label">Age</div>
                    </div>
                    <div class="param-item">
                        <div class="param-value">{{ data.prof_level }}</div>
                        <div class="param-label">Prof. Level</div>
                    </div>
                    <div class="param-item">
                        <div class="param-value">{{ data.emotion }}</div>
                        <div class="param-label">Emotion</div>
                    </div>
                </div>
                <div style="margin-top: 10px; font-size: 11px; color: #00c9a7;">
                    Agent: {{ data.winner_agent }}
                </div>
            {% else %}
                <div style="color: #8899aa; font-size: 12px; margin-top: 10px;">
                    Waiting for first trigger...
                </div>
            {% endif %}
        </div>

        <!-- Trigger Log -->
        <div class="card">
            <h2>Trigger Event Log</h2>
            <div class="log-box">
                {% for entry in data.log %}
                    <div class="log-entry {{ entry.type }}">
                        {{ entry.time }} &nbsp; {{ entry.message }}
                    </div>
                {% endfor %}
                {% if not data.log %}
                    <div class="log-entry">Waiting for events...</div>
                {% endif %}
            </div>
        </div>

    </div>

    <!-- Stats Row -->
    <div class="grid" style="grid-template-columns: repeat(4, 1fr);">
        <div class="card" style="text-align: center;">
            <h2>Total Triggers</h2>
            <div class="value green-text">{{ data.total_triggers }}</div>
        </div>
        <div class="card" style="text-align: center;">
            <h2>YOLO Threshold</h2>
            <div class="value">{{ data.persistence_threshold }} frames</div>
            <div class="label">~1.0 second at 3fps</div>
        </div>
        <div class="card" style="text-align: center;">
            <h2>Gaze Threshold</h2>
            <div class="value">30 deg</div>
            <div class="label">Cheng et al. (2026)</div>
        </div>
        <div class="card" style="text-align: center;">
            <h2>System Status</h2>
            <div class="value green-text" style="font-size: 20px;">RUNNING</div>
            <div class="label">All nodes active</div>
        </div>
    </div>

    <div class="footer">
        Automatic Interaction Triggering for HRI Using RGB Vision &nbsp;·&nbsp;
        Master's Thesis &nbsp;·&nbsp;
        FAU Erlangen-Nürnberg &nbsp;·&nbsp;
        FAPS Chair
    </div>

</body>
</html>
"""


# ── Shared State ─────────────────────────────
state = {
    "timestamp": "--",
    "person_detected": False,
    "yolo_confidence": 0.0,
    "face_detected": False,
    "looking_at_camera": False,
    "pitch_deg": 0.0,
    "yaw_deg": 0.0,
    "persistence_counter": 0,
    "persistence_threshold": 3,
    "cooldown_active": False,
    "cooldown_counter": 0,
    "trigger_fired": False,
    "scene_id": 0,
    "age": "--",
    "prof_level": "--",
    "emotion": "--",
    "winner_agent": "--",
    "total_triggers": 0,
    "log": []
}


def add_log(message, log_type="normal"):
    """Add entry to event log."""
    entry = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "message": message,
        "type": log_type
    }
    state["log"].insert(0, entry)
    # Keep only last 20 entries
    state["log"] = state["log"][:20]


# ── Flask App ────────────────────────────────
app = Flask(__name__)


@app.route('/')
def dashboard():
    return render_template_string(HTML, data=state)


# ── ROS2 Node ────────────────────────────────
class DashboardNode(Node):

    def __init__(self):
        super().__init__('dashboard_node')

        # Subscribe to dashboard updates from trigger node
        self.create_subscription(
            String,
            '/dashboard_update',
            self.dashboard_callback,
            10
        )

        self.get_logger().info(
            'Dashboard node started. '
            'Open browser at http://localhost:5000'
        )

    def dashboard_callback(self, msg):
        """Receives JSON updates from scene_trigger_node."""
        try:
            data = json.loads(msg.data)
            state.update(data)
            state["timestamp"] = datetime.now().strftime("%H:%M:%S")

            # Add to log if trigger fired
            if data.get("trigger_fired"):
                state["total_triggers"] += 1
                add_log(
                    f'Trigger fired! Scene ID: {data.get("scene_id", -1)}',
                    "trigger"
                )
            elif data.get("person_detected") and not data.get("looking_at_camera"):
                pass  # Don't log every frame
            elif data.get("person_detected") and data.get("looking_at_camera"):
                pass  # Don't log every frame

        except Exception as e:
            self.get_logger().error(f'Dashboard update error: {e}')


def run_flask():
    """Run Flask web server in background thread."""
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)


def main(args=None):
    rclpy.init(args=args)
    node = DashboardNode()

    # Start Flask in background thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    add_log('Dashboard started. Waiting for data...', 'normal')

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
