#!/usr/bin/env python3
# ─────────────────────────────────────────────
# generate_graphs.py
#
# THESIS: Automatic Interaction Triggering
#         for HRI using RGB Vision
#
# Author: Athul Sukesh Nair
# FAU Erlangen-Nürnberg, FAPS Chair
# Supervisor: Prof. Sebastian Reitelshöfer
#
# Description:
# Generates thesis-ready graphs from evaluation
# data collected by data_collector.py
#
# Run: python3 generate_graphs.py
# Output: ~/thesis_evaluation/graphs/
# ─────────────────────────────────────────────

import os
import json
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime

# ── Paths ────────────────────────────────────
BASE_DIR   = os.path.expanduser("~/thesis_evaluation")
CSV_FILE   = os.path.join(BASE_DIR, "evaluation_data.csv")
JSON_FILE  = os.path.join(BASE_DIR, "evaluation_summary.json")
GRAPH_DIR  = os.path.join(BASE_DIR, "graphs")
os.makedirs(GRAPH_DIR, exist_ok=True)

# ── Style ────────────────────────────────────
plt.rcParams.update({
    'font.family':   'DejaVu Sans',
    'font.size':     12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi':    150,
    'savefig.dpi':   300,
    'savefig.bbox':  'tight',
})

FAU_BLUE  = '#003865'
FAU_GREEN = '#00c9a7'
FAU_RED   = '#e74c3c'
FAU_GRAY  = '#8899aa'
FAU_LIGHT = '#d0e8f5'


# ── Load Data ────────────────────────────────
def load_data():
    records = []
    with open(CSV_FILE, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert types
            for key in ['yolo_confidence', 'retina_confidence', 'pitch_deg',
                        'yaw_deg', 'time_to_trigger_sec', 'age',
                        'professional_level', 'emotional_state']:
                try:
                    row[key] = float(row[key]) if row[key] not in (None, '', 'None') else None
                except:
                    row[key] = None
            for key in ['person_detected', 'face_detected', 'gaze_looking',
                        'trigger_fired', 'trigger_success']:
                row[key] = str(row[key]).lower() in ('true', '1', 'yes')
            records.append(row)
    return records


def load_summary():
    with open(JSON_FILE) as f:
        return json.load(f)


# ── Graph 1: Trigger Success Rate ─────────────
def plot_trigger_success(records, summary):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        'Trigger Performance Evaluation\nAutomatic Interaction Triggering for HRI',
        fontsize=14, fontweight='bold', color=FAU_BLUE
    )

    # Pie chart
    success = summary['successful_triggers']
    failed  = summary['failed_triggers']
    colors  = [FAU_GREEN, FAU_RED]
    wedges, texts, autotexts = ax1.pie(
        [success, failed],
        labels=['Successful', 'Failed'],
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 12}
    )
    autotexts[0].set_color('white')
    autotexts[1].set_color('white')
    ax1.set_title(
        f'Trigger Success Rate\n(n={summary["total_tests"]} tests)',
        fontweight='bold', color=FAU_BLUE
    )

    # Bar chart per test
    test_ids = [int(r['test_id']) for r in records if r['test_id']]
    successes = [1 if r['trigger_success'] else 0 for r in records if r['test_id']]
    colors_bar = [FAU_GREEN if s else FAU_RED for s in successes]

    ax2.bar(test_ids, successes, color=colors_bar, edgecolor='white', linewidth=0.5)
    ax2.set_xlabel('Test Number')
    ax2.set_ylabel('Result (1=Success, 0=Failed)')
    ax2.set_title('Trigger Result per Test', fontweight='bold', color=FAU_BLUE)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Failed', 'Success'])
    ax2.set_xticks(test_ids)
    ax2.grid(axis='y', alpha=0.3)

    success_patch = mpatches.Patch(color=FAU_GREEN, label='Success')
    failed_patch  = mpatches.Patch(color=FAU_RED,   label='Failed')
    ax2.legend(handles=[success_patch, failed_patch])

    plt.tight_layout()
    path = os.path.join(GRAPH_DIR, '1_trigger_success_rate.png')
    plt.savefig(path)
    plt.close()
    print(f"✅ Saved: {path}")


# ── Graph 2: Time to Trigger ──────────────────
def plot_time_to_trigger(records, summary):
    times = [r['time_to_trigger_sec'] for r in records
             if r['trigger_success'] and r['time_to_trigger_sec']]

    if not times:
        print("⚠️  No successful trigger times to plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        'Time to Trigger Analysis\nAutomatic Interaction Triggering for HRI',
        fontsize=14, fontweight='bold', color=FAU_BLUE
    )

    # Histogram
    ax1.hist(times, bins=10, color=FAU_BLUE, edgecolor='white', alpha=0.85)
    ax1.axvline(summary['avg_time_to_trigger'], color=FAU_GREEN,
                linestyle='--', linewidth=2, label=f'Mean: {summary["avg_time_to_trigger"]}s')
    ax1.axvline(1.0, color=FAU_RED, linestyle='--', linewidth=2,
                label='Persistence threshold: 1.0s')
    ax1.set_xlabel('Time to Trigger (seconds)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Trigger Times', fontweight='bold', color=FAU_BLUE)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Line plot over tests
    test_ids = [int(r['test_id']) for r in records
                if r['trigger_success'] and r['time_to_trigger_sec']]
    ax2.plot(test_ids, times, color=FAU_BLUE, marker='o',
             linewidth=2, markersize=6, label='Time to trigger')
    ax2.axhline(summary['avg_time_to_trigger'], color=FAU_GREEN,
                linestyle='--', linewidth=2,
                label=f'Mean: {summary["avg_time_to_trigger"]}s')
    ax2.axhline(1.0, color=FAU_RED, linestyle='--', linewidth=2,
                label='Persistence threshold: 1.0s')
    ax2.set_xlabel('Test Number')
    ax2.set_ylabel('Time to Trigger (seconds)')
    ax2.set_title('Trigger Time per Test', fontweight='bold', color=FAU_BLUE)
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Stats box
    stats = (f"Min:  {summary['min_time_to_trigger']}s\n"
             f"Max:  {summary['max_time_to_trigger']}s\n"
             f"Mean: {summary['avg_time_to_trigger']}s")
    ax2.text(0.02, 0.98, stats, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor=FAU_LIGHT, alpha=0.8))

    plt.tight_layout()
    path = os.path.join(GRAPH_DIR, '2_time_to_trigger.png')
    plt.savefig(path)
    plt.close()
    print(f"✅ Saved: {path}")


# ── Graph 3: Gaze Angles ──────────────────────
def plot_gaze_angles(records):
    looking_pitch = [r['pitch_deg'] for r in records
                    if r['gaze_looking'] and r['pitch_deg'] is not None]
    looking_yaw   = [r['yaw_deg'] for r in records
                    if r['gaze_looking'] and r['yaw_deg'] is not None]
    away_pitch    = [r['pitch_deg'] for r in records
                    if not r['gaze_looking'] and r['pitch_deg'] is not None]
    away_yaw      = [r['yaw_deg'] for r in records
                    if not r['gaze_looking'] and r['yaw_deg'] is not None]

    if not looking_pitch:
        print("⚠️  No gaze angle data to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        'Gaze Angle Analysis (L2CS-Net)\nAutomatic Interaction Triggering for HRI',
        fontsize=14, fontweight='bold', color=FAU_BLUE
    )

    # Pitch angles
    if looking_pitch:
        axes[0].hist(looking_pitch, bins=15, color=FAU_GREEN,
                    alpha=0.7, label='Looking at robot', edgecolor='white')
    if away_pitch:
        axes[0].hist(away_pitch, bins=15, color=FAU_RED,
                    alpha=0.7, label='Looking away', edgecolor='white')
    axes[0].axvline(-30, color='black', linestyle='--', linewidth=1.5,
                   label='Threshold: ±30°')
    axes[0].axvline(30, color='black', linestyle='--', linewidth=1.5)
    axes[0].set_xlabel('Pitch Angle (degrees)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Pitch Angle Distribution', fontweight='bold', color=FAU_BLUE)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Yaw angles
    if looking_yaw:
        axes[1].hist(looking_yaw, bins=15, color=FAU_GREEN,
                    alpha=0.7, label='Looking at robot', edgecolor='white')
    if away_yaw:
        axes[1].hist(away_yaw, bins=15, color=FAU_RED,
                    alpha=0.7, label='Looking away', edgecolor='white')
    axes[1].axvline(-30, color='black', linestyle='--', linewidth=1.5,
                   label='Threshold: ±30°')
    axes[1].axvline(30, color='black', linestyle='--', linewidth=1.5)
    axes[1].set_xlabel('Yaw Angle (degrees)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Yaw Angle Distribution', fontweight='bold', color=FAU_BLUE)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(GRAPH_DIR, '3_gaze_angles.png')
    plt.savefig(path)
    plt.close()
    print(f"✅ Saved: {path}")


# ── Graph 4: YOLO Confidence ──────────────────
def plot_yolo_confidence(records, summary):
    confidences = [r['yolo_confidence'] for r in records
                  if r['yolo_confidence'] is not None]

    if not confidences:
        print("⚠️  No YOLO confidence data to plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        'Person Detection Confidence (YOLOv8n)\nAutomatic Interaction Triggering for HRI',
        fontsize=14, fontweight='bold', color=FAU_BLUE
    )

    # Histogram
    ax1.hist(confidences, bins=15, color=FAU_BLUE,
            edgecolor='white', alpha=0.85)
    ax1.axvline(summary['avg_yolo_confidence'], color=FAU_GREEN,
               linestyle='--', linewidth=2,
               label=f'Mean: {summary["avg_yolo_confidence"]}%')
    ax1.axvline(50, color=FAU_RED, linestyle='--', linewidth=2,
               label='Min threshold: 50%')
    ax1.set_xlabel('YOLO Confidence (%)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('YOLO Confidence Distribution', fontweight='bold', color=FAU_BLUE)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Over time
    test_ids = list(range(1, len(confidences) + 1))
    ax2.plot(test_ids, confidences, color=FAU_BLUE, marker='o',
            linewidth=2, markersize=6)
    ax2.axhline(summary['avg_yolo_confidence'], color=FAU_GREEN,
               linestyle='--', linewidth=2,
               label=f'Mean: {summary["avg_yolo_confidence"]}%')
    ax2.fill_between(test_ids, confidences, alpha=0.1, color=FAU_BLUE)
    ax2.set_xlabel('Test Number')
    ax2.set_ylabel('YOLO Confidence (%)')
    ax2.set_title('YOLO Confidence per Test', fontweight='bold', color=FAU_BLUE)
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_ylim([0, 105])

    plt.tight_layout()
    path = os.path.join(GRAPH_DIR, '4_yolo_confidence.png')
    plt.savefig(path)
    plt.close()
    print(f"✅ Saved: {path}")


# ── Graph 5: Agent Selection ──────────────────
def plot_agent_selection(summary):
    agents = summary.get('agent_selection_counts', {})
    if not agents:
        print("⚠️  No agent selection data to plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        'Agent Selection Distribution\nMarketplace Bidding Platform',
        fontsize=14, fontweight='bold', color=FAU_BLUE
    )

    names  = list(agents.keys())
    counts = list(agents.values())
    colors = [FAU_BLUE, FAU_GREEN, '#1a8fe3', '#f5a623', '#9b59b6'][:len(names)]

    bars = ax.bar(names, counts, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Agent')
    ax.set_ylabel('Number of Times Selected')
    ax.set_title('Agent Selection Frequency', fontweight='bold', color=FAU_BLUE)
    ax.grid(axis='y', alpha=0.3)

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.1,
               str(count), ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    path = os.path.join(GRAPH_DIR, '5_agent_selection.png')
    plt.savefig(path)
    plt.close()
    print(f"✅ Saved: {path}")


# ── Graph 6: Pipeline Summary ─────────────────
def plot_pipeline_summary(records, summary):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        'Pipeline Performance Summary\nAutomatic Interaction Triggering for HRI',
        fontsize=14, fontweight='bold', color=FAU_BLUE
    )

    metrics = [
        ('Total Tests',           summary['total_tests'],                  FAU_BLUE),
        ('Successful Triggers',   summary['successful_triggers'],          FAU_GREEN),
        ('Failed Triggers',       summary['failed_triggers'],              FAU_RED),
        ('Avg YOLO Conf (%)',     summary['avg_yolo_confidence'],          '#1a8fe3'),
        ('Avg Time to Trigger(s)',summary['avg_time_to_trigger'],          '#f5a623'),
        ('Success Rate (%)',      summary['success_rate_percent'],         '#9b59b6'),
    ]

    names  = [m[0] for m in metrics]
    values = [m[1] for m in metrics]
    colors = [m[2] for m in metrics]

    bars = ax.barh(names, values, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Value')
    ax.set_title('Overall System Performance', fontweight='bold', color=FAU_BLUE)
    ax.grid(axis='x', alpha=0.3)

    for bar, value in zip(bars, values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2.,
               str(value), ha='left', va='center', fontweight='bold')

    # Add thesis info
    info = (f"Thesis: Automatic Interaction Triggering for HRI Using RGB Vision\n"
            f"Author: Athul Sukesh Nair | FAU Erlangen-Nürnberg | FAPS Chair\n"
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    fig.text(0.5, 0.01, info, ha='center', fontsize=9, color=FAU_GRAY,
            style='italic')

    plt.tight_layout()
    path = os.path.join(GRAPH_DIR, '6_pipeline_summary.png')
    plt.savefig(path)
    plt.close()
    print(f"✅ Saved: {path}")


# ── Main ─────────────────────────────────────
def main():
    print("=" * 50)
    print("THESIS GRAPH GENERATOR")
    print("Automatic Interaction Triggering for HRI")
    print("=" * 50)

    # Check files exist
    if not os.path.exists(CSV_FILE):
        print(f"❌ CSV not found: {CSV_FILE}")
        print("Run data_collector first to collect data!")
        return

    if not os.path.exists(JSON_FILE):
        print(f"❌ Summary not found: {JSON_FILE}")
        return

    # Load data
    print(f"\n📂 Loading data from: {CSV_FILE}")
    records = load_data()
    summary = load_summary()

    print(f"📊 Loaded {len(records)} test records")
    print(f"✅ Success rate: {summary['success_rate_percent']}%")
    print(f"⏱️  Avg time to trigger: {summary['avg_time_to_trigger']}s")

    # Generate graphs
    print(f"\n🎨 Generating graphs in: {GRAPH_DIR}")
    print("-" * 50)

    plot_trigger_success(records, summary)
    plot_time_to_trigger(records, summary)
    plot_gaze_angles(records)
    plot_yolo_confidence(records, summary)
    plot_agent_selection(summary)
    plot_pipeline_summary(records, summary)

    print("-" * 50)
    print(f"\n🎉 All graphs saved to: {GRAPH_DIR}")
    print("\nGraphs generated:")
    print("  1. Trigger Success Rate")
    print("  2. Time to Trigger Analysis")
    print("  3. Gaze Angle Distribution")
    print("  4. YOLO Confidence Scores")
    print("  5. Agent Selection Distribution")
    print("  6. Pipeline Performance Summary")
    print("\nReady for thesis! 🎓")


if __name__ == '__main__':
    main()
