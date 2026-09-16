#!/usr/bin/env python3
"""Generate comparison graph for nelssa vs baseline performance."""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Use default matplotlib font (DejaVu Sans)

# ============================================================================
# CONFIGURATION - Modify these variables to customize the graph
# ============================================================================

# Data input (replace with your actual data)
DATA = {
    # 'long_request_ratio': [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
    # 'decode_throughput_nelssa': [1.00, 1.00, 0.91, 0.75, 0.36, 0.27, 0.23, 0.17],
    # 'decode_throughput_baseline': [1.00, 0.92, 0.56, 0.31, 0.23, 0.14, 0.12, 0.12],
    # 'gpu_util_nelssa': [24.61, 23.65, 23.39, 31.12, 63.88, 63.09, 66.71, 64.95],
    # 'pnm_util_nelssa': [0, 0, 0.14, 4.62, 18.4, 27, 29.72, 38.66],
    # 'gpu_util_baseline': [23.42, 26.08, 55.94, 94.5, 79.32, 83.35, 93.87, 92.48],
    
    'long_request_ratio': [0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0],
    'decode_throughput_nelssa': [1.00, 1.00, 0.91, 0.75, 0.36, 0.27, 0.23],
    'decode_throughput_baseline': [1.00, 0.92, 0.56, 0.31, 0.23, 0.14, 0.12],
    'gpu_util_nelssa': [24.61, 23.65, 23.39, 31.12, 63.88, 63.09, 66.71,],
    'pnm_util_nelssa': [0, 0, 0.14, 4.62, 18.4, 27, 29.72,],
    'gpu_util_baseline': [23.42, 26.08, 55.94, 94.5, 79.32, 83.35, 93.87,],
}

# Color configuration
COLORS = {
    'nelssa_throughput': '#8B0000',
    'baseline_throughput': 'black',
    'nelssa_gpu': '#CD5C5C',
    'nelssa_pnm': '#8B0000',
    'baseline_gpu': 'Grey',
}

# Plot settings
BAR_GAP_FRACTION = 0.5  # Gap between bars as fraction of bar width
DECODE_THROUGHPUT_MAX = 1.05  # Max value for left y-axis (with margin above 1.0)
DECODE_THROUGHPUT_TICK_MAX = 1.0  # Max tick label for left y-axis
# ============================================================================

# Load data
df = pd.DataFrame(DATA)

# Create figure and axis (single column width: ~8.5cm)
fig, ax1 = plt.subplots(figsize=(3.35, 1.25))

# Adjust graph box position (leave space for legend at top)
# [left, bottom, width, height] in figure coordinates (0-1)
ax1.set_position([0.12, 0.10, 0.75, 0.05])

x = np.arange(len(df))

# Calculate bar width with gap
bar_width = 0.3
# gap = bar_width * BAR_GAP_FRACTION
gap = 0

# Plot decode throughput (line chart) - Left axis (draw FIRST)
ax1.set_ylabel('Decode Thr.\n(Normalized)', fontsize=7, color='black')
ax1.yaxis.set_label_coords(-0.15, 0.5)
ax1.set_ylim(0, DECODE_THROUGHPUT_MAX)
ax1.set_yticks(np.arange(0, DECODE_THROUGHPUT_TICK_MAX + 0.1, 0.2))
ax1.tick_params(axis='y', labelcolor='black', labelsize=6, width=0.5)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Plot decode throughput lines on ax1 (draw FIRST)
line1 = ax1.plot(x, df['decode_throughput_nelssa'], marker='o', linewidth=1,
                 markersize=3.5, label='NELSSA', color=COLORS['nelssa_throughput'])
line2 = ax1.plot(x, df['decode_throughput_baseline'], marker='s', linewidth=1,
                 markersize=3.5, label='Baseline', color=COLORS['baseline_throughput'])

# Create secondary y-axis for memory utilization (bar chart) (draw SECOND)
ax2 = ax1.twinx()
ax2.set_ylabel('Mem util (%)', fontsize=7, color='black')
ax2.yaxis.set_label_coords(1.15, 0.5)
ax2.set_ylim(0, 105)
ax2.set_yticks(np.arange(0, 101, 20))
ax2.tick_params(axis='y', labelcolor='black', labelsize=6, width=0.5)

# Align y-axis ticks: move ax2 ticks to the right side
ax2.yaxis.tick_right()

# Force ax1 to render on top of ax2
ax1.zorder = 10
ax2.zorder = 1
ax1.patch.set_alpha(0)

# Set spine width
for spine in ax1.spines.values():
    spine.set_linewidth(0.5)
for spine in ax2.spines.values():
    spine.set_linewidth(0.5)

# Plot memory utilization (bar chart) - Right axis (draw AFTER lines, with lower alpha)
offset = -bar_width - gap / 2

# Baseline GPU (only GPU, no PNM)
ax2.bar(x + offset, df['gpu_util_baseline'], width=bar_width,
        label='Baseline GPU', color=COLORS['baseline_gpu'], alpha=0.3, edgecolor='black', linewidth=0.3)
offset += bar_width + gap

# Nelssa GPU
ax2.bar(x + offset, df['gpu_util_nelssa'], width=bar_width,
        label='NELSSA GPU', color=COLORS['nelssa_gpu'], alpha=0.3, edgecolor='black', linewidth=0.3)
offset += bar_width

# Nelssa PNM
ax2.bar(x + offset, df['pnm_util_nelssa'], width=bar_width,
        label='NELSSA PNM', color=COLORS['nelssa_pnm'], alpha=0.3, edgecolor='black', linewidth=0.3)

ax1.set_xlabel('Long Request Ratio (%)', fontsize=7)
ax1.set_xticks(x)
ax1.set_xticklabels(df['long_request_ratio'], fontsize=6)
ax1.tick_params(axis='x', labelsize=6, width=0.5)

# Combine legends from both axes
# First row: Throughput labels (Baseline, NELSSA)
# Second row: Utilization labels (Baseline GPU, NELSSA GPU, NELSSA PNM)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

legend_handles = [
    Patch(color=COLORS['baseline_gpu'], alpha=0.3, edgecolor='black', label='Baseline GPU'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['baseline_throughput'], markersize=6, label='Baseline Throughput'),
    Patch(color=COLORS['nelssa_gpu'], alpha=0.3, edgecolor='black', label='NELSSA GPU'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['nelssa_throughput'], markersize=6, label='NELSSA Throughput'),
    Patch(color=COLORS['nelssa_pnm'], alpha=0.3, edgecolor='black', label='NELSSA PNM'),
]

label_box = ['Baseline', 'NELSSA', 'Baseline GPU', 'NELSSA GPU', 'NELSSA PNM']

# Add single legend at top center of the graph (2 rows: Throughput on top, Util on bottom)
leg = ax1.legend(legend_handles,
                 [h.get_label() for h in legend_handles],
                 loc='upper center', bbox_to_anchor=(0.5, 1.6),
                 fontsize=6, frameon=True, fancybox=False, ncol=3, columnspacing=0.5, handletextpad=0.2)
leg.get_frame().set_linewidth(0.3)
leg.get_frame().set_edgecolor('#505050')
leg.get_frame().set_facecolor('white')

# Remove title

# Adjust layout to prevent legend cutoff
plt.tight_layout()

# Save figure
plt.savefig('nelssa_baseline_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('nelssa_baseline_comparison.pdf', bbox_inches='tight')
print("Saved: nelssa_baseline_comparison.png and nelssa_baseline_comparison.pdf")

# Show plot
plt.show()
