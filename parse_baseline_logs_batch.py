#!/usr/bin/env python3
"""Parse multiple baseline log files and export to Excel with separate sheets."""

import re
import os
import sys

try:
    import pandas as pd
except ImportError:
    print("Please install pandas: pip install pandas openpyxl")
    sys.exit(1)


def parse_baseline_log(log_file):
    """Parse baseline log file and extract stats."""
    rows = []

    # Pattern for Stats line (baseline, enable_flag=False)
    # (Worker pid=1084462) [Stats] Running: 21 | GPU KV Cache: 10.60GB
    stats_pattern = r'\[Stats\] Running: (\d+) \| GPU KV Cache: ([\d.]+)GB'

    # Pattern for Scheduler STATS line
    # (EngineCore pid=1084456) [Scheduler] STATS: Running=23 | Waiting=0
    scheduler_pattern = r'\[Scheduler\] STATS: Running=(\d+) \| Waiting=(\d+)'

    lines = []
    with open(log_file, 'r') as f:
        for line in f:
            lines.append(line)

    # Single pass: track last scheduler waiting and match with Stats
    last_waiting = 0

    for line in lines:
        # Check for Scheduler STATS first (update last_waiting)
        sched_match = re.search(scheduler_pattern, line)
        if sched_match:
            last_waiting = int(sched_match.group(2))
            continue

        # Check for Stats
        match = re.search(stats_pattern, line)
        if match:
            total_reqs = int(match.group(1))
            gpu_kv_gb = float(match.group(2))

            # Use the last seen waiting count
            waiting = last_waiting

            # Calculate memory utilization percentages
            gpu_mem_util = (gpu_kv_gb / 47.82) * 100

            rows.append({
                'Total running requests': total_reqs,
                'Total waiting requests': waiting,
                'GPU KV cache size (GB)': gpu_kv_gb,
                'GPU mem util (%)': round(gpu_mem_util, 2),
            })

    return rows


def get_sheet_name_from_filename(filename):
    """Extract a readable sheet name from the log filename (max 31 chars for Excel)."""
    basename = os.path.basename(filename)
    # Remove extension
    name = os.path.splitext(basename)[0]
    # Remove prefix
    if name.startswith('baseline_'):
        name = name[9:]
    elif name.startswith('nelssa_'):
        name = name[7:]
    # Truncate to 31 chars (Excel limit) and replace dots with underscores
    name = name.replace('.', '_')
    if len(name) > 31:
        name = name[:31]
    return name


def compute_summary_stats(dataframes):
    """Compute summary statistics across all sheets."""
    summary_data = []

    for sheet_name, df in dataframes.items():
        row = {
            'File': sheet_name,
            'Total samples': len(df),
        }

        if 'GPU mem util (%)' in df.columns:
            gpu_vals = df['GPU mem util (%)'].dropna()
            row['GPU mem util (%) - Mean'] = round(gpu_vals.mean(), 2) if len(gpu_vals) > 0 else None
            row['GPU mem util (%) - Median'] = round(gpu_vals.median(), 2) if len(gpu_vals) > 0 else None
            row['GPU mem util (%) - Min'] = round(gpu_vals.min(), 2) if len(gpu_vals) > 0 else None
            row['GPU mem util (%) - Max'] = round(gpu_vals.max(), 2) if len(gpu_vals) > 0 else None

        summary_data.append(row)

    return pd.DataFrame(summary_data)


def main():
    log_dir = 'test_logs_baseline'
    output_file = 'baseline_stats.xlsx'

    if not os.path.isdir(log_dir):
        print(f"Error: Directory '{log_dir}' not found")
        sys.exit(1)

    # Find all .log files
    log_files = sorted([
        os.path.join(log_dir, f)
        for f in os.listdir(log_dir)
        if f.endswith('.log')
    ])

    if not log_files:
        print(f"No .log files found in '{log_dir}'")
        sys.exit(1)

    print(f"Found {len(log_files)} log files:")
    for lf in log_files:
        print(f"  - {lf}")

    # Parse each log file and collect dataframes
    dataframes = {}
    for log_file in log_files:
        sheet_name = get_sheet_name_from_filename(log_file)
        rows = parse_baseline_log(log_file)

        if rows:
            df = pd.DataFrame(rows)
            dataframes[sheet_name] = df
            print(f"Parsed {len(rows)} rows from {sheet_name}")
        else:
            print(f"Warning: No data found in {log_file}")

    if not dataframes:
        print("No data found in any log files")
        sys.exit(1)

    # Compute summary statistics
    summary_df = compute_summary_stats(dataframes)

    # Write to Excel with summary sheet first, then data sheets
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        for sheet_name, df in dataframes.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"\nWrote {len(dataframes) + 1} sheets to {output_file}")
    print(f"  - Summary sheet with {len(summary_df)} rows")


if __name__ == '__main__':
    main()
