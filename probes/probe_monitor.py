#!/usr/bin/env python3
"""
Probe monitoring tool for Tenacious Agent.

Tracks trigger rates over time, visualizes trends, and identifies regressions.
Run after each evaluation cycle to log probe results and generate reports.

Usage:
    # Log a probe run
    python probes/probe_monitor.py log --run-id "2026-04-25-baseline" --results probe_results.json
    
    # Generate trend report
    python probes/probe_monitor.py report --output probes/trigger_trends.html
    
    # Check for regressions
    python probes/probe_monitor.py check --threshold 0.15
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _history_path() -> Path:
    """Path to probe history log (JSONL format)."""
    return Path(__file__).parent / "probe_history.jsonl"


def _library_path() -> Path:
    """Path to probe library markdown."""
    return Path(__file__).parent / "probe_library.md"


# ---------------------------------------------------------------------------
# Probe result logging
# ---------------------------------------------------------------------------

def log_probe_run(run_id: str, results: dict[str, Any]) -> None:
    """
    Append a probe run to the history log.
    
    Args:
        run_id: Unique identifier for this run (e.g., "2026-04-25-baseline")
        results: Dict mapping probe_id -> {"triggered": bool, "cost": float, "notes": str}
    
    Example results format:
        {
            "P-001": {"triggered": True, "cost": 847, "notes": "Still over-claiming hiring signal"},
            "P-002": {"triggered": False, "cost": 0, "notes": "Mixed signal logic fixed"},
            ...
        }
    """
    history_path = _history_path()
    
    entry = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "total_triggered": sum(1 for r in results.values() if r.get("triggered")),
        "total_cost": sum(r.get("cost", 0) for r in results.values() if r.get("triggered")),
        "probe_count": len(results),
    }
    
    with open(history_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    
    print(f"✓ Logged {len(results)} probe results to {history_path}")
    print(f"  Run ID: {run_id}")
    print(f"  Triggered: {entry['total_triggered']}/{entry['probe_count']}")
    print(f"  Total cost: ${entry['total_cost']:,}")


# ---------------------------------------------------------------------------
# Trend analysis
# ---------------------------------------------------------------------------

def load_probe_history() -> list[dict[str, Any]]:
    """Load all probe runs from history log."""
    history_path = _history_path()
    
    if not history_path.exists():
        return []
    
    runs = []
    with open(history_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                runs.append(json.loads(line))
    
    return runs


def generate_trend_report(output_path: str) -> None:
    """
    Generate HTML report showing probe trigger trends over time.
    
    Args:
        output_path: Where to write the HTML report
    """
    runs = load_probe_history()
    
    if not runs:
        print("⚠ No probe history found. Run 'log' command first.")
        sys.exit(1)
    
    # Extract probe IDs from library
    probe_ids = _extract_probe_ids_from_library()
    
    # Build time series for each probe
    probe_trends: dict[str, list[tuple[str, bool]]] = {pid: [] for pid in probe_ids}
    
    for run in runs:
        run_id = run["run_id"]
        for probe_id, result in run.get("results", {}).items():
            if probe_id in probe_trends:
                probe_trends[probe_id].append((run_id, result.get("triggered", False)))
    
    # Generate HTML
    html = _build_html_report(runs, probe_trends)
    
    output = Path(output_path)
    output.write_text(html, encoding="utf-8")
    
    print(f"✓ Generated trend report: {output}")
    print(f"  Runs analyzed: {len(runs)}")
    print(f"  Probes tracked: {len(probe_ids)}")


def _extract_probe_ids_from_library() -> list[str]:
    """Parse probe library markdown to extract probe IDs."""
    library_path = _library_path()
    
    if not library_path.exists():
        return []
    
    probe_ids = []
    content = library_path.read_text(encoding="utf-8")
    
    for line in content.split("\n"):
        if line.startswith("## P-"):
            probe_id = line.strip("## ").strip()
            probe_ids.append(probe_id)
    
    return probe_ids


def _build_html_report(runs: list[dict], probe_trends: dict[str, list[tuple[str, bool]]]) -> str:
    """Build HTML report with trend visualization."""
    
    # Calculate summary stats
    latest_run = runs[-1] if runs else {}
    total_probes = len(probe_trends)
    triggered_count = sum(1 for trends in probe_trends.values() if trends and trends[-1][1])
    
    # Build probe rows
    probe_rows = []
    for probe_id, trends in sorted(probe_trends.items()):
        if not trends:
            continue
        
        # Calculate trend direction
        recent_triggers = [t for _, t in trends[-3:]]
        trend_icon = "📈" if sum(recent_triggers) > len(recent_triggers) / 2 else "📉"
        
        # Build sparkline
        sparkline = "".join("█" if triggered else "░" for _, triggered in trends[-10:])
        
        # Current status
        current_status = "🔴 TRIGGERED" if trends[-1][1] else "🟢 PASSED"
        
        probe_rows.append(f"""
        <tr>
            <td><strong>{probe_id}</strong></td>
            <td>{current_status}</td>
            <td><code>{sparkline}</code></td>
            <td>{trend_icon}</td>
            <td>{sum(1 for _, t in trends if t)}/{len(trends)}</td>
        </tr>
        """)
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Probe Trigger Trends — Tenacious Agent</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                max-width: 1200px;
                margin: 40px auto;
                padding: 0 20px;
                background: #f5f5f5;
            }}
            h1 {{
                color: #333;
                border-bottom: 3px solid #007bff;
                padding-bottom: 10px;
            }}
            .summary {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                margin: 20px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .summary-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-top: 15px;
            }}
            .stat {{
                text-align: center;
            }}
            .stat-value {{
                font-size: 2em;
                font-weight: bold;
                color: #007bff;
            }}
            .stat-label {{
                color: #666;
                font-size: 0.9em;
                margin-top: 5px;
            }}
            table {{
                width: 100%;
                background: white;
                border-collapse: collapse;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            th {{
                background: #007bff;
                color: white;
                padding: 12px;
                text-align: left;
                font-weight: 600;
            }}
            td {{
                padding: 12px;
                border-bottom: 1px solid #eee;
            }}
            tr:hover {{
                background: #f8f9fa;
            }}
            code {{
                font-family: "SF Mono", Monaco, monospace;
                font-size: 0.9em;
                letter-spacing: 1px;
            }}
            .footer {{
                text-align: center;
                color: #666;
                margin-top: 40px;
                padding: 20px;
                font-size: 0.9em;
            }}
        </style>
    </head>
    <body>
        <h1>🔬 Probe Trigger Trends</h1>
        
        <div class="summary">
            <h2>Latest Run: {latest_run.get('run_id', 'N/A')}</h2>
            <div class="summary-grid">
                <div class="stat">
                    <div class="stat-value">{triggered_count}/{total_probes}</div>
                    <div class="stat-label">Probes Triggered</div>
                </div>
                <div class="stat">
                    <div class="stat-value">${latest_run.get('total_cost', 0):,}</div>
                    <div class="stat-label">Total Business Cost</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{len(runs)}</div>
                    <div class="stat-label">Historical Runs</div>
                </div>
            </div>
        </div>
        
        <table>
            <thead>
                <tr>
                    <th>Probe ID</th>
                    <th>Current Status</th>
                    <th>Last 10 Runs</th>
                    <th>Trend</th>
                    <th>Trigger Rate</th>
                </tr>
            </thead>
            <tbody>
                {''.join(probe_rows)}
            </tbody>
        </table>
        
        <div class="footer">
            <p>Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</p>
            <p>Legend: █ = triggered, ░ = passed | 📈 = worsening, 📉 = improving</p>
        </div>
    </body>
    </html>
    """
    
    return html


# ---------------------------------------------------------------------------
# Regression detection
# ---------------------------------------------------------------------------

def check_for_regressions(threshold: float = 0.15) -> None:
    """
    Check if any probes have regressed (trigger rate increased significantly).
    
    Args:
        threshold: Minimum increase in trigger rate to flag as regression (default 0.15 = 15%)
    
    Exit code:
        0 if no regressions detected
        1 if regressions found
    """
    runs = load_probe_history()
    
    if len(runs) < 2:
        print("⚠ Need at least 2 runs to detect regressions")
        sys.exit(0)
    
    # Compare last 2 runs
    previous_run = runs[-2]
    current_run = runs[-1]
    
    regressions = []
    
    for probe_id in current_run.get("results", {}).keys():
        prev_triggered = previous_run.get("results", {}).get(probe_id, {}).get("triggered", False)
        curr_triggered = current_run.get("results", {}).get(probe_id, {}).get("triggered", False)
        
        # Regression: probe passed before, triggers now
        if not prev_triggered and curr_triggered:
            cost = current_run["results"][probe_id].get("cost", 0)
            notes = current_run["results"][probe_id].get("notes", "")
            regressions.append({
                "probe_id": probe_id,
                "cost": cost,
                "notes": notes,
            })
    
    if regressions:
        print(f"❌ REGRESSIONS DETECTED: {len(regressions)} probe(s) now triggering")
        print()
        for reg in regressions:
            print(f"  {reg['probe_id']}: ${reg['cost']:,} — {reg['notes']}")
        print()
        print(f"Previous run: {previous_run['run_id']}")
        print(f"Current run:  {current_run['run_id']}")
        sys.exit(1)
    else:
        print(f"✓ No regressions detected")
        print(f"  Previous: {previous_run['total_triggered']}/{previous_run['probe_count']} triggered")
        print(f"  Current:  {current_run['total_triggered']}/{current_run['probe_count']} triggered")
        sys.exit(0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe monitoring tool for Tenacious Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Log command
    log_parser = subparsers.add_parser("log", help="Log a probe run")
    log_parser.add_argument("--run-id", required=True, help="Unique run identifier")
    log_parser.add_argument("--results", required=True, help="Path to probe results JSON")
    
    # Report command
    report_parser = subparsers.add_parser("report", help="Generate trend report")
    report_parser.add_argument("--output", default="probes/trigger_trends.html", help="Output HTML path")
    
    # Check command
    check_parser = subparsers.add_parser("check", help="Check for regressions")
    check_parser.add_argument("--threshold", type=float, default=0.15, help="Regression threshold")
    
    args = parser.parse_args()
    
    if args.command == "log":
        with open(args.results, "r", encoding="utf-8") as fh:
            results = json.load(fh)
        log_probe_run(args.run_id, results)
    
    elif args.command == "report":
        generate_trend_report(args.output)
    
    elif args.command == "check":
        check_for_regressions(args.threshold)


if __name__ == "__main__":
    main()
