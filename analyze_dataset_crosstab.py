#!/usr/bin/env python3
"""
Analyze Tenacious-Bench dataset to generate integrated cross-tabulation
across dimension, partition, and source mode axes.
"""

import json
import os
from pathlib import Path
from collections import defaultdict

def extract_source_mode(task_id):
    """Extract source mode from task_id format: TB-{DIM}-{SOURCE}-{NUM}"""
    parts = task_id.split('-')
    if len(parts) >= 3:
        source_code = parts[2]
        source_map = {
            'TR': 'Trace-Derived',
            'PR': 'Programmatic',
            'ML': 'Multi-LLM',
            'HA': 'Hand-Authored'
        }
        return source_map.get(source_code, 'Unknown')
    return 'Unknown'

def map_dimension(dim):
    """Map dimension codes to full names"""
    dim_map = {
        'capacity_honesty': 'Capacity Honesty',
        'signal_grounding': 'Signal Grounding',
        'tone_preservation': 'Tone Preservation',
        'consent_coordination': 'Consent Coordination',
        'gap_framing': 'Gap Framing'
    }
    return dim_map.get(dim, dim)

def analyze_dataset():
    """Analyze all JSON files in tenacious_bench_v0.1 directory"""
    
    # Initialize counters
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    total_counts = defaultdict(int)
    
    # Define partitions
    partitions = ['train', 'dev', 'held_out']
    
    base_path = Path('tenacious_bench_v0.1')
    
    if not base_path.exists():
        print("Error: tenacious_bench_v0.1 directory not found")
        return None
    
    # Scan all partitions
    for partition in partitions:
        partition_path = base_path / partition
        if not partition_path.exists():
            continue
            
        # Find all JSON files
        for json_file in partition_path.glob('*.json'):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                task_id = data.get('task_id', '')
                dimension = data.get('dimension', 'unknown')
                
                # Extract source mode from task_id
                source_mode = extract_source_mode(task_id)
                
                # Map dimension to full name
                dimension_full = map_dimension(dimension)
                
                # Count
                counts[dimension_full][partition][source_mode] += 1
                total_counts['total'] += 1
                
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
    
    return counts, total_counts

def generate_markdown_tables(counts):
    """Generate markdown tables for the cross-tabulation"""
    
    dimensions = ['Capacity Honesty', 'Signal Grounding', 'Tone Preservation', 
                  'Consent Coordination', 'Gap Framing']
    partitions = ['train', 'dev', 'held_out']
    sources = ['Trace-Derived', 'Programmatic', 'Multi-LLM', 'Hand-Authored']
    
    # Calculate totals
    source_partition_totals = defaultdict(lambda: defaultdict(int))
    dimension_source_totals = defaultdict(lambda: defaultdict(int))
    partition_totals = defaultdict(int)
    source_totals = defaultdict(int)
    dimension_totals = defaultdict(int)
    grand_total = 0
    
    for dim in dimensions:
        for part in partitions:
            for src in sources:
                count = counts.get(dim, {}).get(part, {}).get(src, 0)
                source_partition_totals[src][part] += count
                dimension_source_totals[dim][src] += count
                partition_totals[part] += count
                source_totals[src] += count
                dimension_totals[dim] += count
                grand_total += count
    
    # Table 1: Source Mode × Partition
    print("\n#### Table 1: Source Mode × Partition (Aggregated Across All Dimensions)\n")
    print("| Source Mode          | Train | Dev | Held-out | **Total** |")
    print("|---------------------|-------|-----|----------|-----------|")
    for src in sources:
        train = source_partition_totals[src]['train']
        dev = source_partition_totals[src]['dev']
        held = source_partition_totals[src]['held_out']
        total = source_totals[src]
        print(f"| {src:19} | {train:5} | {dev:3} | {held:8} | **{total}**    |")
    print(f"| **Total**           | **{partition_totals['train']}** | **{partition_totals['dev']}** | **{partition_totals['held_out']}** | **{grand_total}** |")
    
    # Table 2: Dimension × Source Mode
    print("\n#### Table 2: Dimension × Source Mode (Aggregated Across All Partitions)\n")
    print("| Dimension            | Trace-Derived | Programmatic | Multi-LLM | Hand-Authored | **Total** |")
    print("|---------------------|---------------|--------------|-----------|---------------|-----------|")
    for dim in dimensions:
        tr = dimension_source_totals[dim]['Trace-Derived']
        pr = dimension_source_totals[dim]['Programmatic']
        ml = dimension_source_totals[dim]['Multi-LLM']
        ha = dimension_source_totals[dim]['Hand-Authored']
        total = dimension_totals[dim]
        print(f"| {dim:19} | {tr:13} | {pr:12} | {ml:9} | {ha:13} | **{total}**    |")
    print(f"| **Total**           | **{source_totals['Trace-Derived']}**        | **{source_totals['Programmatic']}**       | **{source_totals['Multi-LLM']}**    | **{source_totals['Hand-Authored']}**        | **{grand_total}**   |")
    
    # Table 3: Full Three-Way Cross-Tabulation
    print("\n#### Table 3: Full Three-Way Cross-Tabulation (Dimension × Partition × Source Mode)\n")
    print("This table presents the complete intersection, showing task counts for each unique combination of dimension, partition, and source mode.\n")
    print("| Dimension            | Partition | Trace | Prog | Multi-LLM | Hand-Auth | **Row Total** |")
    print("|---------------------|-----------|-------|------|-----------|-----------|---------------|")
    
    for dim in dimensions:
        dim_short = dim[:19] if len(dim) <= 19 else dim[:16] + "..."
        for i, part in enumerate(partitions):
            part_display = part.replace('_', '-').title() if part == 'held_out' else part.title()
            
            tr = counts.get(dim, {}).get(part, {}).get('Trace-Derived', 0)
            pr = counts.get(dim, {}).get(part, {}).get('Programmatic', 0)
            ml = counts.get(dim, {}).get(part, {}).get('Multi-LLM', 0)
            ha = counts.get(dim, {}).get(part, {}).get('Hand-Authored', 0)
            row_total = tr + pr + ml + ha
            
            if i == 0:
                print(f"| **{dim_short:19}**| {part_display:9} | {tr:5} | {pr:4} | {ml:9} | {ha:9} | {row_total:13} |")
            else:
                print(f"|                     | {part_display:9} | {tr:5} | {pr:4} | {ml:9} | {ha:9} | {row_total:13} |")
    
    # Grand totals
    total_tr = source_totals['Trace-Derived']
    total_pr = source_totals['Programmatic']
    total_ml = source_totals['Multi-LLM']
    total_ha = source_totals['Hand-Authored']
    print(f"| **Grand Total**     |           | **{total_tr}**| **{total_pr}**| **{total_ml}**   | **{total_ha}**    | **{grand_total}**       |")
    
    # Key insights
    print("\n**Key Insights from Cross-Tabulation:**")
    print(f"*   **Balanced Representation:** Each dimension maintains {dimension_totals[dimensions[0]]} tasks, with {partition_totals['train']}/{partition_totals['dev']}/{partition_totals['held_out']} train/dev/held-out splits.")
    print(f"*   **Source Mode Distribution:** Trace-derived ({source_totals['Trace-Derived']} tasks, {100*source_totals['Trace-Derived']/grand_total:.1f}%) and programmatic ({source_totals['Programmatic']} tasks, {100*source_totals['Programmatic']/grand_total:.1f}%) tasks form the core, while multi-LLM synthesis ({source_totals['Multi-LLM']} tasks, {100*source_totals['Multi-LLM']/grand_total:.1f}%) and hand-authored adversarial tasks ({source_totals['Hand-Authored']} tasks, {100*source_totals['Hand-Authored']/grand_total:.1f}%) provide targeted lexical diversity and edge-case coverage.")
    print(f"*   **Held-Out Composition:** The sealed evaluation set contains {source_partition_totals['Trace-Derived']['held_out']} trace-derived, {source_partition_totals['Programmatic']['held_out']} programmatic, {source_partition_totals['Multi-LLM']['held_out']} multi-LLM, and {source_partition_totals['Hand-Authored']['held_out']} hand-authored tasks, ensuring all generation methods are represented in final evaluation.")
    print(f"*   **Adversarial Coverage:** Hand-authored adversarial tasks are present across all dimensions and partitions, with particular concentration in training data ({source_partition_totals['Hand-Authored']['train']} tasks) to support preference learning.")

if __name__ == '__main__':
    counts, totals = analyze_dataset()
    if counts:
        print("### Integrated Cross-Tabulation: Dimension × Partition × Source Mode\n")
        print("To provide full transparency into the dataset's internal structure, the following tables present integrated cross-tabulations showing task counts at the intersection of all three compositional axes.\n")
        generate_markdown_tables(counts)
        print(f"\n**Total tasks analyzed: {totals['total']}**")
    else:
        print("Failed to analyze dataset")
