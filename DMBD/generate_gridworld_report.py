#!/usr/bin/env python3
"""
Gridworld DMBD Analysis Report Generator
=======================================

This script generates an HTML report summarizing the results of a
gridworld Dynamic Markov Blanket analysis, including visualizations,
metrics, and animations.
"""

import os
import sys
import argparse
import time
import json
import glob
from pathlib import Path
from typing import Dict, List, Any, Union, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate HTML report for gridworld DMBD analysis'
    )
    
    parser.add_argument(
        '--analysis-dir', 
        type=str, 
        required=True,
        help='Directory containing the analysis results'
    )
    
    parser.add_argument(
        '--output-file', 
        type=str, 
        default=None,
        help='Path to output HTML file (default: <analysis_dir>/report.html)'
    )
    
    parser.add_argument(
        '--title', 
        type=str, 
        default='Gridworld Dynamic Markov Blanket Analysis',
        help='Title for the report'
    )
    
    parser.add_argument(
        '--include-raw-data', 
        action='store_true',
        help='Include raw data in the report (default: False)'
    )
    
    return parser.parse_args()


def read_summary(analysis_dir: str) -> Dict[str, Any]:
    """
    Read the summary file from the analysis directory.
    
    Args:
        analysis_dir: Path to the analysis directory
        
    Returns:
        Dictionary with summary information
    """
    summary_path = os.path.join(analysis_dir, 'summary.txt')
    
    if not os.path.exists(summary_path):
        return {}
    
    summary = {}
    with open(summary_path, 'r') as f:
        lines = f.readlines()
        
    # Parse key-value pairs
    for line in lines:
        line = line.strip()
        if ': ' in line:
            key, value = line.split(': ', 1)
            key = key.strip().strip('  ')
            value = value.strip()
            summary[key] = value
            
    return summary


def find_outputs(analysis_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Find all output files from the analysis.
    
    Args:
        analysis_dir: Path to the analysis directory
        
    Returns:
        Dictionary with categories of output files
    """
    outputs = {
        'simulation': {},
        'analysis': {},
        'figures': {}
    }
    
    # Simulation outputs
    simulation_dir = os.path.join(analysis_dir, 'simulation')
    if os.path.exists(simulation_dir):
        for file in glob.glob(os.path.join(simulation_dir, '*')):
            file_name = os.path.basename(file)
            if file_name.endswith('.mp4'):
                outputs['simulation']['animation'] = file
            elif file_name.endswith('.png'):
                outputs['simulation']['frame'] = file
            elif file_name.endswith('.csv'):
                outputs['simulation']['data'] = file
    
    # Analysis outputs
    analysis_subdir = os.path.join(analysis_dir, 'analysis')
    if os.path.exists(analysis_subdir):
        for file in glob.glob(os.path.join(analysis_subdir, '*')):
            file_name = os.path.basename(file)
            if file_name.endswith('.mp4'):
                if 'markov_blanket' in file_name:
                    outputs['analysis']['markov_blanket_animation'] = file
                elif 'cognitive_structure' in file_name:
                    outputs['analysis']['cognitive_structure_animation'] = file
            elif file_name.endswith('.png'):
                if 'markov_blanket' in file_name:
                    outputs['analysis']['markov_blanket_partition'] = file
                elif 'cognitive' in file_name:
                    outputs['analysis']['cognitive_partition'] = file
    
    # Look for more figures in figures directory
    figures_dir = os.path.join(analysis_dir, 'figures')
    if os.path.exists(figures_dir):
        for file in glob.glob(os.path.join(figures_dir, '*.png')):
            file_name = os.path.basename(file)
            outputs['figures'][file_name] = file
    
    return outputs


def generate_html_report(
    analysis_dir: str,
    output_file: Optional[str] = None,
    title: str = 'Gridworld Dynamic Markov Blanket Analysis',
    include_raw_data: bool = False
) -> str:
    """
    Generate an HTML report summarizing the analysis results.
    
    Args:
        analysis_dir: Path to the analysis directory
        output_file: Path to output HTML file
        title: Title for the report
        include_raw_data: Whether to include raw data in the report
        
    Returns:
        Path to the generated HTML file
    """
    if output_file is None:
        output_file = os.path.join(analysis_dir, 'report.html')
    
    # Read summary information
    summary = read_summary(analysis_dir)
    
    # Find output files
    outputs = find_outputs(analysis_dir)
    
    # Start building the HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
            background-color: #f8f9fa;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: #fff;
            padding: 20px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #3498db;
            margin-top: 30px;
        }}
        h3 {{
            color: #2980b9;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 0.9em;
            margin-bottom: 20px;
        }}
        .summary-table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}
        .summary-table th, .summary-table td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        .summary-table th {{
            background-color: #f2f2f2;
        }}
        .image-container {{
            margin: 20px 0;
            text-align: center;
        }}
        .image-container img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
        }}
        .video-container {{
            margin: 20px 0;
            text-align: center;
        }}
        .video-container video {{
            max-width: 100%;
            border: 1px solid #ddd;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .flex-container {{
            display: flex;
            flex-wrap: wrap;
            justify-content: space-around;
        }}
        .flex-item {{
            flex: 1 1 45%;
            margin: 10px;
            min-width: 300px;
            max-width: 100%;
        }}
        .code {{
            font-family: monospace;
            background-color: #f4f4f4;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ecf0f1;
            font-size: 0.9em;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div class="timestamp">Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}</div>
        
        <div class="section">
            <h2>Analysis Summary</h2>
            <table class="summary-table">
                <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                </tr>
"""
    
    # Add summary information to the table
    for key, value in summary.items():
        if key and value and not key.startswith('Generated'):
            html += f"""                <tr>
                    <td>{key}</td>
                    <td>{value}</td>
                </tr>
"""
    
    html += """            </table>
        </div>
        
        <div class="section">
            <h2>Gridworld Simulation</h2>
            <div class="flex-container">
"""
    
    # Add simulation frame
    if 'frame' in outputs['simulation']:
        frame_path = outputs['simulation']['frame']
        relative_path = os.path.relpath(frame_path, os.path.dirname(output_file))
        html += f"""                <div class="flex-item">
                    <h3>Example Frame</h3>
                    <div class="image-container">
                        <img src="{relative_path}" alt="Example Frame">
                    </div>
                </div>
"""
    
    # Add simulation animation
    if 'animation' in outputs['simulation']:
        animation_path = outputs['simulation']['animation']
        relative_path = os.path.relpath(animation_path, os.path.dirname(output_file))
        html += f"""                <div class="flex-item">
                    <h3>Simulation Animation</h3>
                    <div class="video-container">
                        <video controls loop>
                            <source src="{relative_path}" type="video/mp4">
                            Your browser does not support the video tag.
                        </video>
                    </div>
                </div>
"""
    
    html += """            </div>
        </div>
        
        <div class="section">
            <h2>Dynamic Markov Blanket Analysis</h2>
            <div class="flex-container">
"""
    
    # Add Markov blanket partition
    if 'markov_blanket_partition' in outputs['analysis']:
        partition_path = outputs['analysis']['markov_blanket_partition']
        relative_path = os.path.relpath(partition_path, os.path.dirname(output_file))
        html += f"""                <div class="flex-item">
                    <h3>Markov Blanket Partition</h3>
                    <div class="image-container">
                        <img src="{relative_path}" alt="Markov Blanket Partition">
                    </div>
                    <p>
                        The Markov Blanket partition shows the target cell (red) and its Markov Blanket components:
                        parents (green), children (blue), and spouses (yellow).
                    </p>
                </div>
"""
    
    # Add cognitive partition
    if 'cognitive_partition' in outputs['analysis']:
        partition_path = outputs['analysis']['cognitive_partition']
        relative_path = os.path.relpath(partition_path, os.path.dirname(output_file))
        html += f"""                <div class="flex-item">
                    <h3>Cognitive Partition</h3>
                    <div class="image-container">
                        <img src="{relative_path}" alt="Cognitive Partition">
                    </div>
                    <p>
                        The Cognitive partition shows the target cell (red) and its cognitive structure components:
                        sensory nodes (green), action nodes (blue), and internal state nodes (purple).
                    </p>
                </div>
"""
    
    html += """            </div>
            
            <h3>Dynamic Evolution</h3>
            <div class="flex-container">
"""
    
    # Add Markov blanket animation
    if 'markov_blanket_animation' in outputs['analysis']:
        animation_path = outputs['analysis']['markov_blanket_animation']
        relative_path = os.path.relpath(animation_path, os.path.dirname(output_file))
        html += f"""                <div class="flex-item">
                    <h3>Markov Blanket Evolution</h3>
                    <div class="video-container">
                        <video controls loop>
                            <source src="{relative_path}" type="video/mp4">
                            Your browser does not support the video tag.
                        </video>
                    </div>
                    <p>
                        This animation shows how the Markov Blanket components evolve over time as the
                        Gaussian blur moves through the gridworld.
                    </p>
                </div>
"""
    
    # Add cognitive structure animation
    if 'cognitive_structure_animation' in outputs['analysis']:
        animation_path = outputs['analysis']['cognitive_structure_animation']
        relative_path = os.path.relpath(animation_path, os.path.dirname(output_file))
        html += f"""                <div class="flex-item">
                    <h3>Cognitive Structure Evolution</h3>
                    <div class="video-container">
                        <video controls loop>
                            <source src="{relative_path}" type="video/mp4">
                            Your browser does not support the video tag.
                        </video>
                    </div>
                    <p>
                        This animation shows how the cognitive structure components evolve over time as the
                        Gaussian blur moves through the gridworld.
                    </p>
                </div>
"""
    
    html += """            </div>
        </div>
"""
    
    # Add additional figures if any
    if outputs['figures']:
        html += """        <div class="section">
            <h2>Additional Visualizations</h2>
            <div class="flex-container">
"""
        
        for name, path in outputs['figures'].items():
            relative_path = os.path.relpath(path, os.path.dirname(output_file))
            html += f"""                <div class="flex-item">
                    <h3>{name.replace('_', ' ').replace('.png', '').title()}</h3>
                    <div class="image-container">
                        <img src="{relative_path}" alt="{name}">
                    </div>
                </div>
"""
        
        html += """            </div>
        </div>
"""
    
    # Add raw data if requested
    if include_raw_data and 'data' in outputs['simulation']:
        data_path = outputs['simulation']['data']
        try:
            df = pd.read_csv(data_path)
            html += """        <div class="section">
            <h2>Raw Data Sample</h2>
            <div class="code">
"""
            # Add a sample of the raw data (first 10 rows)
            html += df.head(10).to_html(index=False)
            
            html += """            </div>
        </div>
"""
        except Exception as e:
            print(f"Error reading raw data: {str(e)}")
    
    # Add conclusion and footer
    html += """        <div class="section">
            <h2>Interpretation</h2>
            <p>
                The Dynamic Markov Blanket analysis of the gridworld simulation demonstrates how 
                causal relationships and information flow can be inferred from time-series data.
                The moving Gaussian blur creates a dynamic pattern of activation across the grid,
                which allows us to study how information propagates spatially and temporally.
            </p>
            <p>
                Key observations from this analysis:
            </p>
            <ul>
                <li>The Markov Blanket of a cell typically includes its spatial neighbors, reflecting
                    the local nature of information flow in the gridworld.</li>
                <li>As the Gaussian blur moves, the Markov Blankets of cells adapt to track this
                    movement, showing how causal relationships change over time.</li>
                <li>The cognitive partition identifies sensory nodes (which receive external information),
                    action nodes (which influence the external environment), and internal state nodes
                    (which mediate between sensory and action nodes).</li>
                <li>The analysis reveals the self-organization of cognitive structures in the system
                    even without explicit design.</li>
            </ul>
        </div>
        
        <div class="footer">
            <p>Generated by the Dynamic Markov Blanket Detection (DMBD) Framework</p>
            <p>© OpenManus Project</p>
        </div>
    </div>
</body>
</html>
"""
    
    # Write the HTML to file
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"Generated HTML report: {output_file}")
    return output_file


def main():
    """Run the report generator."""
    args = parse_args()
    
    try:
        report_path = generate_html_report(
            analysis_dir=args.analysis_dir,
            output_file=args.output_file,
            title=args.title,
            include_raw_data=args.include_raw_data
        )
        
        print(f"Successfully generated report at: {report_path}")
        return 0
        
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 