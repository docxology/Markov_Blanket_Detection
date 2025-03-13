#!/usr/bin/env python3
"""
DMBD HTML Report Generator
=========================

Generates an HTML report from test results and visualizations 
to improve accessibility and interpretability of the DMBD test suite.
"""

import os
import glob
import json
import time
import pandas as pd
from pathlib import Path
import pickle
import base64

def generate_html_report():
    """Generate a comprehensive HTML report from test outputs."""
    # Set paths
    output_dir = Path('output')
    figures_dir = output_dir / 'figures'
    data_dir = output_dir / 'data'
    analysis_dir = output_dir / 'analysis'
    report_path = output_dir / 'dmbd_report.html'
    
    # Check if output directory exists
    if not output_dir.exists():
        print("Output directory not found. Run tests first.")
        return
    
    # Read test summary if available
    summary_text = "Test summary not available."
    if (output_dir / 'test_summary.txt').exists():
        with open(output_dir / 'test_summary.txt', 'r') as f:
            summary_text = f.read()
    
    # Get list of figures
    figures = sorted(glob.glob(str(figures_dir / '*.png')))
    
    # Get list of datasets
    datasets = sorted(glob.glob(str(data_dir / '*.csv')))
    
    # Get list of analysis results
    analyses = sorted(glob.glob(str(analysis_dir / '*.pkl')))
    
    # Read log file if available
    log_content = "Log file not available."
    if (output_dir / 'dmbd_tests.log').exists():
        with open(output_dir / 'dmbd_tests.log', 'r') as f:
            log_content = f.read()
    
    # Start building HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>DMBD Test Results Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                color: #333;
            }}
            h1, h2, h3, h4 {{
                color: #2c3e50;
            }}
            .container {{
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                margin-bottom: 30px;
            }}
            .figure-container {{
                flex: 1 1 calc(33% - 20px);
                min-width: 300px;
                margin-bottom: 20px;
                box-shadow: 0 0 5px rgba(0,0,0,0.1);
                border-radius: 5px;
                overflow: hidden;
            }}
            .figure-container img {{
                width: 100%;
                height: auto;
                display: block;
            }}
            .figure-caption {{
                padding: 10px;
                background: #f8f9fa;
                font-size: 0.9rem;
            }}
            pre {{
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
                font-size: 0.9rem;
                max-height: 400px;
                overflow-y: auto;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .tabs {{
                display: flex;
                flex-wrap: wrap;
                margin-bottom: 20px;
            }}
            .tab-button {{
                padding: 10px 20px;
                background: #f8f9fa;
                border: none;
                cursor: pointer;
                margin-right: 5px;
                border-radius: 5px 5px 0 0;
            }}
            .tab-button.active {{
                background: #007bff;
                color: white;
            }}
            .tab-content {{
                display: none;
                padding: 20px;
                border: 1px solid #ddd;
                border-radius: 0 5px 5px 5px;
            }}
            .tab-content.active {{
                display: block;
            }}
        </style>
    </head>
    <body>
        <h1>DMBD Comprehensive Test Report</h1>
        <p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="tabs">
            <button class="tab-button active" onclick="openTab(event, 'summary')">Summary</button>
            <button class="tab-button" onclick="openTab(event, 'markov')">Markov Blanket</button>
            <button class="tab-button" onclick="openTab(event, 'dynamic')">Dynamic Markov Blanket</button>
            <button class="tab-button" onclick="openTab(event, 'cognitive')">Cognitive Structures</button>
            <button class="tab-button" onclick="openTab(event, 'dataset')">Datasets</button>
            <button class="tab-button" onclick="openTab(event, 'log')">Logs</button>
        </div>
        
        <div id="summary" class="tab-content active">
            <h2>Test Summary</h2>
            <pre>{summary_text}</pre>
            
            <h3>Test Overview</h3>
            <ul>
                <li>Total figures generated: {len(figures)}</li>
                <li>Synthetic datasets created: {len(datasets)}</li>
                <li>Analysis results saved: {len(analyses)}</li>
            </ul>
        </div>
        
        <div id="markov" class="tab-content">
            <h2>Static Markov Blanket Visualizations</h2>
            <div class="container">
    """
    
    # Add static Markov blanket figures
    mb_figures = [f for f in figures if '_blanket.png' in f or '_mb.png' in f]
    for fig_path in mb_figures:
        fig_name = os.path.basename(fig_path)
        html_content += f"""
                <div class="figure-container">
                    <img src="{fig_path}" alt="{fig_name}">
                    <div class="figure-caption">{fig_name}</div>
                </div>
        """
    
    html_content += """
            </div>
            
            <h2>Information Flow Visualizations</h2>
            <div class="container">
    """
    
    # Add information flow figures
    info_figures = [f for f in figures if '_info_flow.png' in f]
    for fig_path in info_figures:
        fig_name = os.path.basename(fig_path)
        html_content += f"""
                <div class="figure-container">
                    <img src="{fig_path}" alt="{fig_name}">
                    <div class="figure-caption">{fig_name}</div>
                </div>
        """
    
    html_content += """
            </div>
        </div>
        
        <div id="dynamic" class="tab-content">
            <h2>Dynamic Markov Blanket Visualizations</h2>
            <div class="container">
    """
    
    # Add dynamic Markov blanket figures
    dynamic_figures = [f for f in figures if '_dynamics.png' in f or '_temporal.png' in f]
    for fig_path in dynamic_figures:
        fig_name = os.path.basename(fig_path)
        html_content += f"""
                <div class="figure-container">
                    <img src="{fig_path}" alt="{fig_name}">
                    <div class="figure-caption">{fig_name}</div>
                </div>
        """
    
    html_content += """
            </div>
        </div>
        
        <div id="cognitive" class="tab-content">
            <h2>Cognitive Structure Visualizations</h2>
            <div class="container">
    """
    
    # Add cognitive structure figures
    cognitive_figures = [f for f in figures if '_cognitive.png' in f]
    for fig_path in cognitive_figures:
        fig_name = os.path.basename(fig_path)
        html_content += f"""
                <div class="figure-container">
                    <img src="{fig_path}" alt="{fig_name}">
                    <div class="figure-caption">{fig_name}</div>
                </div>
        """
    
    html_content += """
            </div>
            
            <h2>Example Visualizations</h2>
            <div class="container">
    """
    
    # Add example figures
    example_figures = [f for f in figures if 'example_figure' in f]
    for fig_path in example_figures:
        fig_name = os.path.basename(fig_path)
        html_content += f"""
                <div class="figure-container">
                    <img src="{fig_path}" alt="{fig_name}">
                    <div class="figure-caption">{fig_name}</div>
                </div>
        """
    
    html_content += """
            </div>
        </div>
        
        <div id="dataset" class="tab-content">
            <h2>Synthetic Datasets</h2>
    """
    
    # Add dataset previews
    for dataset_path in datasets:
        dataset_name = os.path.basename(dataset_path)
        try:
            df = pd.read_csv(dataset_path)
            html_content += f"""
            <h3>{dataset_name}</h3>
            <p>Shape: {df.shape[0]} rows × {df.shape[1]} columns</p>
            {df.head(5).to_html(index=False)}
            """
        except Exception as e:
            html_content += f"""
            <h3>{dataset_name}</h3>
            <p>Error loading dataset: {str(e)}</p>
            """
    
    html_content += """
        </div>
        
        <div id="log" class="tab-content">
            <h2>Test Logs</h2>
            <pre>{}</pre>
        </div>
        
        <script>
            function openTab(evt, tabName) {{
                var i, tabcontent, tabbuttons;
                
                // Hide all tab content
                tabcontent = document.getElementsByClassName("tab-content");
                for (i = 0; i < tabcontent.length; i++) {{
                    tabcontent[i].className = tabcontent[i].className.replace(" active", "");
                }}
                
                // Remove active class from tab buttons
                tabbuttons = document.getElementsByClassName("tab-button");
                for (i = 0; i < tabbuttons.length; i++) {{
                    tabbuttons[i].className = tabbuttons[i].className.replace(" active", "");
                }}
                
                // Show the selected tab content and add active class to the button
                document.getElementById(tabName).className += " active";
                evt.currentTarget.className += " active";
            }}
        </script>
    </body>
    </html>
    """.format(log_content)
    
    # Write HTML to file
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    print(f"HTML report generated at {report_path}")

if __name__ == "__main__":
    generate_html_report() 