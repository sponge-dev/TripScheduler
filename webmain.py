#!/usr/bin/env python3
"""
Web-based Location Visitation Scheduler
A Flask web application that provides a user-friendly interface for the scheduling tool.
"""

import os
import json
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_file
from werkzeug.utils import secure_filename
import subprocess
import threading
import time
from datetime import datetime
import webbrowser
from pathlib import Path

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production
app.config['UPLOAD_FOLDER'] = 'spreadsheets'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variable to track processing status
processing_status = {
    'is_processing': False,
    'progress': 0,
    'message': '',
    'output_dir': None,
    'error': None
}

def load_config():
    """Load configuration from config.json"""
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
            
        # Add default ORS config if not present
        if 'ors' not in config:
            config['ors'] = {
                "enabled": True,
                "profile": "driving-car",
                "units": "m",
                "rate_limit_delay": 0.5,
                "max_retries": 3,
                "timeout": 30,
                "use_for_validation": True,
                "show_directions": True
            }
            
        return config
    except FileNotFoundError:
        # Return default config if file doesn't exist
        return {
            "default_buffer_minutes": 15,
            "default_visit_hours": 0.75,
            "max_end_times": ["22:00", "21:00"],
            "start_times": ["12:00", "09:00"],
            "visit_dates": ["July 16, 2024", "July 17, 2024"],
            "geocoding": {
                "max_retries": 3,
                "max_wait_seconds": 300,
                "rate_limit_delay": 1
            },
            "clustering": {
                "n_clusters": 2
            },
            "ors": {
                "enabled": True,
                "profile": "driving-car",
                "units": "m",
                "rate_limit_delay": 0.5,
                "max_retries": 3,
                "timeout": 30,
                "use_for_validation": True,
                "show_directions": True
            }
        }

def save_config(config):
    """Save configuration to config.json"""
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=2)

def get_csv_files():
    """Get list of CSV files in the spreadsheets folder"""
    csv_files = []
    spreadsheets_dir = app.config['UPLOAD_FOLDER']
    
    if os.path.exists(spreadsheets_dir):
        for file in os.listdir(spreadsheets_dir):
            if file.lower().endswith('.csv'):
                csv_files.append(file)
    
    return sorted(csv_files)

def preview_csv(filename, rows=10):
    """Preview first few rows of CSV file"""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        df = pd.read_csv(filepath)
        
        # Handle NaN values by replacing them with empty string
        df = df.fillna('')
        
        # Convert to dict for JSON serialization
        preview_data = {
            'columns': list(df.columns),
            'rows': df.head(rows).to_dict('records'),
            'total_rows': len(df),
            'filename': filename
        }
        
        return preview_data
    except Exception as e:
        return {'error': str(e)}

def run_scheduler(csv_file, debug=False):
    """Run the main scheduler script in a separate process"""
    global processing_status
    
    try:
        processing_status['is_processing'] = True
        processing_status['progress'] = 10
        processing_status['message'] = 'Starting scheduler...'
        processing_status['error'] = None
        
        # Build command using the new simplified interface
        cmd = ['python', 'main.py', csv_file]
        if debug:
            cmd.append('--debug')
        
        processing_status['progress'] = 20
        processing_status['message'] = 'Running geocoding...'
        
        # Execute the main script directly
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        
        processing_status['progress'] = 80
        processing_status['message'] = 'Processing results...'
        
        if result.returncode == 0:
            processing_status['progress'] = 100
            processing_status['message'] = 'Completed successfully!'
            
            # Try to find the output directory
            output_base = 'output'
            if os.path.exists(output_base):
                # Find the most recent output directory
                subdirs = [d for d in os.listdir(output_base) if os.path.isdir(os.path.join(output_base, d))]
                if subdirs:
                    # Get the most recently modified directory
                    latest_dir = max(subdirs, key=lambda d: os.path.getmtime(os.path.join(output_base, d)))
                    processing_status['output_dir'] = os.path.join(output_base, latest_dir)
        else:
            processing_status['error'] = f"Error: {result.stderr}\nOutput: {result.stdout}"
            processing_status['message'] = 'Processing failed'
            
    except Exception as e:
        processing_status['error'] = str(e)
        processing_status['message'] = 'Processing failed'
    finally:
        processing_status['is_processing'] = False

@app.route('/')
def index():
    """Main page"""
    csv_files = get_csv_files()
    config = load_config()
    return render_template('index.html', csv_files=csv_files, config=config)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '' or file.filename is None:
        flash('No file selected')
        return redirect(request.url)
    
    if file and file.filename and file.filename.lower().endswith('.csv'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        flash(f'File {filename} uploaded successfully')
    else:
        flash('Please upload a CSV file')
    
    return redirect(url_for('index'))

@app.route('/preview/<filename>')
def preview_file(filename):
    """Preview CSV file"""
    preview_data = preview_csv(filename)
    return jsonify(preview_data)

@app.route('/config', methods=['GET', 'POST'])
def config_page():
    """Configuration page"""
    if request.method == 'POST':
        try:
            # Get form data and update config
            config = load_config()
            
            # Update basic settings
            config['default_buffer_minutes'] = int(request.form.get('default_buffer_minutes', 15))
            config['default_visit_hours'] = float(request.form.get('default_visit_hours', 0.75))
            config['clustering']['n_clusters'] = int(request.form.get('n_clusters', 2))
            
            # Update arrays (start times, end times, dates)
            n_clusters = config['clustering']['n_clusters']
            
            start_times = []
            max_end_times = []
            visit_dates = []
            
            for i in range(n_clusters):
                start_times.append(request.form.get(f'start_time_{i}', '09:00'))
                max_end_times.append(request.form.get(f'max_end_time_{i}', '22:00'))
                visit_dates.append(request.form.get(f'visit_date_{i}', f'July {16+i}, 2024'))
            
            config['start_times'] = start_times
            config['max_end_times'] = max_end_times
            config['visit_dates'] = visit_dates
            
            # Update geocoding settings
            config['geocoding']['max_retries'] = int(request.form.get('max_retries', 3))
            config['geocoding']['max_wait_seconds'] = int(request.form.get('max_wait_seconds', 300))
            config['geocoding']['rate_limit_delay'] = int(request.form.get('rate_limit_delay', 1))
            
            # Update ORS settings
            if 'ors' not in config:
                config['ors'] = {}
            
            config['ors']['enabled'] = 'ors_enabled' in request.form
            config['ors']['use_for_validation'] = 'use_for_validation' in request.form
            config['ors']['show_directions'] = 'show_directions' in request.form
            config['ors']['profile'] = request.form.get('ors_profile', 'driving-car')
            
            # Set default ORS values if not present
            config['ors'].setdefault('rate_limit_delay', 0.5)
            config['ors'].setdefault('max_retries', 3)
            config['ors'].setdefault('timeout', 30)
            config['ors'].setdefault('units', 'm')
            
            save_config(config)
            flash('Configuration saved successfully')
            
        except Exception as e:
            flash(f'Error saving configuration: {str(e)}')
    
    config = load_config()
    return render_template('config.html', config=config)

@app.route('/generate', methods=['POST'])
def generate_report():
    """Generate scheduling report"""
    global processing_status
    
    if processing_status['is_processing']:
        return jsonify({'error': 'Another process is already running'}), 400
    
    csv_file = request.form.get('csv_file')
    debug = request.form.get('debug') == 'true'
    
    if not csv_file:
        return jsonify({'error': 'No CSV file selected'}), 400
    
    # Reset processing status
    processing_status = {
        'is_processing': True,
        'progress': 0,
        'message': 'Initializing...',
        'output_dir': None,
        'error': None
    }
    
    # Start processing in a separate thread
    thread = threading.Thread(target=run_scheduler, args=(csv_file, debug))
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'started'})

@app.route('/status')
def get_status():
    """Get current processing status"""
    return jsonify(processing_status)

@app.route('/results-data')
def get_results_data():
    """Get all available results from output directory"""
    try:
        output_base = 'output'
        if not os.path.exists(output_base):
            return jsonify({'results': []})
        
        results = []
        subdirs = [d for d in os.listdir(output_base) if os.path.isdir(os.path.join(output_base, d))]
        
        for subdir in subdirs:
            subdir_path = os.path.join(output_base, subdir)
            
            # Get directory info
            dir_stats = os.stat(subdir_path)
            created_time = dir_stats.st_ctime
            modified_time = dir_stats.st_mtime
            
            # Check what files exist in this directory
            files = os.listdir(subdir_path)
            
            # Count locations by analyzing the CSV file
            location_count = 0
            day_count = 0
            csv_path = os.path.join(subdir_path, 'complete_location_schedule.csv')
            if os.path.exists(csv_path):
                try:
                    import pandas as pd
                    df = pd.read_csv(csv_path)
                    location_count = len(df)
                    day_count = len(df['Date'].unique()) if 'Date' in df.columns else 0
                except:
                    pass
            
            # Check for individual day files
            day_files = []
            for i in range(1, 10):  # Check up to 9 days
                day_html = f'location_schedule_day_{i}.html'
                day_csv = f'location_schedule_day_{i}.csv'
                if day_html in files and day_csv in files:
                    day_files.append(i)
            
            result_info = {
                'name': subdir,
                'display_name': subdir.replace('_', ' '),
                'path': subdir_path,
                'created': created_time,
                'modified': modified_time,
                'location_count': location_count,
                'day_count': day_count,
                'available_days': day_files,
                'has_combined': 'combined_map.html' in files,
                'has_complete_csv': 'complete_location_schedule.csv' in files,
                'files': files
            }
            results.append(result_info)
        
        # Sort by modification time (newest first)
        results.sort(key=lambda x: x['modified'], reverse=True)
        
        return jsonify({'results': results})
        
    except Exception as e:
        return jsonify({'error': str(e), 'results': []})

@app.route('/results')
def results_page():
    """Results page"""
    return render_template('results.html')

@app.route('/edit-csv')
def edit_csv_page():
    """CSV editing page"""
    csv_files = get_csv_files()
    return render_template('edit_csv.html', csv_files=csv_files)

@app.route('/load-csv/<filename>')
def load_csv_data(filename):
    """Load CSV data for editing"""
    try:
        import pandas as pd
        
        # Try to load from spreadsheets folder first
        filepath = os.path.join('spreadsheets', filename)
        if not os.path.exists(filepath):
            filepath = filename
            
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
            
        # Read the CSV file
        df = pd.read_csv(filepath)
        
        # Add scheduling columns if they don't exist
        if 'Scheduled_Day' not in df.columns:
            df['Scheduled_Day'] = ''
        if 'Scheduled_Time' not in df.columns:
            df['Scheduled_Time'] = ''
        if 'Priority' not in df.columns:
            df['Priority'] = 'Normal'
        if 'Notes' not in df.columns:
            df['Notes'] = ''
            
        # Clean up data for JSON serialization
        # Replace NaN values with empty strings
        df = df.fillna('')
        
        # Convert to records for JSON serialization
        data = df.to_dict('records')
        columns = list(df.columns)
        
        # Ensure all data is JSON serializable
        for record in data:
            for key, value in record.items():
                if pd.isna(value) or value is None:
                    record[key] = ''
                else:
                    record[key] = str(value)
        
        return jsonify({
            'data': data,
            'columns': columns,
            'filename': filename
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/save-csv/<filename>', methods=['POST'])
def save_csv_data(filename):
    """Save edited CSV data"""
    try:
        import pandas as pd
        
        data = request.json.get('data', [])
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        # Convert back to DataFrame
        df = pd.DataFrame(data)
        
        # Save to spreadsheets folder
        filepath = os.path.join('spreadsheets', filename)
        os.makedirs('spreadsheets', exist_ok=True)
        
        # Save with index=False to avoid adding row numbers
        df.to_csv(filepath, index=False)
        
        return jsonify({'success': True, 'message': f'Saved {filename} successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<path:filename>')
def download_file(filename):
    """Download or view generated files"""
    try:
        # Construct the full file path relative to the output directory
        full_file_path = os.path.join('output', filename)
        
        # Ensure the file path is safe and exists
        if not os.path.exists(full_file_path):
            flash(f'File not found: {filename}')
            return redirect(url_for('results_page'))
        
        # Convert to absolute path for send_file
        abs_file_path = os.path.abspath(full_file_path)
        
        # For HTML files, open in browser; for other files, download
        if filename.lower().endswith('.html'):
            return send_file(abs_file_path, as_attachment=False)
        else:
            return send_file(abs_file_path, as_attachment=True)
    except Exception as e:
        flash(f'Error accessing file: {str(e)}')
        return redirect(url_for('results_page'))

# Removed the complex view functionality - just using download now

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    print("Starting Location Visitation Scheduler Web Interface...")
    print("Open your browser to: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 