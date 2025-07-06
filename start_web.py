#!/usr/bin/env python3
"""
Startup script for the Location Visitation Scheduler Web Interface
"""

import sys
import subprocess
import os

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'flask',
        'pandas',
        'folium',
        'geopy',
        'sklearn',
        'numpy',
        'colorama',
        'openai',
        'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("[ERROR] Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n[TIP] To install missing packages, run:")
        print("   pip install -r requirements-web.txt")
        print("\n   Or install individually:")
        for package in missing_packages:
            print(f"   pip install {package}")
        return False
    
    print("[OK] All required packages are installed!")
    return True

def main():
    """Main startup function"""
    print("Location Visitation Scheduler - Web Interface")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('main.py'):
        print("[ERROR] main.py not found in current directory")
        print("   Please run this script from the ScheduleMap directory")
        sys.exit(1)
    
    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    # Create necessary directories
    os.makedirs('spreadsheets', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    print("[OK] Created necessary directories")
    
    # Check if config file exists
    if not os.path.exists('config.json'):
        print("[CONFIG] Creating default configuration...")
        # Import and run the config creation from main.py
        try:
            sys.path.append('.')
            from main import create_default_config
            create_default_config()
            print("[OK] Default configuration created")
        except Exception as e:
            print(f"[WARNING] Could not create default config: {e}")
    
    print("\nStarting web interface...")
    print("   Open your browser to: http://localhost:5000")
    print("   Press Ctrl+C to stop the server")
    print("-" * 50)
    
    # Start the web interface
    try:
        from webmain import app
        import webbrowser
        import threading
        
        # Open browser after a short delay to ensure server is running
        def open_browser():
            import time
            time.sleep(1.5)  # Wait for server to start
            webbrowser.open('http://localhost:5000')
        
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        app.run(debug=False, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n\nWeb interface stopped. Goodbye!")
    except Exception as e:
        print(f"\n[ERROR] Error starting web interface: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 