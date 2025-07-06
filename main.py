import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from sklearn.cluster import KMeans
import folium
from tqdm import tqdm
import sys
from datetime import datetime, timedelta
import time
import requests
import types
import json
import os
from openai import OpenAI
from colorama import init, Fore, Style
import threading
import itertools
import argparse

init(autoreset=True)

# Global debug flag
DEBUG_MODE = False

# Loading spinner context manager
class Spinner:
    def __init__(self, message="Loading..."):
        self.spinner = itertools.cycle(['|', '/', '-', '\\'])
        self.stop_running = False
        self.message = message
        self.thread = None
    def start(self):
        def run():
            while not self.stop_running:
                sys.stdout.write(f"\r{Fore.CYAN}{self.message} {next(self.spinner)}{Style.RESET_ALL}")
                sys.stdout.flush()
                time.sleep(0.1)
            sys.stdout.write("\r" + " " * (len(self.message) + 2) + "\r")
        self.thread = threading.Thread(target=run)
        self.thread.start()
    def stop(self):
        self.stop_running = True
        if self.thread:
            self.thread.join()

# Colored output functions
def print_info(msg):
    print(f"{Fore.GREEN}{Style.BRIGHT}✓ {msg}{Style.RESET_ALL}")

def print_warning(msg):
    print(f"{Fore.YELLOW}{Style.BRIGHT}⚠ {msg}{Style.RESET_ALL}")

def print_error(msg):
    print(f"{Fore.RED}{Style.BRIGHT}✗ {msg}{Style.RESET_ALL}")

def print_debug(msg):
    if DEBUG_MODE:
        print(f"{Fore.BLUE}{Style.DIM}🔍 {msg}{Style.RESET_ALL}")

def print_header(msg):
    print(f"\n{Fore.MAGENTA}{Style.BRIGHT}{'='*50}")
    print(f"{msg}")
    print(f"{'='*50}{Style.RESET_ALL}\n")

def input_colored(msg):
    return input(f"{Fore.CYAN}{Style.BRIGHT}{msg}{Style.RESET_ALL}")

def create_trip_output_directory(trip_name=None):
    """Create trip-specific output directory with automatic numbering."""
    # Create base output directory
    create_output_directory()
    
    # Create a single directory for this trip/run
    if trip_name:
        # Clean trip name for directory name
        clean_name = trip_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        base_dir = os.path.join('output', clean_name)
    else:
        base_dir = os.path.join('output', 'trip')
    
    dir_path = base_dir
    counter = 1
    
    while os.path.exists(dir_path):
        dir_path = f"{base_dir}_{counter}"
        counter += 1
    
    # Create the directory
    os.makedirs(dir_path)
    print_info(f"Created trip output directory: {dir_path}")
    
    return dir_path

def create_output_directory():
    """Create output directory if it doesn't exist."""
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print_debug(f"Created output directory: {output_dir}")

def create_default_config():
    """Create default config.json if it doesn't exist."""
    config_file = 'config.json'
    if not os.path.exists(config_file):
        default_config = {
            "default_buffer_minutes": 30,
            "default_visit_hours": 1,
            "start_times": ["09:00", "09:00"],  # Fixed: use start_times array
            "visit_dates": ["July 16, 2024", "July 17, 2024"],
            "geocoding": {
                "max_retries": 3,
                "max_wait_seconds": 300,
                "rate_limit_delay": 1
            },
            "clustering": {
                "n_clusters": 2
            }
        }
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        print_info(f"Created default config file: {config_file}")

def create_spreadsheets_directory():
    """Create spreadsheets directory if it doesn't exist."""
    spreadsheets_dir = 'spreadsheets'
    if not os.path.exists(spreadsheets_dir):
        os.makedirs(spreadsheets_dir)
        print_debug(f"Created spreadsheets directory: {spreadsheets_dir}")

def load_config():
    """Load configuration from config.json."""
    config_file = 'config.json'
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
            
        # Fix old config format: convert start_time string to start_times array
        if 'start_time' in config and 'start_times' not in config:
            start_time_str = config['start_time']
            if ',' in start_time_str:
                # Handle "12:00, 9:00" format
                start_times = [time.strip() for time in start_time_str.split(',')]
            else:
                # Handle single time
                start_times = [start_time_str.strip()]
            config['start_times'] = start_times
            del config['start_time']
            
            # Save updated config
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            print_info("Updated config.json to use start_times array format")
            
        return config
    else:
        print_warning("Config file not found, using default values")
        return {
            "default_buffer_minutes": 30,
            "default_visit_hours": 1,
            "start_times": ["09:00", "09:00"],
            "visit_dates": ["July 16, 2024", "July 17, 2024"]
        }

def read_csv(filename):
    """Read the CSV file and return a DataFrame."""
    try:
        df = pd.read_csv(filename)
        print_info(f"Successfully read {len(df)} locations from {filename}")
        if DEBUG_MODE:
            print_debug(f"Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print_error(f"Error reading CSV file: {e}")
        sys.exit(1)

def geocode_with_photon(address):
    """Fallback geocoding using Photon API with rate limiting."""
    try:
        import urllib.parse
        encoded_address = urllib.parse.quote(address)
        url = f"https://photon.komoot.io/api/?q={encoded_address}&limit=1"
        
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('features') and len(data['features']) > 0:
                coords = data['features'][0]['geometry']['coordinates']
                # Photon returns [lon, lat], we need [lat, lon]
                return coords[1], coords[0]  # lat, lon
    except Exception as e:
        if DEBUG_MODE:
            print_debug(f"Photon API error: {e}")
    return None

def geocode_addresses(df, cache_file='geocode_cache.json', config=None):
    """Geocode addresses using Nominatim with persistent cache and retry with exponential backoff."""
    if config is None:
        config = load_config()
    
    geolocator = Nominatim(user_agent="location_scheduler")
    
    # Verify column names exist in the DataFrame
    if DEBUG_MODE:
        print_debug(f"Available columns: {list(df.columns)}")
    
    # Determine the correct column names with flexible detection
    name_col = None
    address_col = None
    price_col = None
    time_col = None
    
    # Check for name column (apartment, event, location, name, etc.)
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['apartment', 'event', 'location', 'name', 'venue', 'place']):
            name_col = col
            break
    
    # Check for address column
    for col in df.columns:
        if 'address' in col.lower():
            address_col = col
            break
    
    # Check for price/cost column
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['price', 'cost', 'fee', 'rate']):
            price_col = col
            break
    
    # Check for time/buffer column
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['time', 'buffer', 'duration', 'wait']):
            time_col = col
            break
    
    if not name_col or not address_col:
        print_error("Error: Could not find required columns. Available columns: " + str(list(df.columns)))
        print_error("Please ensure your CSV has columns for location/event name and address.")
        return pd.DataFrame()
    
    if DEBUG_MODE:
        print_debug(f"Using columns: {name_col}, {address_col}, {price_col if price_col else 'N/A'}, {time_col if time_col else 'N/A'}")
    
    # Load OpenAI API key
    openai_client = None
    try:
        with open('api_keys.json', 'r') as f:
            api_keys = json.load(f)
            api_key = api_keys.get('openai_api_key')
            if api_key and api_key != "your-openai-api-key-here":
                openai_client = OpenAI(api_key=api_key)
                print_info("OpenAI API key loaded successfully")
            else:
                print_warning("OpenAI API key not configured, reformatting will be disabled")
    except FileNotFoundError:
        print_warning("Warning: api_keys.json not found. OpenAI reformatting will be disabled.")
    except KeyError:
        print_warning("Warning: openai_api_key not found in api_keys.json. OpenAI reformatting will be disabled.")
    except Exception as e:
        print_warning(f"Warning: Error loading OpenAI API key: {e}. OpenAI reformatting will be disabled.")
    
    # Load cache if exists
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache = json.load(f)
    else:
        cache = {}
    
    print_info("Geocoding addresses...")
    spinner = Spinner("Geocoding addresses")
    spinner.start()
    
    # Process each row and store results
    results = []
    for idx, row in df.iterrows():
        address = row[address_col]
        location_name = row[name_col]
        price_info = row[price_col] if price_col else "N/A"
        
        # Get custom buffer time if available, otherwise use default
        buffer_minutes = config.get('default_buffer_minutes', 30)
        if time_col and pd.notna(row[time_col]):
            try:
                buffer_minutes = int(row[time_col])
            except (ValueError, TypeError):
                pass
        
        # Check cache first
        if address in cache:
            cached_result = cache[address]
            if cached_result is not None:
                # Handle both old tuple format and new dict format
                if isinstance(cached_result, dict):
                    lat, lon = cached_result['lat'], cached_result['lon']
                elif isinstance(cached_result, (list, tuple)) and len(cached_result) == 2:
                    lat, lon = cached_result[0], cached_result[1]
                else:
                    # Invalid cache format, skip
                    if DEBUG_MODE:
                        print_debug(f"Invalid cache format for {address}, re-geocoding")
                    cached_result = None
                if cached_result is not None:
                    if DEBUG_MODE:
                        print_debug(f"Using cached result for: {address}")
                    results.append({
                        'name': location_name,
                        'address': address,
                        'price_info': price_info,
                        'latitude': lat,
                        'longitude': lon,
                        'buffer_minutes': buffer_minutes
                    })
                    continue
            # If cached_result is None, treat as cache miss and retry geocoding
        
        # Geocode the address
        max_retries = config.get('geocoding', {}).get('max_retries', 3)
        max_wait = config.get('geocoding', {}).get('max_wait_seconds', 300)
        rate_limit_delay = config.get('geocoding', {}).get('rate_limit_delay', 1)
        
        location = None
        tried_addresses = set()
        photon_tried = False
        
        for attempt in range(max_retries):
            try:
                if DEBUG_MODE:
                    print_debug(f"Attempting to geocode: {address} (attempt {attempt + 1})")
                
                location = geolocator.geocode(address)
                
                if location is not None:
                    break
                else:
                    if DEBUG_MODE:
                        print_debug(f"FAILURE: Geocoder returned None for address: {address}")
                        print_debug("This usually means the address couldn't be found or is invalid")
                    
                    # Try OpenAI reformatting if available
                    if openai_client and address not in tried_addresses:
                        if DEBUG_MODE:
                            print_debug(f"Trying OpenAI to reformat address: {address}")
                        
                        reformatted_address = reformat_address_with_openai(address, openai_client)
                        if reformatted_address and reformatted_address != address:
                            tried_addresses.add(address)
                            address = reformatted_address
                            if DEBUG_MODE:
                                print_debug(f"OpenAI suggested: {reformatted_address}")
                            continue
                        else:
                            if DEBUG_MODE:
                                print_debug("OpenAI suggestion already tried or same as original, skipping...")
                            break
                    else:
                        break
                        
            except (GeocoderTimedOut, GeocoderUnavailable, requests.exceptions.RequestException) as e:
                if DEBUG_MODE:
                    print_debug(f"Network error on attempt {attempt + 1}: {e}")
                
                if attempt < max_retries - 1:
                    wait_time = min(rate_limit_delay * (2 ** attempt), max_wait)
                    if DEBUG_MODE:
                        print_debug(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    if DEBUG_MODE:
                        print_debug(f"Max retries reached for {address}")
                    break
        
        # If Nominatim failed, try Photon API as fallback
        if location is None and not photon_tried:
            if DEBUG_MODE:
                print_debug(f"Trying Photon API as fallback for: {address}")
            
            # Respect Photon's 1 request/second rate limit
            time.sleep(1)
            
            photon_result = geocode_with_photon(address)
            if photon_result:
                lat, lon = photon_result
                if DEBUG_MODE:
                    print_debug(f"SUCCESS: Photon API found coordinates for {address}: ({lat}, {lon})")
                location = type('Location', (), {'latitude': lat, 'longitude': lon})()
                photon_tried = True
            else:
                if DEBUG_MODE:
                    print_debug(f"Photon API also failed for: {address}")
        
        # Cache the result (even if None)
        if location is not None and hasattr(location, 'latitude') and hasattr(location, 'longitude'):
            cache[address] = {
                'lat': location.latitude,
                'lon': location.longitude
            }
            if DEBUG_MODE:
                print_debug(f"SUCCESS: Found coordinates for {address}: ({location.latitude}, {location.longitude})")
            results.append({
                'name': location_name,
                'address': address,
                'price_info': price_info,
                'latitude': location.latitude,
                'longitude': location.longitude,
                'buffer_minutes': buffer_minutes
            })
        else:
            cache[address] = None
            if DEBUG_MODE:
                print_debug(f"SKIPPING: Failed to geocode {address} after {max_retries} attempts!")
                print_debug("Address appears to be invalid or not found in the database")
    
    spinner.stop()
    
    # Save updated cache
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2)
    
    # Convert results to DataFrame
    if results:
        df_result = pd.DataFrame(results)
        print_info(f"Successfully processed {len(df_result)} locations (skipped {len(df) - len(df_result)} invalid addresses)")
        return df_result
    else:
        print_error("No addresses were successfully geocoded!")
        return pd.DataFrame()

def reformat_address_with_openai(address, client):
    """Reformat address using OpenAI for better geocoding success."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an address formatting expert. Reformulate the given address to be more geocoding-friendly while preserving the essential location information."},
                {"role": "user", "content": f"Please reformat this address for better geocoding: {address}"}
            ],
            max_tokens=100,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        if DEBUG_MODE:
            print_debug(f"OpenAI reformatting failed: {e}")
        return None

def cluster_locations(df, n_clusters=2):
    """Cluster locations into multiple days using K-means."""
    if len(df) == 0:
        return df
    
    # Filter out any rows with NaN coordinates
    df_clean = df.dropna(subset=['latitude', 'longitude'])
    
    if len(df_clean) == 0:
        print_warning("No valid coordinates found for clustering")
        return df
    
    if len(df_clean) < n_clusters:
        print_warning(f"Not enough locations ({len(df_clean)}) for {n_clusters} clusters. Using {len(df_clean)} clusters instead.")
        n_clusters = len(df_clean)
    
    # Extract coordinates
    coords = df_clean[['latitude', 'longitude']].values
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_clean['cluster'] = kmeans.fit_predict(coords)
    
    # Merge back with original dataframe to preserve all rows
    df = df.merge(df_clean[['latitude', 'longitude', 'cluster']], 
                  on=['latitude', 'longitude'], 
                  how='left')
    
    print_info(f"Clustered {len(df_clean)} locations into {n_clusters} days")
    return df

def calculate_distance_matrix(coords):
    """Calculate distance matrix between all coordinates."""
    n = len(coords)
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                distances[i][j] = geodesic(coords[i], coords[j]).kilometers
    
    return distances

def nearest_neighbor_route(coords):
    """Find nearest neighbor route through all coordinates."""
    n = len(coords)
    if n <= 1:
        return list(range(n))
    
    distances = calculate_distance_matrix(coords)
    unvisited = set(range(n))
    route = [0]  # Start with first location
    unvisited.remove(0)
    
    while unvisited:
        current = route[-1]
        nearest = min(unvisited, key=lambda x: distances[current][x])
        route.append(nearest)
        unvisited.remove(nearest)
    
    return route

def optimize_routes(df):
    """Optimize routes within each cluster using nearest neighbor algorithm."""
    if len(df) == 0:
        return df
    
    optimized_df = pd.DataFrame()
    
    for cluster_id in df['cluster'].unique():
        cluster_data = df[df['cluster'] == cluster_id].copy()
        
        if len(cluster_data) > 1:
            # Get coordinates for this cluster
            coords = cluster_data[['latitude', 'longitude']].values
            
            # Find optimal route
            route = nearest_neighbor_route(coords)
            
            # Reorder cluster data according to route
            cluster_data = cluster_data.iloc[route].reset_index(drop=True)
        
        optimized_df = pd.concat([optimized_df, cluster_data], ignore_index=True)
    
    print_info("Optimized routes within each day")
    return optimized_df

def create_schedule(df, config=None):
    """Create a schedule for the locations."""
    if config is None:
        config = load_config()
    
    schedule = []
    
    # Define visit dates from config
    dates = config.get('visit_dates', ['July 16, 2024', 'July 17, 2024'])
    start_times = config.get('start_times', ["09:00", "09:00"])
    visit_hours = config.get('default_visit_hours', 1)
    
    for cluster_id in df['cluster'].dropna().unique():
        cluster_data = df[df['cluster'] == cluster_id].copy()
        date = dates[int(cluster_id)]
        
        # Start time for each day
        current_time = datetime.strptime(start_times[int(cluster_id)], '%H:%M')
        
        for idx, row in cluster_data.iterrows():
            # Get custom buffer time for this location, or use default
            buffer_minutes = row.get('buffer_minutes', config.get('default_buffer_minutes', 30))
            
            # Add visit duration for the visit
            end_time = current_time + timedelta(hours=visit_hours)
            
            schedule.append({
                'Date': date,
                'Time': f"{current_time.strftime('%I:%M %p')} - {end_time.strftime('%I:%M %p')}",
                'Location': row['name'],
                'Address': row['address'],
                'Price/Cost': row['price_info'],
                'Buffer Time': f"{buffer_minutes} minutes",
                'Cluster': f"Day {cluster_id + 1}"
            })
            
            # Add buffer time + move to next location
            current_time = end_time + timedelta(minutes=buffer_minutes)
    
    return pd.DataFrame(schedule)

def suggest_times_for_unlocated(df_located, unlocated_df, config=None):
    """Suggest times for unlocated addresses using OpenAI or algorithm, always at the end of the day."""
    if config is None:
        config = load_config()
    
    # Try to use OpenAI first
    try:
        api_keys_file = 'api_keys.json'
        if os.path.exists(api_keys_file):
            with open(api_keys_file, 'r') as f:
                api_keys = json.load(f)
                openai_key = api_keys.get('openai_api_key')
                
            if openai_key:
                client = OpenAI(api_key=openai_key)
                
                # Create context from located addresses
                context = "Based on these located addresses and their scheduled times (do not output these, just use for context):\n"
                for _, row in df_located.iterrows():
                    context += f"- {row['name']}: {row['address']} (Day {row['cluster'] + 1})\n"
                
                context += "\nPlease suggest visit times for these unlocated addresses, placing them at the end of the day after all other visits. Only output the unlocated addresses, their suggested times, and the day. Do NOT output the full schedule.\n"
                context += "Unlocated addresses:\n"
                
                for _, row in unlocated_df.iterrows():
                    context += f"- {row['name']}: {row['address']}\n"
                
                context += "\nRespond ONLY with the suggested times in this format (one per line):\n"
                context += "Day 1: [time] - [location name]\nDay 2: [time] - [location name]\n"
                
                try:
                    spinner = Spinner("Generating OpenAI suggestions")
                    spinner.start()
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a scheduling assistant that suggests optimal visit times for locations."},
                            {"role": "user", "content": context}
                        ],
                        max_tokens=500,
                        temperature=0.3
                    )
                    spinner.stop()
                    suggestions = response.choices[0].message.content
                    if suggestions:
                        return suggestions.strip()
                except Exception as e:
                    spinner.stop()
                    print_warning(f"OpenAI API error: {e}")
    except Exception as e:
        print_warning(f"Error with OpenAI integration: {e}")
    
    # Algorithm-based time suggestion: always at the end of the day
    dates = config.get('visit_dates', ['July 16, 2024', 'July 17, 2024'])
    start_times = config.get('start_times', ["09:00", "09:00"])
    visit_hours = config.get('default_visit_hours', 1)
    buffer_minutes = config.get('default_buffer_minutes', 30)
    
    # Calculate end times for each day (after all located visits)
    day_end_times = {}
    for cluster_id in df_located['cluster'].unique():
        cluster_data = df_located[df_located['cluster'] == cluster_id]
        if len(cluster_data) > 0:
            last_time = datetime.strptime(start_times[cluster_id], '%H:%M')
            for _, row in cluster_data.iterrows():
                last_time += timedelta(hours=visit_hours)
                last_time += timedelta(minutes=row.get('buffer_minutes', buffer_minutes))
            day_end_times[cluster_id] = last_time
        else:
            day_end_times[cluster_id] = datetime.strptime(start_times[cluster_id], '%H:%M')
    
    # Distribute unlocated addresses across days, always at the end
    suggestions = []
    unlocated_count = len(unlocated_df)
    n_days = len(dates)
    # Assign unlocated addresses to days in round-robin, but always after all located visits
    for i, (_, row) in enumerate(unlocated_df.iterrows()):
        day_choice = i % n_days
        cluster_id = day_choice
        suggested_time = day_end_times.get(cluster_id, datetime.strptime(start_times[cluster_id], '%H:%M'))
        end_time = suggested_time + timedelta(hours=visit_hours)
        time_str = f"{suggested_time.strftime('%I:%M %p')} - {end_time.strftime('%I:%M %p')}"
        suggestions.append(f"Day {cluster_id + 1} ({dates[cluster_id]}): {time_str} - {row['name']}")
        # Update end time for next unlocated on this day
        day_end_times[cluster_id] = end_time + timedelta(minutes=buffer_minutes)
    return "\n".join(suggestions)

def create_side_by_side_schedule_html(schedule_df):
    """Create HTML for schedule with days displayed side by side."""
    # Group by date
    dates = schedule_df['Date'].unique()
    
    html_parts = []
    
    # Create a row for each day
    for date in dates:
        day_data = schedule_df[schedule_df['Date'] == date]
        
        day_html = f"""
        <div class="day-column">
            <div class="day-header">{date}</div>
        """
        
        for _, visit in day_data.iterrows():
            day_html += f"""
            <div class="visit-item">
                <div class="visit-time">{visit['Time']}</div>
                <div class="visit-location">{visit['Location']}</div>
                <div class="visit-address">{visit['Address']}</div>
                <div class="visit-price">{visit['Price/Cost']}</div>
                <small class="text-muted">Buffer: {visit['Buffer Time']}</small>
            </div>
            """
        
        day_html += "</div>"
        html_parts.append(day_html)
    
    # Join all days in a row
    html_content = f"""
    <div class="row">
        {''.join(f'<div class="col-md-{12//len(dates)}">{html_part}</div>' for html_part in html_parts)}
    </div>
    """
    
    return html_content

def load_html_template(template_name='schedule_template.html'):
    """Load HTML template from templates directory."""
    template_path = os.path.join('templates', template_name)
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print_warning(f"Template {template_name} not found, using default template")
        return None

def render_html_template(template_content, **kwargs):
    """Render HTML template with provided variables."""
    if template_content is None:
        # Fallback to basic template if file not found
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{kwargs.get('title', 'Location Schedule')}</title>
            <meta charset="utf-8">
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        </head>
        <body>
            <div class="container-fluid">
                <h1>{kwargs.get('title', 'Location Schedule')}</h1>
                <div class="row">
                    <div class="col-md-6">
                        <h3>Map</h3>
                        {kwargs.get('map_html', '')}
                    </div>
                    <div class="col-md-6">
                        <h3>Schedule</h3>
                        {kwargs.get('schedule_html', '')}
                    </div>
                </div>
                {kwargs.get('unlocated_section', '')}
            </div>
        </body>
        </html>
        """
    
    # Replace placeholders in template
    rendered = template_content
    for key, value in kwargs.items():
        placeholder = f"{{{{{key}}}}}"
        rendered = rendered.replace(placeholder, str(value))
    
    return rendered

def generate_html_output(df, schedule_df, output_dir, filename='location_schedule.html'):
    """Generate HTML output with map and schedule side by side."""
    
    # Create the map
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    
    # Color scheme for clusters
    colors = ['red', 'blue']
    
    # Add markers to the map
    for idx, row in df.iterrows():
        cluster_id = row['cluster']
        if pd.isna(cluster_id):
            color = 'gray'
        else:
            color = colors[int(cluster_id)]
        
        popup_text = f"""
        <b>{row['name']}</b><br>
        Address: {row['address']}<br>
        Price/Cost: {row['price_info']}<br>
        Day: {int(cluster_id) + 1 if not pd.isna(cluster_id) else 'N/A'}
        """
        
        folium.Marker(
            [row['latitude'], row['longitude']],
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=f"{row['name']} (Day {int(cluster_id) + 1 if not pd.isna(cluster_id) else 'N/A'})",
            icon=folium.Icon(color=color, icon='home')
        ).add_to(m)
    
    # Convert map to HTML
    map_html = m._repr_html_()
    
    # Create schedule HTML with days side by side
    schedule_html = create_side_by_side_schedule_html(schedule_df)
    
    # Load and render template
    template_content = load_html_template()
    html_content = render_html_template(
        template_content,
        title="Location Visitation Schedule",
        map_html=map_html,
        schedule_html=schedule_html,
        unlocated_section=""
    )
    
    # Save to specified output directory
    output_path = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print_info(f"HTML output saved to {output_path}")

def generate_combined_html_output(df, schedule_df, output_dir, filename='location_schedule_combined.html', unlocated_df=None, unlocated_suggestions=None):
    """Generate HTML output with map and combined schedule showing all days side by side."""
    
    # Create the map
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    
    # Color scheme for clusters
    colors = ['red', 'blue']
    
    # Add markers to the map
    for idx, row in df.iterrows():
        cluster_id = row['cluster']
        if pd.isna(cluster_id):
            color = 'gray'
        else:
            color = colors[int(cluster_id)]
        
        popup_text = f"""
        <b>{row['name']}</b><br>
        Address: {row['address']}<br>
        Price/Cost: {row['price_info']}<br>
        Day: {int(cluster_id) + 1 if not pd.isna(cluster_id) else 'N/A'}
        """
        
        folium.Marker(
            [row['latitude'], row['longitude']],
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=f"{row['name']} (Day {int(cluster_id) + 1 if not pd.isna(cluster_id) else 'N/A'})",
            icon=folium.Icon(color=color, icon='home')
        ).add_to(m)
    
    # Convert map to HTML
    map_html = m._repr_html_()
    
    # Create schedule HTML with days side by side
    schedule_html = create_side_by_side_schedule_html(schedule_df)
    
    # Create unlocated section HTML
    unlocated_section = ""
    if unlocated_df is not None and len(unlocated_df) > 0:
        unlocated_list = "".join([f"• {row['name']}: {row['address']}<br>" for _, row in unlocated_df.iterrows()])
        suggestions_html = f"<br><br><strong>Suggested Times:</strong><br><pre style='white-space: pre-wrap; margin: 0;'>{unlocated_suggestions}</pre>" if unlocated_suggestions else ""
        unlocated_section = f"""
        <div class="row mt-3">
            <div class="col-12">
                <div class="alert alert-warning">
                    <strong>Unlocated Addresses ({len(unlocated_df)}):</strong><br>
                    {unlocated_list}
                    {suggestions_html}
                </div>
            </div>
        </div>
        """
    
    # Load and render template
    template_content = load_html_template()
    html_content = render_html_template(
        template_content,
        title="Location Visitation Schedule - Combined View",
        map_html=map_html,
        schedule_html=schedule_html,
        unlocated_section=unlocated_section
    )

    # Save to specified output directory
    output_path = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print_info(f"Combined HTML output saved to {output_path}")

    # Auto-open the HTML file
    import webbrowser
    webbrowser.open('file://' + os.path.abspath(output_path))

def main():
    global DEBUG_MODE
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Location Visitation Scheduler')
    parser.add_argument('csv_file', nargs='?', help='CSV file to process')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()
    
    DEBUG_MODE = args.debug
    
    print_header("Location Visitation Scheduler")
    
    # Create default config and spreadsheets directory on startup
    create_default_config()
    create_spreadsheets_directory()
    
    # Load configuration
    config = load_config()
    
    # Check for spreadsheets folder and list available CSV files
    spreadsheets_dir = 'spreadsheets'
    csv_files = []
    
    if os.path.exists(spreadsheets_dir) and os.path.isdir(spreadsheets_dir):
        # Get all CSV files in the spreadsheets directory
        for file in os.listdir(spreadsheets_dir):
            if file.lower().endswith('.csv'):
                csv_files.append(os.path.join(spreadsheets_dir, file))
    
    if csv_files:
        print_info("Available CSV files in /spreadsheets/ folder:")
        for i, file_path in enumerate(csv_files, 1):
            file_name = os.path.basename(file_path)
            print(f"  {i}. {file_name}")
        
        while True:
            try:
                choice = input_colored(f"\nSelect a file (1-{len(csv_files)}) or press Enter for default: ")
                if choice == "":
                    # Use default file
                    csv_file = 'Updated_Madison_Area_Apartments.csv'
                    break
                else:
                    choice_num = int(choice)
                    if 1 <= choice_num <= len(csv_files):
                        csv_file = csv_files[choice_num - 1]
                        break
                    else:
                        print_error(f"Please enter a number between 1 and {len(csv_files)}")
            except ValueError:
                print_error("Please enter a valid number")
    else:
        # No CSV files found in spreadsheets folder, use default
        if args.csv_file:
            csv_file = args.csv_file
        else:
            csv_file = 'Updated_Madison_Area_Apartments.csv'
    
    print_info(f"Processing data from: {csv_file}")
    
    # Step 1: Read CSV
    original_df = read_csv(csv_file)
    
    # Step 2: Geocode addresses
    df = geocode_addresses(original_df, config=config)
    
    # Step 3: Identify unlocated addresses robustly
    # Detect name and address columns in original_df
    orig_name_col = None
    orig_address_col = None
    for col in original_df.columns:
        col_lower = col.lower()
        if not orig_name_col and any(keyword in col_lower for keyword in ['apartment', 'event', 'location', 'name', 'venue', 'place']):
            orig_name_col = col
        if not orig_address_col and 'address' in col_lower:
            orig_address_col = col
    
    # Use address as the unique key (or name+address if available)
    if orig_name_col and orig_address_col:
        geocoded_keys = set(zip(df['name'], df['address']))
        original_keys = set(zip(original_df[orig_name_col], original_df[orig_address_col]))
        unlocated_mask = ~original_df.apply(lambda row: (row[orig_name_col], row[orig_address_col]) in geocoded_keys, axis=1)
    elif orig_address_col:
        geocoded_keys = set(df['address'])
        original_keys = set(original_df[orig_address_col])
        unlocated_mask = ~original_df[orig_address_col].isin(df['address'])
    else:
        print_error('Could not find address column in original CSV!')
        unlocated_mask = pd.Series([False]*len(original_df))
    
    unlocated_df = original_df[unlocated_mask].copy()
    
    # Ensure unlocated_df has the same column structure as df
    if len(unlocated_df) > 0:
        # Determine column names from the geocoding process
        name_col = None
        address_col = None
        price_col = None
        
        # Check for name column (apartment, event, location, name, etc.)
        for col in unlocated_df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['apartment', 'event', 'location', 'name', 'venue', 'place']):
                name_col = col
                break
        
        # Check for address column
        for col in unlocated_df.columns:
            if 'address' in col.lower():
                address_col = col
                break
        
        # Check for price/cost column
        for col in unlocated_df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['price', 'cost', 'fee', 'rate']):
                price_col = col
                break
        
        # Rename columns to match the processed DataFrame
        if name_col and name_col != 'name':
            unlocated_df = unlocated_df.rename(columns={name_col: 'name'})
        if address_col and address_col != 'address':
            unlocated_df = unlocated_df.rename(columns={address_col: 'address'})
        if price_col and price_col != 'price_info':
            unlocated_df = unlocated_df.rename(columns={price_col: 'price_info'})
        elif not price_col:
            unlocated_df['price_info'] = 'N/A'
    
    # Step 4: Cluster locations (only for located addresses)
    df = cluster_locations(df, n_clusters=config.get('clustering', {}).get('n_clusters', 2))
    
    # Step 5: Optimize routes
    df = optimize_routes(df)
    
    # Step 6: Create schedule
    schedule_df = create_schedule(df, config=config)
    
    # Step 7: Generate time suggestions for unlocated addresses
    unlocated_suggestions = None
    if len(unlocated_df) > 0:
        print_info(f"Generating time suggestions for {len(unlocated_df)} unlocated addresses...")
        unlocated_suggestions = suggest_times_for_unlocated(df, unlocated_df, config)
        print_info("Time suggestions generated for unlocated addresses.")
    
    # Step 8: Create output directory using trip name (from CSV filename, no extension)
    trip_name = os.path.splitext(os.path.basename(csv_file))[0]
    trip_output_dir = create_trip_output_directory(trip_name)
    
    # Step 9: Generate combined HTML output with all days side by side
    generate_combined_html_output(df, schedule_df, trip_output_dir, 'location_schedule_combined.html', unlocated_df, unlocated_suggestions)
    
    # Step 10: Generate HTML output for each day (for reference)
    for cluster_id in df['cluster'].unique():
        cluster_data = df[df['cluster'] == cluster_id].copy()
        cluster_schedule = schedule_df[schedule_df['Cluster'] == f"Day {cluster_id + 1}"]
        
        if len(cluster_data) > 0:
            # Generate HTML for this day
            generate_html_output(cluster_data, cluster_schedule, trip_output_dir, f'location_schedule_day_{cluster_id + 1}.html')
            
            # Save CSV schedule for this day
            csv_filename = f"location_schedule_day_{cluster_id + 1}.csv"
            csv_path = os.path.join(trip_output_dir, csv_filename)
            # Reset index to ensure no index column is included
            cluster_schedule.reset_index(drop=True).to_csv(csv_path, index=False)
            if DEBUG_MODE:
                print_debug(f"Schedule saved to {csv_path}")
    
    # Step 11: Save complete schedule to trip directory
    complete_schedule_path = os.path.join(trip_output_dir, 'complete_location_schedule.csv')
    # Reset index to ensure no index column is included
    schedule_df.reset_index(drop=True).to_csv(complete_schedule_path, index=False)
    if DEBUG_MODE:
        print_debug(f"Complete schedule saved to {complete_schedule_path}")
    
    print_header("Summary")
    print_info(f"Total locations processed: {len(df)}")
    for cluster_id in df['cluster'].dropna().unique():
        date = config.get('visit_dates', ['July 16, 2024', 'July 17, 2024'])[int(cluster_id)]
        count = len(df[df['cluster'] == cluster_id])
        print_info(f"Day {int(cluster_id) + 1} ({date}): {count} locations")
    
    if len(unlocated_df) > 0:
        print_warning(f"Unlocated addresses: {len(unlocated_df)} (shown at bottom of combined schedule)")
    
    print_info(f"All output files saved to: {trip_output_dir}")
    print_info("Combined HTML file opened automatically in your browser!")

if __name__ == "__main__":
    main() 