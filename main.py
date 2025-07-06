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
import re
import webbrowser
try:
    import openrouteservice
except ImportError:
    openrouteservice = None

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
    try:
        print(f"{Fore.GREEN}{Style.BRIGHT}[OK] {msg}{Style.RESET_ALL}")
    except UnicodeEncodeError:
        print(f"[OK] {msg}")

def print_warning(msg):
    try:
        print(f"{Fore.YELLOW}{Style.BRIGHT}[WARNING] {msg}{Style.RESET_ALL}")
    except UnicodeEncodeError:
        print(f"[WARNING] {msg}")

def print_error(msg):
    try:
        print(f"{Fore.RED}{Style.BRIGHT}[ERROR] {msg}{Style.RESET_ALL}")
    except UnicodeEncodeError:
        print(f"[ERROR] {msg}")

def print_debug(msg):
    if DEBUG_MODE:
        try:
            print(f"{Fore.BLUE}{Style.DIM}[DEBUG] {msg}{Style.RESET_ALL}")
        except UnicodeEncodeError:
            print(f"[DEBUG] {msg}")

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
            "min_start_time": "08:00",
            "max_end_time": "22:00",
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
        if 'unnamed' in col_lower:
            continue  # Always skip unnamed columns
        if any(keyword in col_lower for keyword in ['apartment', 'event', 'location', 'name', 'venue', 'place']):
            name_col = col
            break
    
    # If no name column found, fallback to first non-unnamed column
    if not name_col:
        for col in df.columns:
            if 'unnamed' in col.lower():
                continue
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
    
    # Check for buffer/time column (prioritize buffer-specific columns)
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['buffer', 'duration', 'wait']):
            time_col = col
            break
    
    # If no buffer column found, look for time columns
    if not time_col:
        for col in df.columns:
            col_lower = col.lower()
            if 'time' in col_lower and 'scheduled' not in col_lower:
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
                # Clean the key (remove any whitespace)
                api_key = api_key.strip()
                if DEBUG_MODE:
                    masked = api_key[:4] + "..." + api_key[-4:]
                    print_debug(f"Loaded OpenAI key: {masked} (length: {len(api_key)})")
                    print_debug(f"Key starts with: '{api_key[:10]}...'")
                    print_debug(f"Key ends with: '...{api_key[-10:]}'")
                try:
                    openai_client = OpenAI(api_key=api_key)
                    print_info("OpenAI API key loaded successfully")
                except Exception as e:
                    print_warning(f"OpenAI API key test failed: {e}")
                    print_warning("Disabling OpenAI functionality - will use Photon API fallback only")
                    openai_client = None
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
                    
                    # Create result with all original columns preserved
                    result = {
                        'name': location_name,
                        'address': address,
                        'original_address': address,
                        'price_info': price_info,
                        'latitude': lat,
                        'longitude': lon,
                        'buffer_minutes': buffer_minutes
                    }
                    
                    # Add all other columns from the original row
                    for col in df.columns:
                        if col not in [name_col, address_col, price_col, time_col]:
                            result[col] = row[col]
                    
                    results.append(result)
                    continue
            # If cached_result is None, treat as cache miss and retry geocoding
        
        # Geocode the address
        max_retries = config.get('geocoding', {}).get('max_retries', 3)
        max_wait = config.get('geocoding', {}).get('max_wait_seconds', 300)
        rate_limit_delay = config.get('geocoding', {}).get('rate_limit_delay', 1)
        
        location = None
        tried_addresses = set()
        photon_tried = False
        original_address = address  # Store the original address
        
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
        
        # Cache the result (even if None) using the final address that was tried
        if location is not None and hasattr(location, 'latitude') and hasattr(location, 'longitude'):
            cache[address] = {
                'lat': location.latitude,
                'lon': location.longitude
            }
            if DEBUG_MODE:
                print_debug(f"SUCCESS: Found coordinates for {address}: ({location.latitude}, {location.longitude})")
            
            # Use the successfully geocoded address (which may be different from original)
            # This ensures the unlocated address detection works correctly
            result = {
                'name': location_name,
                'address': address,  # Use the reformatted address that worked
                'original_address': original_address,  # Keep track of the original for reference
                'price_info': price_info,
                'latitude': location.latitude,
                'longitude': location.longitude,
                'buffer_minutes': buffer_minutes
            }
            
            # Add all other columns from the original row
            for col in df.columns:
                if col not in [name_col, address_col, price_col, time_col]:
                    result[col] = row[col]
            
            results.append(result)
        else:
            cache[original_address] = None  # Cache the original address as None
            if DEBUG_MODE:
                print_debug(f"SKIPPING: Failed to geocode {original_address} after {max_retries} attempts!")
                print_debug("Address appears to be invalid or not found in the database")
    
    spinner.stop()
    
    # Save updated cache
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2)
    
    # Convert results to DataFrame
    if results:
        df_result = pd.DataFrame(results)
        # Count actual unlocated addresses by comparing original addresses with geocoded ones
        original_addresses = set(df[address_col].tolist())
        geocoded_addresses = set(df_result['address'].tolist())
        unlocated_count = len(original_addresses - geocoded_addresses)
        print_info(f"Successfully processed {len(df_result)} locations (skipped {unlocated_count} invalid addresses)")
        return df_result
    else:
        print_error("No addresses were successfully geocoded!")
        return pd.DataFrame()

def reformat_address_with_openai(address, client):
    """Reformat address using OpenAI for better geocoding success."""
    try:
        if DEBUG_MODE:
            print_debug(f"Making OpenAI API call to reformat: {address}")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an address formatting expert. Reformulate the given address to be more geocoding-friendly while preserving the essential location information."},
                {"role": "user", "content": f"Please reformat this address for better geocoding: {address}"}
            ],
            max_tokens=100,
            temperature=0.3
        )
        result = response.choices[0].message.content.strip()
        if DEBUG_MODE:
            print_debug(f"OpenAI reformatting successful: {address} -> {result}")
        return result
    except Exception as e:
        if DEBUG_MODE:
            print_debug(f"OpenAI reformatting failed: {type(e).__name__}: {e}")
            if hasattr(e, 'response') and hasattr(e.response, 'json'):
                try:
                    error_detail = e.response.json()
                    print_debug(f"OpenAI API error details: {error_detail}")
                except:
                    pass
        return None

# OpenRouteService functions
def load_ors_client(config=None):
    """Load OpenRouteService client with API key."""
    if config is None:
        config = load_config()
    
    # Check if openrouteservice is available
    if openrouteservice is None:
        if DEBUG_MODE:
            print_debug("openrouteservice package not available")
        return None
    
    # Check if ORS is enabled in config
    if not config.get('ors', {}).get('enabled', True):
        if DEBUG_MODE:
            print_debug("ORS is disabled in config")
        return None
    
    try:
        with open('api_keys.json', 'r') as f:
            api_keys = json.load(f)
            ors_api_key = api_keys.get('ors_api_key')
            if ors_api_key and ors_api_key != "your-openrouteservice-api-key-here":
                # Clean the key (remove any whitespace)
                ors_api_key = ors_api_key.strip()
                try:
                    client = openrouteservice.Client(key=ors_api_key)
                    if DEBUG_MODE:
                        print_debug("ORS client loaded successfully")
                    print_info("OpenRouteService API key loaded successfully")
                    return client
                except Exception as e:
                    print_warning(f"ORS API key test failed: {e}")
                    print_warning("Disabling ORS functionality - will use estimated travel times")
                    return None
            else:
                print_warning("ORS API key not configured, real driving times will be disabled")
                return None
    except FileNotFoundError:
        print_warning("Warning: api_keys.json not found. ORS functionality will be disabled.")
        return None
    except KeyError:
        print_warning("Warning: ors_api_key not found in api_keys.json. ORS functionality will be disabled.")
        return None
    except Exception as e:
        print_warning(f"Warning: Error loading ORS API key: {e}. ORS functionality will be disabled.")
        return None

def get_ors_route_info(client, coords_list, config=None):
    """Get route information including duration and distance from ORS."""
    if not client or len(coords_list) < 2:
        return None
    
    if config is None:
        config = load_config()
    
    ors_config = config.get('ors', {})
    profile = ors_config.get('profile', 'driving-car')
    
    try:
        # Convert coordinates to [lon, lat] format for ORS
        ors_coords = [[coord[1], coord[0]] for coord in coords_list]
        
        # Get route information
        route = client.directions(
            coordinates=ors_coords,
            profile=profile,
            format='geojson',
            instructions=True
        )
        
        if route and 'features' in route and len(route['features']) > 0:
            properties = route['features'][0]['properties']
            
            # Extract route information
            duration_seconds = properties.get('summary', {}).get('duration', 0)
            distance_meters = properties.get('summary', {}).get('distance', 0)
            
            # Get turn-by-turn directions
            segments = properties.get('segments', [])
            directions = []
            for segment in segments:
                for step in segment.get('steps', []):
                    instruction = step.get('instruction', '')
                    if instruction:
                        directions.append(instruction)
            
            return {
                'duration_seconds': duration_seconds,
                'duration_minutes': duration_seconds / 60,
                'distance_meters': distance_meters,
                'distance_km': distance_meters / 1000,
                'directions': directions,
                'route_geometry': route['features'][0]['geometry']
            }
    except Exception as e:
        if DEBUG_MODE:
            print_debug(f"ORS route request failed: {e}")
        return None
    
    return None

def calculate_ors_travel_times(df, config=None, cache_file='ors_cache.json'):
    """Calculate travel times between consecutive locations using ORS with caching."""
    if config is None:
        config = load_config()
    
    ors_client = load_ors_client(config)
    if not ors_client:
        print_warning("ORS client not available, using estimated travel times")
        return df
    
    # Load ORS cache if exists
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            ors_cache = json.load(f)
    else:
        ors_cache = {}
    
    print_info("Calculating travel times using OpenRouteService...")
    spinner = Spinner("Calculating travel times")
    spinner.start()
    
    # Add columns for ORS data with appropriate data types
    df['ors_travel_time_minutes'] = 0.0  # Use float to avoid dtype warnings
    df['ors_travel_distance_km'] = 0.0   # Use float to avoid dtype warnings
    df['ors_directions'] = ''
    df['ors_route_geometry'] = ''
    
    ors_config = config.get('ors', {})
    rate_limit_delay = ors_config.get('rate_limit_delay', 0.5)
    
    try:
        # Process each cluster separately
        for cluster_id in df['cluster'].unique():
            cluster_data = df[df['cluster'] == cluster_id].copy()
            
            if len(cluster_data) < 2:
                continue
                
            # Get coordinates for this cluster in route order
            coords = []
            indices = []
            for idx, row in cluster_data.iterrows():
                coords.append([row['latitude'], row['longitude']])
                indices.append(idx)
            
            # Calculate travel times between consecutive locations
            for i in range(len(coords) - 1):
                current_coords = [coords[i], coords[i + 1]]
                
                # Create cache key from coordinates
                cache_key = f"{current_coords[0][0]:.6f},{current_coords[0][1]:.6f}_to_{current_coords[1][0]:.6f},{current_coords[1][1]:.6f}_{ors_config.get('profile', 'driving-car')}"
                
                # Check cache first
                route_info = None
                if cache_key in ors_cache:
                    route_info = ors_cache[cache_key]
                    if DEBUG_MODE:
                        print_debug(f"Using cached ORS data for route segment")
                else:
                    # Get route info from ORS API
                    route_info = get_ors_route_info(ors_client, current_coords, config)
                    if route_info:
                        # Cache the result
                        ors_cache[cache_key] = route_info
                        if DEBUG_MODE:
                            print_debug(f"Cached ORS data for route segment")
                    
                    # Rate limiting only when making API calls
                    time.sleep(rate_limit_delay)
                
                if route_info:
                    # Store the travel time to reach the next location
                    next_idx = indices[i + 1]
                    df.at[next_idx, 'ors_travel_time_minutes'] = route_info['duration_minutes']
                    df.at[next_idx, 'ors_travel_distance_km'] = route_info['distance_km']
                    df.at[next_idx, 'ors_directions'] = json.dumps(route_info['directions'])
                    df.at[next_idx, 'ors_route_geometry'] = json.dumps(route_info['route_geometry'])
                    
                    if DEBUG_MODE:
                        print_debug(f"Travel time to {df.at[next_idx, 'name']}: {route_info['duration_minutes']:.1f} min, {route_info['distance_km']:.1f} km")
                
    except Exception as e:
        print_error(f"Error calculating ORS travel times: {e}")
    finally:
        spinner.stop()
        
        # Save ORS cache
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(ors_cache, f, indent=2)
            if DEBUG_MODE:
                print_debug(f"ORS cache saved to {cache_file}")
        except Exception as e:
            if DEBUG_MODE:
                print_debug(f"Error saving ORS cache: {e}")
    
    return df

def validate_schedule_with_ors(schedule_df, df, config=None):
    """Validate that the schedule fits within time constraints using ORS travel times."""
    if config is None:
        config = load_config()
    
    ors_config = config.get('ors', {})
    if not ors_config.get('use_for_validation', True):
        return schedule_df, []
    
    # Check if ORS client is available
    ors_client = load_ors_client(config)
    if not ors_client:
        if DEBUG_MODE:
            print_debug("ORS client not available, skipping schedule validation")
        return schedule_df, []
    
    warnings = []
    buffer_adjustments = []
    
    # Check each day's schedule
    for cluster_name in schedule_df['Cluster'].unique():
        day_schedule = schedule_df[schedule_df['Cluster'] == cluster_name].copy()
        
        if len(day_schedule) < 2:
            continue
            
        # Get the cluster ID from the cluster name
        cluster_id = int(cluster_name.split()[-1]) - 1
        
        # Check if actual travel times exceed buffer times
        for i in range(len(day_schedule) - 1):
            current_visit = day_schedule.iloc[i]
            next_visit = day_schedule.iloc[i + 1]
            
            # Find the corresponding entries in the main DataFrame
            current_match = df[(df['name'] == current_visit['Location']) & 
                             (df['address'] == current_visit['Address'])]
            next_match = df[(df['name'] == next_visit['Location']) & 
                          (df['address'] == next_visit['Address'])]
            
            if not current_match.empty and not next_match.empty:
                next_row = next_match.iloc[0]
                ors_travel_time = next_row.get('ors_travel_time_minutes', 0)
                
                # Parse buffer time from schedule
                buffer_str = next_visit.get('Buffer Time', '0 minutes')
                buffer_minutes = int(buffer_str.split()[0]) if buffer_str.split() else 0
                
                # Check if this is an explicitly requested shorter buffer
                original_buffer = next_row.get('buffer_minutes', config.get('default_buffer_minutes', 15))
                is_explicit_short_buffer = buffer_minutes < original_buffer
                
                if ors_travel_time > buffer_minutes:
                    if is_explicit_short_buffer:
                        # User explicitly requested shorter buffer - show warning but don't auto-adjust
                        warning = f"WARNING: Day {cluster_id + 1}: Travel time to {next_visit['Location']} ({ors_travel_time:.1f} min) exceeds explicitly set buffer time ({buffer_minutes} min)"
                        warnings.append(warning)
                        # Add warning flag to the schedule entry
                        schedule_df.loc[schedule_df['Location'] == next_visit['Location'], 'Buffer_Warning'] = True
                        if DEBUG_MODE:
                            print_debug(warning)
                    else:
                        # Auto-adjust buffer in 15-minute increments
                        required_buffer = int(((ors_travel_time + 14) // 15) * 15)  # Round up to nearest 15
                        adjustment = f"Day {cluster_id + 1}: Auto-adjusted buffer for {next_visit['Location']} from {buffer_minutes} to {required_buffer} min (drive time: {ors_travel_time:.1f} min)"
                        buffer_adjustments.append(adjustment)
                        
                        # Update the schedule with new buffer time
                        schedule_df.loc[schedule_df['Location'] == next_visit['Location'], 'Buffer Time'] = f"{required_buffer} minutes"
                        
                        if DEBUG_MODE:
                            print_debug(adjustment)
    
    # Add buffer adjustment info to warnings for display
    if buffer_adjustments:
        warnings.extend([f"🔧 Buffer Auto-Adjustments:"] + buffer_adjustments)
    
    return schedule_df, warnings

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
            # Use .iloc to get the actual data rows, not the original indices
            cluster_data = cluster_data.iloc[route].copy()
            # Reset index to ensure clean indexing
            cluster_data = cluster_data.reset_index(drop=True)
        
        optimized_df = pd.concat([optimized_df, cluster_data], ignore_index=True)
    
    print_info("Optimized routes within each day")
    return optimized_df

def calculate_relocation_cost(location_idx, from_day, to_day, df, day_schedules, config=None):
    """Calculate the cost of moving a location from one day to another.
    
    Cost considers:
    - Travel time impact on both days
    - Geographic clustering efficiency
    - Schedule disruption
    """
    if config is None:
        config = load_config()
    
    # Get location data
    location = df.loc[location_idx]
    location_coords = (location['latitude'], location['longitude'])
    
    # Calculate cost for removing from original day
    from_day_cost = 0
    from_day_locations = df[df['cluster'] == from_day]
    if len(from_day_locations) > 1:
        # Calculate average distance to other locations on the from_day
        distances = []
        for _, other_loc in from_day_locations.iterrows():
            if other_loc.name != location_idx:
                other_coords = (other_loc['latitude'], other_loc['longitude'])
                distances.append(geodesic(location_coords, other_coords).kilometers)
        if distances:
            from_day_cost = -np.mean(distances)  # Negative because removing reduces travel
    
    # Calculate cost for adding to target day
    to_day_cost = 0
    to_day_locations = df[df['cluster'] == to_day]
    if len(to_day_locations) > 0:
        # Calculate average distance to other locations on the to_day
        distances = []
        for _, other_loc in to_day_locations.iterrows():
            other_coords = (other_loc['latitude'], other_loc['longitude'])
            distances.append(geodesic(location_coords, other_coords).kilometers)
        if distances:
            to_day_cost = np.mean(distances)  # Positive because adding increases travel
    
    # Time constraint penalty - check if the target day has enough time
    time_penalty = 0
    visit_hours = config.get('default_visit_hours', 1)
    buffer_minutes = config.get('default_buffer_minutes', 30)
    max_end_times = config.get('max_end_times', ['22:00', '22:00'])
    
    if to_day < len(max_end_times):
        current_time = day_schedules.get(to_day, datetime.strptime('09:00', '%H:%M'))
        end_time = current_time + timedelta(hours=visit_hours)
        max_end_time = datetime.strptime(max_end_times[to_day], '%H:%M')
        
        if end_time > max_end_time:
            time_penalty = 1000  # High penalty for impossible scheduling
        else:
            # Small penalty for tight scheduling
            remaining_time = (max_end_time - end_time).total_seconds() / 3600
            if remaining_time < 1:  # Less than 1 hour remaining
                time_penalty = 50 * (1 - remaining_time)
    
    # Total cost is the sum of geographic and time costs
    total_cost = from_day_cost + to_day_cost + time_penalty
    
    if DEBUG_MODE:
        print_debug(f"Relocation cost for {location['name']} from Day {from_day + 1} to Day {to_day + 1}: "
                   f"from_day_cost={from_day_cost:.2f}, to_day_cost={to_day_cost:.2f}, "
                   f"time_penalty={time_penalty:.2f}, total={total_cost:.2f}")
    
    return total_cost

def intelligent_reschedule_location(location_idx, original_day, df, day_schedules, config=None):
    """Intelligently reschedule a location considering geographic proximity and time constraints."""
    if config is None:
        config = load_config()
    
    dates = config.get('visit_dates', ['July 16, 2024', 'July 17, 2024'])
    available_days = list(range(len(dates)))
    
    # Remove the original day from consideration
    available_days = [day for day in available_days if day != original_day]
    
    if not available_days:
        return None, float('inf')
    
    # Calculate relocation cost for each possible day
    best_day = None
    best_cost = float('inf')
    
    for candidate_day in available_days:
        cost = calculate_relocation_cost(location_idx, original_day, candidate_day, df, day_schedules, config)
        
        if cost < best_cost:
            best_cost = cost
            best_day = candidate_day
    
    # Only suggest the move if it's not impossibly expensive (time_penalty < 1000)
    if best_cost >= 1000:
        return None, best_cost
    
    return best_day, best_cost

def optimize_clusters_with_time_constraints(df, config=None):
    """Re-optimize clusters after initial scheduling to handle time constraint violations better."""
    if config is None:
        config = load_config()
    
    print_info("Optimizing clusters with time constraints...")
    
    # Get configuration
    dates = config.get('visit_dates', ['July 16', 'July 17'])
    start_times = config.get('start_times', ['09:00', '09:00'])
    max_end_times = config.get('max_end_times', ['22:00', '22:00'])
    visit_hours = config.get('default_visit_hours', 1)
    buffer_minutes = config.get('default_buffer_minutes', 30)
    
    n_days = len(dates)
    max_iterations = 5
    
    for iteration in range(max_iterations):
        improved = False
        
        # Calculate current day schedules
        day_schedules = {}
        day_loads = {}  # Track time load for each day
        
        for day in range(n_days):
            day_locations = df[df['cluster'] == day]
            total_time = len(day_locations) * (visit_hours + buffer_minutes / 60.0)
            
            start_time = datetime.strptime(start_times[day], '%H:%M')
            max_end_time = datetime.strptime(max_end_times[day], '%H:%M')
            available_hours = (max_end_time - start_time).total_seconds() / 3600.0
            
            day_schedules[day] = start_time
            day_loads[day] = total_time / available_hours if available_hours > 0 else float('inf')
        
        # Find overloaded days
        overloaded_days = [day for day, load in day_loads.items() if load > 1.0]
        
        if not overloaded_days:
            print_debug(f"Optimization iteration {iteration + 1}: All days within time constraints")
            break
        
        print_debug(f"Optimization iteration {iteration + 1}: Found {len(overloaded_days)} overloaded days")
        
        # For each overloaded day, try to move locations to less loaded days
        for overloaded_day in overloaded_days:
            day_locations = df[df['cluster'] == overloaded_day]
            
            # Sort by priority (move lower priority items first) and distance from day centroid
            if 'Priority' in day_locations.columns:
                priority_order = {'High': 0, 'Normal': 1, 'Low': 2}
                day_locations = day_locations.copy()
                day_locations['priority_score'] = day_locations['Priority'].map(priority_order).fillna(1)
            else:
                day_locations = day_locations.copy()
                day_locations['priority_score'] = 1
            
            # Calculate distance from day centroid for each location
            if len(day_locations) > 1:
                centroid_lat = day_locations['latitude'].mean()
                centroid_lon = day_locations['longitude'].mean()
                
                distances_from_centroid = []
                for _, loc in day_locations.iterrows():
                    dist = geodesic((centroid_lat, centroid_lon), (loc['latitude'], loc['longitude'])).kilometers
                    distances_from_centroid.append(dist)
                day_locations['centroid_distance'] = distances_from_centroid
            else:
                day_locations['centroid_distance'] = 0
            
            # Sort by priority (low priority first) and distance (far from centroid first)
            day_locations = day_locations.sort_values(['priority_score', 'centroid_distance'], 
                                                     ascending=[False, False])
            
            # Try to move locations until the day is no longer overloaded
            for idx, location in day_locations.iterrows():
                if day_loads[overloaded_day] <= 1.0:
                    break  # Day is no longer overloaded
                
                # Find the best alternative day for this location
                best_day, best_cost = intelligent_reschedule_location(idx, overloaded_day, df, day_schedules, config)
                
                if best_day is not None and best_cost < 1000:
                    # Move the location
                    df.loc[idx, 'cluster'] = best_day
                    improved = True
                    
                    # Update day loads
                    location_time = visit_hours + buffer_minutes / 60.0
                    
                    # Remove from overloaded day
                    old_total_time = day_loads[overloaded_day] * (datetime.strptime(max_end_times[overloaded_day], '%H:%M') - datetime.strptime(start_times[overloaded_day], '%H:%M')).total_seconds() / 3600.0
                    new_total_time = old_total_time - location_time
                    available_hours = (datetime.strptime(max_end_times[overloaded_day], '%H:%M') - datetime.strptime(start_times[overloaded_day], '%H:%M')).total_seconds() / 3600.0
                    day_loads[overloaded_day] = new_total_time / available_hours if available_hours > 0 else 0
                    
                    # Add to new day
                    old_total_time = day_loads[best_day] * (datetime.strptime(max_end_times[best_day], '%H:%M') - datetime.strptime(start_times[best_day], '%H:%M')).total_seconds() / 3600.0
                    new_total_time = old_total_time + location_time
                    available_hours = (datetime.strptime(max_end_times[best_day], '%H:%M') - datetime.strptime(start_times[best_day], '%H:%M')).total_seconds() / 3600.0
                    day_loads[best_day] = new_total_time / available_hours if available_hours > 0 else float('inf')
                    
                    if DEBUG_MODE:
                        print_debug(f"Moved {location['name']} from Day {overloaded_day + 1} to Day {best_day + 1}")
                        print_debug(f"Day {overloaded_day + 1} load: {day_loads[overloaded_day]:.2f} -> Day {best_day + 1} load: {day_loads[best_day]:.2f}")
        
        if not improved:
            print_debug(f"Optimization iteration {iteration + 1}: No beneficial moves found")
            break
    
    # Final load report
    for day in range(n_days):
        load = day_loads[day]
        status = "OK" if load <= 1.0 else "OVERLOADED"
        print_debug(f"Final Day {day + 1} load: {load:.2f} ({status})")
    
    print_info("Cluster optimization completed")
    return df

def create_schedule(df, config=None):
    """Create a schedule for the locations."""
    if config is None:
        config = load_config()
    
    schedule = []
    
    # Define visit dates from config
    dates = config.get('visit_dates', ['July 16, 2024', 'July 17, 2024'])
    start_times = config.get('start_times', ["09:00", "09:00"])
    visit_hours = config.get('default_visit_hours', 1)
    
    # Get time constraints as arrays
    max_end_times = config.get('max_end_times', ['22:00', '22:00'])
    
    # First, handle locations with specific day/time constraints
    scheduled_locations = set()
    day_schedules = {}  # Track current time for each day
    
    # Initialize day schedules
    for i in range(len(dates)):
        day_schedules[i] = datetime.strptime(start_times[i], '%H:%M')
    
    # Process high priority and specifically scheduled items first
    for cluster_id in df['cluster'].dropna().unique():
        cluster_data = df[df['cluster'] == cluster_id].copy()
        
        # Sort by priority (High -> Normal -> Low) and then by scheduled time
        def get_priority_order(priority):
            priority_order = {'High': 0, 'Normal': 1, 'Low': 2}
            return priority_order.get(priority, 1)
        
        # Only sort by priority if the column exists
        if 'Priority' in cluster_data.columns:
            cluster_data = cluster_data.sort_values(by=['Priority'], key=lambda x: x.map(get_priority_order))
        else:
            # Add default priority column if it doesn't exist
            cluster_data['Priority'] = 'Normal'
        
        for idx, row in cluster_data.iterrows():
            # Check if this location has specific scheduling constraints
            scheduled_day_raw = row.get('Scheduled_Day', '')
            scheduled_time_raw = row.get('Scheduled_Time', '')
            
            # Handle NaN values and convert to string
            scheduled_day = str(scheduled_day_raw).strip() if pd.notna(scheduled_day_raw) else ''
            scheduled_time = str(scheduled_time_raw).strip() if pd.notna(scheduled_time_raw) else ''
            
            # Determine target day
            target_day = None
            if scheduled_day:
                if scheduled_day.lower() == 'any':
                    # Can be scheduled on any day - use original cluster assignment
                    target_day = int(cluster_id)
                elif scheduled_day.lower().startswith('day '):
                    # Specific day requested
                    try:
                        target_day = int(scheduled_day.split()[1]) - 1
                    except (ValueError, IndexError):
                        target_day = int(cluster_id)  # Fallback to original
                else:
                    target_day = int(cluster_id)  # Fallback to original
            else:
                target_day = int(cluster_id)  # Use original cluster assignment
            
            # Ensure target day is valid
            if target_day >= len(dates):
                target_day = int(cluster_id)
            
            # Determine target time
            target_time = None
            if scheduled_time:
                target_time = parse_scheduled_time(scheduled_time, start_times[target_day])
            
            # Use target time if specified, otherwise use current time for that day
            if target_time:
                current_time = target_time
            else:
                current_time = day_schedules[target_day]
            
            # Get custom buffer time for this location, or use default
            buffer_minutes = row.get('buffer_minutes', config.get('default_buffer_minutes', 30))
            
            # Calculate end time for this visit
            end_time = current_time + timedelta(hours=visit_hours)
            
            # Check if this visit would exceed max end time for this day
            max_end_time = datetime.strptime(max_end_times[target_day], '%H:%M')
            if end_time > max_end_time:
                # Try to reschedule to another day with available time
                rescheduled = False
                original_target_day = target_day
                
                if DEBUG_MODE:
                    print_debug(f"Location {row['name']} would end at {end_time.strftime('%I:%M %p')} which exceeds max end time {max_end_time.strftime('%I:%M %p')} for day {target_day + 1}")
                
                # Use intelligent rescheduling that considers geographic proximity
                best_day, best_cost = intelligent_reschedule_location(idx, target_day, df, day_schedules, config)
                
                if best_day is not None and best_cost < 1000:
                    # This day has available time and reasonable geographic cost!
                    target_day = best_day
                    current_time = day_schedules[target_day]
                    end_time = current_time + timedelta(hours=visit_hours)
                    rescheduled = True
                    
                    # Update the cluster assignment in the original dataframe
                    df.loc[idx, 'cluster'] = target_day
                    
                    if DEBUG_MODE:
                        print_debug(f"Intelligently rescheduled {row['name']} from Day {original_target_day + 1} to Day {target_day + 1} at {current_time.strftime('%I:%M %p')} (cost: {best_cost:.2f})")
                        print_debug(f"  Updated cluster assignment from {original_target_day} to {target_day}")
                else:
                    # Fallback to original sequential approach if intelligent rescheduling fails
                    for try_day in range(len(dates)):
                        if try_day == target_day:
                            continue  # Skip the original day that didn't work
                        
                        # Calculate what time this would be on the new day
                        try_current_time = day_schedules[try_day]
                        try_end_time = try_current_time + timedelta(hours=visit_hours)
                        try_max_end_time = datetime.strptime(max_end_times[try_day], '%H:%M')
                        
                        if try_end_time <= try_max_end_time:
                            # This day has available time!
                            target_day = try_day
                            current_time = try_current_time
                            end_time = try_end_time
                            rescheduled = True
                            
                            # Update the cluster assignment in the original dataframe
                            df.loc[idx, 'cluster'] = target_day
                            
                            if DEBUG_MODE:
                                print_debug(f"Fallback rescheduled {row['name']} from Day {original_target_day + 1} to Day {target_day + 1} at {current_time.strftime('%I:%M %p')}")
                                print_debug(f"  Updated cluster assignment from {original_target_day} to {target_day}")
                            break
                
                if not rescheduled:
                    if DEBUG_MODE:
                        print_debug(f"✗ Unable to reschedule {row['name']} - no available time on any day")
                    continue
            
            schedule.append({
                'Date': dates[target_day],
                'Time': f"{current_time.strftime('%I:%M %p')} - {end_time.strftime('%I:%M %p')}",
                'Location': row['name'],
                'Address': row['address'],
                'Price/Cost': row['price_info'],
                'Buffer Time': f"{buffer_minutes} minutes",
                'Cluster': f"Day {target_day + 1}",
                'Priority': row.get('Priority', 'Normal'),
                'Notes': row.get('Notes', '')
            })
            
            scheduled_locations.add(idx)
            
            # Update the day schedule time only if we didn't use a specific time
            if not target_time:
                day_schedules[target_day] = end_time + timedelta(minutes=buffer_minutes)
    
    return pd.DataFrame(schedule)

def parse_scheduled_time(time_str, default_start_time):
    """Parse scheduled time string into datetime object."""
    time_str = time_str.strip().lower()
    
    # Handle time ranges
    if time_str == 'morning':
        return datetime.strptime('09:00', '%H:%M')
    elif time_str == 'afternoon':
        return datetime.strptime('13:00', '%H:%M')
    elif time_str == 'evening':
        return datetime.strptime('17:00', '%H:%M')
    
    # Handle specific times
    try:
        # Try parsing different time formats
        for fmt in ['%I:%M %p', '%H:%M', '%I:%M%p', '%I %p']:
            try:
                return datetime.strptime(time_str, fmt)
            except ValueError:
                continue
        
        # If no format worked, return None to use default scheduling
        return None
    except:
        return None

def suggest_visit_times_for_unlocated(unlocated_df, schedule_df, config=None):
    """Suggest visit times for unlocated addresses using OpenAI with ORS context."""
    if config is None:
        config = load_config()
    
    if unlocated_df.empty:
        return ""
    
    # Get time constraints as arrays
    max_end_times = config.get('max_end_times', ['22:00', '22:00'])
    start_times = config.get('start_times', ['09:00', '09:00'])
    dates = config.get('visit_dates', ['July 16, 2024', 'July 17, 2024'])
    visit_hours = config.get('default_visit_hours', 0.75)
    buffer_minutes = config.get('default_buffer_minutes', 15)
    n_clusters = config.get('clustering', {}).get('n_clusters', 2)
    
    # Calculate the actual end time of each day's schedule
    day_end_times = {}
    for day_idx in range(n_clusters):
        # Get all scheduled visits for this day
        day_schedule = schedule_df[schedule_df['Cluster'] == f"Day {day_idx + 1}"]
        
        if len(day_schedule) > 0:
            # Find the latest end time from the schedule
            latest_end_time = None
            for _, visit in day_schedule.iterrows():
                # Parse the time range (e.g., "09:00 AM - 10:00 AM")
                time_range = visit['Time']
                if ' - ' in time_range:
                    end_time_str = time_range.split(' - ')[1]
                    try:
                        # Parse time like "10:00 AM" to datetime
                        end_time = datetime.strptime(end_time_str, '%I:%M %p')
                        if latest_end_time is None or end_time > latest_end_time:
                            latest_end_time = end_time
                    except:
                        pass
            
            # If we found a latest end time, add buffer to get the next available slot
            if latest_end_time:
                day_end_times[day_idx] = latest_end_time + timedelta(minutes=buffer_minutes)
            else:
                # Fallback to start time
                day_end_times[day_idx] = datetime.strptime(start_times[day_idx], '%H:%M')
        else:
            # No scheduled visits for this day, use start time
            day_end_times[day_idx] = datetime.strptime(start_times[day_idx], '%H:%M')
    
    # Try to use OpenAI for intelligent suggestions
    try:
        with open('api_keys.json', 'r') as f:
            api_keys = json.load(f)
            openai_api_key = api_keys.get('openai_api_key')
            if openai_api_key and openai_api_key != "your-openai-api-key-here":
                openai_client = OpenAI(api_key=openai_api_key.strip())
                
                # Build context for OpenAI
                schedule_context = "Current Schedule:\n"
                for day_idx in range(n_clusters):
                    day_schedule = schedule_df[schedule_df['Cluster'] == f"Day {day_idx + 1}"]
                    schedule_context += f"\nDay {day_idx + 1} ({dates[day_idx]}):\n"
                    schedule_context += f"  Start: {start_times[day_idx]}, End: {max_end_times[day_idx]}\n"
                    schedule_context += f"  Next available: {day_end_times[day_idx].strftime('%I:%M %p')}\n"
                    if len(day_schedule) > 0:
                        schedule_context += "  Scheduled visits:\n"
                        for _, visit in day_schedule.iterrows():
                            schedule_context += f"    - {visit['Time']}: {visit['Location']}\n"
                    else:
                        schedule_context += "  No visits scheduled\n"
                
                # Add ORS context if available
                ors_context = ""
                ors_client = load_ors_client(config)
                if ors_client:
                    ors_context = "\nTravel Time Considerations:\n"
                    ors_context += "- Real driving times are calculated using OpenRouteService\n"
                    ors_context += "- Consider traffic patterns and actual road distances\n"
                    ors_context += "- Buffer times should account for real travel times\n"
                else:
                    ors_context = "\nTravel Time Considerations:\n"
                    ors_context += "- Using estimated travel times (ORS not available)\n"
                    ors_context += "- Consider adding buffer time for traffic and road conditions\n"
                
                # Build prompt for unlocated addresses
                unlocated_list = []
                for _, row in unlocated_df.iterrows():
                    location_name = row.get('Apartment', row.get('name', 'Unknown Location'))
                    unlocated_list.append(f"{location_name}: {row['address']}")
                
                prompt = f"""You are a scheduling assistant. Given the current schedule and unlocated addresses, suggest optimal visit times.

{schedule_context}
{ors_context}

Unlocated Addresses:
{chr(10).join(unlocated_list)}

Constraints:
- Each visit takes {visit_hours} hours
- Buffer time between visits: {buffer_minutes} minutes
- Consider real travel times and traffic patterns
- Place addresses at the end of existing schedules when possible
- If a day is full, suggest the next available day

For each unlocated address, respond with ONLY the time in format:
"Day X (Date): HH:MM AM/PM - Location Name"

If impossible to schedule within constraints, respond with:
"Day X (Date): Impossible to schedule within allowed time - Location Name"
"""
                
                response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a scheduling assistant. Provide concise, properly formatted time suggestions."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0.3
                )
                
                ai_suggestions = response.choices[0].message.content.strip()
                if ai_suggestions:
                    return ai_suggestions
                    
    except Exception as e:
        if DEBUG_MODE:
            print_debug(f"OpenAI suggestion failed: {e}")
    
    # Fallback to algorithmic suggestions
    suggestions = []
    
    for _, row in unlocated_df.iterrows():
        location_name = row.get('Apartment', row.get('name', 'Unknown Location'))
        full_label = f"{location_name}: {row['address']}"
        
        # Find the best day for this address
        best_day = None
        best_time = None
        impossible_all_days = True
        
        for day_idx in range(n_clusters):
            start_time = start_times[day_idx]
            max_end_time = max_end_times[day_idx]
            date = dates[day_idx]
            current_time = day_end_times[day_idx]
            
            # Check if we can fit this visit on this day
            visit_end_time = current_time + timedelta(hours=visit_hours)
            max_end_dt = datetime.strptime(max_end_time, '%H:%M')
            
            if visit_end_time <= max_end_dt:
                # This day works!
                best_day = day_idx
                best_time = current_time
                impossible_all_days = False
                break
        
        if impossible_all_days:
            # Try to find any day with some time available, even if it's at the very end
            for day_idx in range(n_clusters):
                max_end_dt = datetime.strptime(max_end_times[day_idx], '%H:%M')
                if day_end_times[day_idx] < max_end_dt:
                    best_day = day_idx
                    best_time = max_end_dt - timedelta(hours=visit_hours)
                    impossible_all_days = False
                    break
        
        if best_day is not None:
            date = dates[best_day]
            if impossible_all_days:
                suggestions.append(f"Day {best_day + 1} ({date}): Impossible to schedule within allowed time - {full_label}")
            else:
                suggestions.append(f"Day {best_day + 1} ({date}): {best_time.strftime('%I:%M %p')} - {full_label}")
        else:
            # No days available at all
            suggestions.append(f"All Days: Impossible to schedule within allowed time - {full_label}")
    
    return "\n".join(suggestions)

def create_side_by_side_schedule_html(schedule_df, df=None):
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
        
        for idx, visit in day_data.iterrows():
            # Find driving info to next location if available
            driving_info = ""
            if df is not None and idx < len(day_data) - 1:
                # Find the corresponding entry in the main DataFrame for the NEXT location
                next_visit = day_data.iloc[list(day_data.index).index(idx) + 1]
                next_match = df[(df['name'] == next_visit['Location']) & 
                               (df['address'] == next_visit['Address'])]
                
                if not next_match.empty:
                    next_row = next_match.iloc[0]
                    travel_time = next_row.get('ors_travel_time_minutes', 0)
                    travel_distance = next_row.get('ors_travel_distance_km', 0)
                    
                    if travel_time > 0 and travel_distance > 0:
                        driving_info = f"""
                        <div class="visit-driving-info">
                            <i class="fas fa-route text-primary"></i>
                            <small class="text-primary">
                                Next: {travel_distance:.1f} km, {travel_time:.0f} min drive
                            </small>
                        </div>
                        """
            
            # Add priority and notes display
            priority_info = ""
            if 'Priority' in visit and visit['Priority'] and visit['Priority'] != 'Normal':
                priority_class = 'text-danger' if visit['Priority'] == 'High' else 'text-warning'
                priority_info = f'<div class="visit-priority"><small class="{priority_class}"><i class="fas fa-star"></i> {visit["Priority"]} Priority</small></div>'
            
            notes_info = ""
            if 'Notes' in visit and visit['Notes'] and str(visit['Notes']).strip() and str(visit['Notes']).lower() != 'nan':
                notes_info = f'<div class="visit-notes"><small class="text-info"><i class="fas fa-sticky-note"></i> {visit["Notes"]}</small></div>'
            
            # Add buffer warning display
            buffer_warning_info = ""
            if 'Buffer_Warning' in visit and visit['Buffer_Warning']:
                buffer_warning_info = f'<div class="visit-buffer-warning"><small class="text-danger"><i class="fas fa-exclamation-triangle"></i> <strong>Warning:</strong> Drive time exceeds buffer time</small></div>'

            day_html += f"""
            <div class="visit-item">
                <div class="visit-time">{visit['Time']}</div>
                <div class="visit-location">{visit['Location']}</div>
                <div class="visit-address">{visit['Address']}</div>
                <div class="visit-price">{visit['Price/Cost']}</div>
                <small class="text-muted">Buffer: {visit['Buffer Time']}</small>
                {priority_info}
                {notes_info}
                {buffer_warning_info}
                {driving_info}
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
    
    # Add markers and routes to the map
    for cluster_id in df['cluster'].unique():
        if pd.isna(cluster_id):
            continue
            
        cluster_data = df[df['cluster'] == cluster_id].copy()
        cluster_data = cluster_data.sort_index()  # Maintain route order
        color = colors[int(cluster_id) % len(colors)]
        route_coordinates = []
        
        # Add markers
        for idx, row in cluster_data.iterrows():
            route_coordinates.append([row['latitude'], row['longitude']])
            
            popup_text = f"""
            <b>{row['name']}</b><br>
            Address: {row['address']}<br>
            Price/Cost: {row['price_info']}<br>
            Day: {int(cluster_id) + 1}
            """
            
            folium.Marker(
                [row['latitude'], row['longitude']],
                popup=folium.Popup(popup_text, max_width=300),
                tooltip=f"{row['name']} (Day {int(cluster_id) + 1})",
                icon=folium.Icon(color=color, icon='home')
            ).add_to(m)
        
        # Add route lines between consecutive locations
        if len(route_coordinates) > 1:
            for i in range(len(cluster_data) - 1):
                current_row = cluster_data.iloc[i]
                next_row = cluster_data.iloc[i + 1]
                route_geometry_json = next_row.get('ors_route_geometry', '')
                
                try:
                    if route_geometry_json:
                        route_geometry = json.loads(route_geometry_json)
                        if route_geometry and 'coordinates' in route_geometry:
                            # Convert ORS coordinates [lon, lat] to folium format [lat, lon]
                            route_coords = [[coord[1], coord[0]] for coord in route_geometry['coordinates']]
                            folium.PolyLine(
                                locations=route_coords,
                                color=color,
                                weight=3,
                                opacity=0.8,
                                popup=f"Route: {current_row['name']} → {next_row['name']}"
                            ).add_to(m)
                        else:
                            # Fallback to straight line
                            folium.PolyLine(
                                locations=[route_coordinates[i], route_coordinates[i + 1]],
                                color=color,
                                weight=2,
                                opacity=0.6,
                                popup=f"Route: {current_row['name']} → {next_row['name']}"
                            ).add_to(m)
                    else:
                        # No route geometry, use straight line
                        folium.PolyLine(
                            locations=[route_coordinates[i], route_coordinates[i + 1]],
                            color=color,
                            weight=2,
                            opacity=0.6,
                            popup=f"Route: {current_row['name']} → {next_row['name']}"
                        ).add_to(m)
                except (json.JSONDecodeError, KeyError, IndexError):
                    # Error parsing route geometry, use straight line
                    folium.PolyLine(
                        locations=[route_coordinates[i], route_coordinates[i + 1]],
                        color=color,
                        weight=2,
                        opacity=0.6,
                        popup=f"Route: {current_row['name']} → {next_row['name']}"
                    ).add_to(m)
    
    # Convert map to HTML
    map_html = m._repr_html_()
    
    # Create schedule HTML with days side by side
    schedule_html = create_side_by_side_schedule_html(schedule_df, df)
    
    # Load and render template
    template_content = load_html_template()
    html_content = render_html_template(
        template_content,
        title="Location Visitation Schedule",
        map_html=map_html,
        schedule_html=schedule_html,
        unlocated_section="",
        warnings=[]  # Default empty warnings for individual day files
    )
    
    # Save to specified output directory
    output_path = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print_info(f"HTML output saved to {output_path}")

def prepare_directions_data(df, schedule_df, config=None):
    """Prepare directions data for HTML template."""
    if config is None:
        config = load_config()
    
    ors_config = config.get('ors', {})
    if not ors_config.get('show_directions', True):
        return {}, ""
    
    # Check if ORS client is available
    ors_client = load_ors_client(config)
    if not ors_client:
        if DEBUG_MODE:
            print_debug("ORS client not available, skipping directions data preparation")
        return {}, ""
    
    directions_data = {}
    day_options = ""
    
    # Process each day
    for day_idx, date in enumerate(schedule_df['Date'].unique()):
        day_name = f"day{day_idx + 1}"
        day_schedule = schedule_df[schedule_df['Date'] == date].reset_index(drop=True)
        day_options += f'<option value="{day_name}">Day {day_idx + 1} ({date})</option>'
        
        if len(day_schedule) < 2:
            directions_data[day_name] = {
                'routes': [],
                'totalDistance': 0,
                'totalTime': 0
            }
            continue
        
        routes = []
        total_distance = 0
        total_time = 0
        
        # Get route information between consecutive locations
        for i in range(len(day_schedule) - 1):
            current_visit = day_schedule.iloc[i]
            next_visit = day_schedule.iloc[i + 1]
            
            # Find the corresponding entries in the main DataFrame
            current_match = df[(df['name'] == current_visit['Location']) & 
                             (df['address'] == current_visit['Address'])]
            next_match = df[(df['name'] == next_visit['Location']) & 
                          (df['address'] == next_visit['Address'])]
            
            if not current_match.empty and not next_match.empty:
                next_row = next_match.iloc[0]
                
                # Get ORS data
                travel_time = next_row.get('ors_travel_time_minutes', 0)
                travel_distance = next_row.get('ors_travel_distance_km', 0)
                directions_json = next_row.get('ors_directions', '[]')
                
                try:
                    directions_list = json.loads(directions_json) if directions_json else []
                except:
                    directions_list = []
                
                route_info = {
                    'from': current_visit['Location'],
                    'to': next_visit['Location'],
                    'distance': f"{travel_distance:.1f}",
                    'duration': f"{travel_time:.0f}",
                    'directions': directions_list
                }
                
                routes.append(route_info)
                total_distance += travel_distance
                total_time += travel_time
        
        directions_data[day_name] = {
            'routes': routes,
            'totalDistance': f"{total_distance:.1f}",
            'totalTime': f"{total_time:.0f}"
        }
    
    return directions_data, day_options

def generate_combined_html_output(df, schedule_df, output_dir, filename='location_schedule_combined.html', unlocated_df=None, unlocated_suggestions=None, warnings=None):
    """Generate combined HTML output with map and schedule side by side."""
    # Create the map
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    
    # Create a map centered on the data
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    
    # Color scheme for clusters
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']

    # For each day, get the route order from schedule_df and add markers and routes
    for day_idx, date in enumerate(schedule_df['Date'].unique()):
        day_schedule = schedule_df[schedule_df['Date'] == date].reset_index(drop=True)
        route_coordinates = []
        
        for order, visit in day_schedule.iterrows():
            match = df[(df['name'] == visit['Location']) & (df['address'] == visit['Address'])]
            if not match.empty:
                row = match.iloc[0]
                cluster_id = row['cluster'] if not pd.isna(row['cluster']) else 0
                color = colors[int(cluster_id) % len(colors)]
                
                # Add coordinates to route
                route_coordinates.append([row['latitude'], row['longitude']])
                
                popup_text = f"""
                <b>{row['name']}</b><br>
                Address: {row['address']}<br>
                Price/Cost: {row['price_info']}<br>
                <b>Visit Time:</b> {visit['Time']}<br>
                Day: {int(cluster_id) + 1 if not pd.isna(cluster_id) else 'N/A'}
                """
                marker = folium.Marker(
                    [row['latitude'], row['longitude']],
                    popup=folium.Popup(popup_text, max_width=300),
                    tooltip=f"{row['name']} (Day {int(cluster_id) + 1 if not pd.isna(cluster_id) else 'N/A'})",
                    icon=folium.Icon(color=color, icon='home')
                )
                marker.add_to(m)
        
        # Add route lines between consecutive locations for this day
        if len(route_coordinates) > 1:
            cluster_id = day_idx  # Use day index as cluster ID for coloring
            route_color = colors[cluster_id % len(colors)]
            
            # Try to use actual route geometry if available, otherwise use straight lines
            for i in range(len(day_schedule) - 1):
                current_visit = day_schedule.iloc[i]
                next_visit = day_schedule.iloc[i + 1]
                
                # Find the corresponding entries in the main DataFrame
                next_match = df[(df['name'] == next_visit['Location']) & (df['address'] == next_visit['Address'])]
                
                if not next_match.empty:
                    next_row = next_match.iloc[0]
                    route_geometry_json = next_row.get('ors_route_geometry', '')
                    
                    try:
                        if route_geometry_json:
                            route_geometry = json.loads(route_geometry_json)
                            if route_geometry and 'coordinates' in route_geometry:
                                # Convert ORS coordinates [lon, lat] to folium format [lat, lon]
                                route_coords = [[coord[1], coord[0]] for coord in route_geometry['coordinates']]
                                folium.PolyLine(
                                    locations=route_coords,
                                    color=route_color,
                                    weight=3,
                                    opacity=0.8,
                                    popup=f"Route: {current_visit['Location']} → {next_visit['Location']}"
                                ).add_to(m)
                            else:
                                # Fallback to straight line
                                folium.PolyLine(
                                    locations=[route_coordinates[i], route_coordinates[i + 1]],
                                    color=route_color,
                                    weight=2,
                                    opacity=0.6,
                                    popup=f"Route: {current_visit['Location']} → {next_visit['Location']}"
                                ).add_to(m)
                        else:
                            # No route geometry, use straight line
                            folium.PolyLine(
                                locations=[route_coordinates[i], route_coordinates[i + 1]],
                                color=route_color,
                                weight=2,
                                opacity=0.6,
                                popup=f"Route: {current_visit['Location']} → {next_visit['Location']}"
                            ).add_to(m)
                    except (json.JSONDecodeError, KeyError, IndexError):
                        # Error parsing route geometry, use straight line
                        folium.PolyLine(
                            locations=[route_coordinates[i], route_coordinates[i + 1]],
                            color=route_color,
                            weight=2,
                            opacity=0.6,
                            popup=f"Route: {current_visit['Location']} → {next_visit['Location']}"
                        ).add_to(m)

    # Convert map to HTML
    map_html = m._repr_html_()

    # Create schedule HTML with days side by side
    schedule_html = create_side_by_side_schedule_html(schedule_df, df)

    # --- Fix unlocated section to show correct name ---
    unlocated_section = ""
    if unlocated_df is not None and len(unlocated_df) > 0:
        # Prefer original name column if present, else fallback to 'name'
        name_col = None
        for col in unlocated_df.columns:
            if col.lower() in ['apartment', 'event', 'location', 'name', 'venue', 'place']:
                name_col = col
                break
        if not name_col:
            name_col = 'name' if 'name' in unlocated_df.columns else unlocated_df.columns[0]
        
        # Use the correct name column - look for apartment name first
        unlocated_list = ""
        for _, row in unlocated_df.iterrows():
            location_name = row.get('Apartment', row.get('name', 'Unknown Location'))
            unlocated_list += f"• {location_name}: {row['address']}<br>"
        
        suggestions_html = f"<br><br><strong>Suggested Times:</strong><br><pre style='white-space: pre-wrap; margin: 0;'>{unlocated_suggestions}</pre>" if unlocated_suggestions else ""
        unlocated_section = f"""
        <div class=\"row mt-3\">
            <div class=\"col-12\">
                <div class=\"alert alert-warning\">
                    <strong>Unlocated Addresses ({len(unlocated_df)}):</strong><br>
                    {unlocated_list}
                    {suggestions_html}
                </div>
            </div>
        </div>
        """

    # Prepare directions data
    directions_data, day_options = prepare_directions_data(df, schedule_df)
    
    # Load and render template
    template_content = load_html_template()
    # Generate warnings section HTML
    warnings_section = ""
    if warnings:
        warnings_html = ""
        for warning in warnings:
            warnings_html += f'<div class="mb-1">{warning}</div>'
        warnings_section = f"""
        <div class="alert alert-warning mb-3">
            <h6><i class="fas fa-exclamation-triangle me-2"></i>Schedule Warnings</h6>
            {warnings_html}
        </div>
        """
    
    html_content = render_html_template(
        template_content,
        title="Location Visitation Schedule - Combined View",
        map_html=map_html,
        schedule_html=schedule_html,
        unlocated_section=unlocated_section,
        directions_data=json.dumps(directions_data),
        day_options=day_options,
        warnings_section=warnings_section
    )
    
    # Save to specified output directory
    output_path = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print_info(f"Combined HTML output saved to {output_path}")
    
    # Auto-open the HTML file
    webbrowser.open('file://' + os.path.abspath(output_path))

def generate_html_maps(df, schedule_df, output_dir, unlocated_suggestions=None, unlocated_df=None, warnings=None):
    """Generate HTML maps for the schedule."""
    # Only generate the combined map now
    generate_combined_html_output(df, schedule_df, output_dir, 'combined_map.html', unlocated_df=unlocated_df, unlocated_suggestions=unlocated_suggestions, warnings=warnings)

def main():
    global DEBUG_MODE
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Location Visitation Scheduler')
    parser.add_argument('csv_file', nargs='?', help='CSV file to process (searches in spreadsheets/ folder first)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()
    
    DEBUG_MODE = args.debug
    
    print_header("Location Visitation Scheduler")
    
    # Create default config and spreadsheets directory on startup
    create_default_config()
    create_spreadsheets_directory()
    
    # Load configuration
    config = load_config()
    
    # Determine which CSV file to use
    if args.csv_file:
        # Check if file exists in spreadsheets folder first
        spreadsheets_path = os.path.join('spreadsheets', args.csv_file)
        if os.path.exists(spreadsheets_path):
            csv_file = spreadsheets_path
        elif os.path.exists(args.csv_file):
            csv_file = args.csv_file
        else:
            print_error(f"CSV file not found: {args.csv_file}")
            print_info("Available CSV files in /spreadsheets/ folder:")
            spreadsheets_dir = 'spreadsheets'
            if os.path.exists(spreadsheets_dir):
                for file in os.listdir(spreadsheets_dir):
                    if file.lower().endswith('.csv'):
                        print(f"  {file}")
            return
    else:
        # No file specified, use default
        csv_file = 'Updated_Madison_Area_Apartments.csv'
        if not os.path.exists(csv_file):
            # Try in spreadsheets folder
            spreadsheets_path = os.path.join('spreadsheets', csv_file)
            if os.path.exists(spreadsheets_path):
                csv_file = spreadsheets_path
            else:
                print_error(f"Default CSV file not found: {csv_file}")
                print_info("Available CSV files in /spreadsheets/ folder:")
                spreadsheets_dir = 'spreadsheets'
                if os.path.exists(spreadsheets_dir):
                    for file in os.listdir(spreadsheets_dir):
                        if file.lower().endswith('.csv'):
                            print(f"  {file}")
                return
    
    print_info(f"Processing data from: {csv_file}")
    
    # Step 1: Read CSV
    original_df = read_csv(csv_file)
    
    # Step 2: Geocode addresses
    df = geocode_addresses(original_df, config=config)

    # Step 3: Identify unlocated addresses correctly
    # The issue is that we need to compare the original data with what was successfully geocoded
    # Find the name and address columns in the original DataFrame
    orig_name_col = None
    orig_address_col = None
    for col in original_df.columns:
        col_lower = col.lower()
        if not orig_name_col and any(keyword in col_lower for keyword in ['apartment', 'event', 'location', 'name', 'venue', 'place']):
            orig_name_col = col
        if not orig_address_col and 'address' in col_lower:
            orig_address_col = col
    
    if DEBUG_MODE:
        print_debug(f"Original name column: {orig_name_col}")
        print_debug(f"Original address column: {orig_address_col}")
        print_debug(f"Original DataFrame length: {len(original_df)}")
        print_debug(f"Geocoded DataFrame length: {len(df)}")
    
    # Create a set of successfully geocoded original addresses
    # Use the original_address field if available, otherwise fall back to address
    geocoded_original_addresses = set()
    if len(df) > 0:
        for _, row in df.iterrows():
            original_addr = row.get('original_address', row['address'])
            geocoded_original_addresses.add(original_addr)
    
    # Find addresses in original DataFrame that are NOT in the geocoded set
    unlocated_mask = ~original_df[orig_address_col].isin(geocoded_original_addresses)
    unlocated_df = original_df[unlocated_mask].copy()
    
    if DEBUG_MODE:
        print_debug(f"Geocoded original addresses: {list(geocoded_original_addresses)}")
        print_debug(f"Unlocated addresses: {list(unlocated_df[orig_address_col]) if len(unlocated_df) > 0 else []}")
        print_debug(f"Unlocated DataFrame length: {len(unlocated_df)}")
    
    # Ensure unlocated_df has the correct column structure for later use
    if len(unlocated_df) > 0:
        # Rename columns to match the processed DataFrame structure
        if orig_name_col and orig_name_col != 'name':
            unlocated_df = unlocated_df.rename(columns={orig_name_col: 'name'})
        if orig_address_col and orig_address_col != 'address':
            unlocated_df = unlocated_df.rename(columns={orig_address_col: 'address'})
        
        # Add price_info column if it doesn't exist
        if 'price_info' not in unlocated_df.columns:
            # Find price column in original data
            price_col = None
            for col in unlocated_df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['price', 'cost', 'fee', 'rate']):
                    price_col = col
                    break
            if price_col:
                unlocated_df = unlocated_df.rename(columns={price_col: 'price_info'})
            else:
                unlocated_df['price_info'] = 'N/A'
        
        # Instead of round-robin assignment, we'll let the suggest_visit_times_for_unlocated function
        # determine the best day for each unlocated address based on available time slots
        # For now, assign all to day 0 (first day) and let the function redistribute them
        unlocated_df['cluster'] = 0

    # Step 4: Cluster locations (only for located addresses)
    df = cluster_locations(df, n_clusters=config.get('clustering', {}).get('n_clusters', 2))
    
    # Step 4.5: Optimize clusters with time constraints
    df = optimize_clusters_with_time_constraints(df, config=config)
    
    # Step 5: Optimize routes
    df = optimize_routes(df)
    
    # Step 5.5: Calculate ORS travel times
    df = calculate_ors_travel_times(df, config=config)
    
    # Step 6: Create schedule
    schedule_df = create_schedule(df, config=config)
    
    # Step 6.5: Validate schedule with ORS
    schedule_df, ors_warnings = validate_schedule_with_ors(schedule_df, df, config=config)
    if ors_warnings:
        print_warning("Schedule validation warnings:")
        for warning in ors_warnings:
            print_warning(f"  {warning}")
    
    # Step 7: Generate time suggestions for unlocated addresses
    unlocated_suggestions = None
    if len(unlocated_df) > 0:
        print_info(f"Generating time suggestions for {len(unlocated_df)} unlocated addresses...")
        unlocated_suggestions = suggest_visit_times_for_unlocated(unlocated_df, schedule_df, config)
        print_info("Time suggestions generated for unlocated addresses.")
    
    # Step 8: Create output directory using trip name (from CSV filename, no extension)
    trip_name = os.path.splitext(os.path.basename(csv_file))[0]
    trip_output_dir = create_trip_output_directory(trip_name)
    
    # Step 9: Generate HTML maps for the schedule
    generate_html_maps(df, schedule_df, trip_output_dir, unlocated_suggestions, unlocated_df, ors_warnings)
    
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