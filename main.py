import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
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

def create_default_config():
    """Create default config.json if it doesn't exist."""
    config_file = 'config.json'
    if not os.path.exists(config_file):
        default_config = {
            "default_buffer_minutes": 30,
            "default_visit_hours": 1,
            "start_time": "09:00",
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
        print(f"Created default config file: {config_file}")

def create_spreadsheets_directory():
    """Create spreadsheets directory if it doesn't exist."""
    spreadsheets_dir = 'spreadsheets'
    if not os.path.exists(spreadsheets_dir):
        os.makedirs(spreadsheets_dir)
        print(f"Created spreadsheets directory: {spreadsheets_dir}")

def load_config():
    """Load configuration from config.json."""
    config_file = 'config.json'
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return json.load(f)
    else:
        print("Config file not found, using default values")
        return {
            "default_buffer_minutes": 30,
            "default_visit_hours": 1,
            "start_time": "09:00",
            "visit_dates": ["July 16, 2024", "July 17, 2024"]
        }

# 1. Read CSV file
# 2. Geocode addresses
# 3. Cluster apartments into two days
# 4. Optimize route within each cluster
# 5. Create schedule for each day
# 6. Generate HTML map
# 7. Output schedule

def read_csv(filename):
    """Read the CSV file and return a DataFrame."""
    try:
        df = pd.read_csv(filename)
        print(f"Successfully read {len(df)} apartments from {filename}")
        print(f"Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

def geocode_addresses(df, cache_file='geocode_cache.json', config=None):
    """Geocode addresses using Nominatim with persistent cache and retry with exponential backoff."""
    if config is None:
        config = load_config()
    
    geolocator = Nominatim(user_agent="apartment_scheduler")
    
    # Verify column names exist in the DataFrame
    print(f"Available columns: {list(df.columns)}")
    
    # Determine the correct column names
    apartment_col = None
    address_col = None
    price_col = None
    time_col = None
    
    # Check for apartment name column
    for col in df.columns:
        if 'apartment' in col.lower() or 'name' in col.lower():
            apartment_col = col
            break
    
    # Check for address column
    for col in df.columns:
        if 'address' in col.lower():
            address_col = col
            break
    
    # Check for price column
    for col in df.columns:
        if 'price' in col.lower():
            price_col = col
            break
    
    # Check for time/buffer column
    for col in df.columns:
        if 'time' in col.lower() or 'buffer' in col.lower():
            time_col = col
            break
    
    if not apartment_col or not address_col:
        print("Error: Could not find required columns. Available columns:", list(df.columns))
        print("Please ensure your CSV has columns for apartment name and address.")
        return pd.DataFrame()
    
    print(f"Using columns: {apartment_col}, {address_col}, {price_col if price_col else 'N/A'}, {time_col if time_col else 'N/A'}")
    
    # Load OpenAI API key
    try:
        with open('api_keys.json', 'r') as f:
            api_keys = json.load(f)
            openai_client = OpenAI(api_key=api_keys.get('openai_api_key'))
    except FileNotFoundError:
        print("Warning: api_keys.json not found. OpenAI reformatting will be disabled.")
        openai_client = None
    except KeyError:
        print("Warning: openai_api_key not found in api_keys.json. OpenAI reformatting will be disabled.")
        openai_client = None
    
    # Load cache if exists
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache = json.load(f)
    else:
        cache = {}
    
    print("Geocoding addresses with persistent cache and retry...")
    
    # Process each row and store results
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        address = row[address_col]
        apartment_name = row[apartment_col]
        price_range = row[price_col] if price_col else "N/A"
        
        # Get custom buffer time if available, otherwise use default
        try:
            if time_col and pd.notna(row[time_col]):
                buffer_minutes = int(row[time_col])
            else:
                buffer_minutes = config.get('default_buffer_minutes', 30)
        except (ValueError, TypeError):
            buffer_minutes = config.get('default_buffer_minutes', 30)
        
        if address in cache:
            lat, lon = cache[address]
            if lat is not None and lon is not None:  # Skip cached failures
                results.append({
                    'index': idx,
                    'apartment': apartment_name,
                    'address': address,
                    'latitude': lat,
                    'longitude': lon,
                    'price_range': price_range,
                    'buffer_minutes': buffer_minutes
                })
                print(f"Using cached result for: {address}")
                continue
        
        success = False
        retries = 0
        max_wait = config.get('geocoding', {}).get('max_wait_seconds', 300)
        max_retries = config.get('geocoding', {}).get('max_retries', 3)
        original_address = address
        tried_addresses = set()  # Track addresses we've already tried
        
        while not success and retries < max_retries:
            try:
                # Skip if we've already tried this exact address
                if address in tried_addresses:
                    print(f"Already tried address: {address}, skipping...")
                    break
                
                tried_addresses.add(address)
                print(f"Attempting to geocode: {address} (attempt {retries + 1})")
                result = geolocator.geocode(address)
                
                print(f"Geocoder returned: {result} (type: {type(result)})")
                
                if isinstance(result, types.CoroutineType):
                    print(f"ERROR: geocode returned a coroutine for address: {address}. This shouldn't happen!")
                    wait_time = min(2 ** retries, max_wait)
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    retries += 1
                elif result is not None:
                    lat, lon = result.latitude, result.longitude
                    print(f"SUCCESS: Found coordinates for {address}: ({lat}, {lon})")
                    results.append({
                        'index': idx,
                        'apartment': apartment_name,
                        'address': address,
                        'latitude': lat,
                        'longitude': lon,
                        'price_range': price_range,
                        'buffer_minutes': buffer_minutes
                    })
                    cache[address] = (lat, lon)
                    # Save cache after each new geocode
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        json.dump(cache, f)
                    success = True
                else:
                    print(f"FAILURE: Geocoder returned None for address: {address}")
                    print(f"This usually means the address couldn't be found or is invalid")
                    
                    # Try OpenAI reformatting only once if we haven't tried it yet
                    if retries == 0 and openai_client:
                        print(f"Trying OpenAI to reformat address: {address}")
                        try:
                            reformatted_address = reformat_address_with_openai(address, openai_client)
                            if reformatted_address and reformatted_address != address and reformatted_address not in tried_addresses:
                                print(f"OpenAI suggested: {reformatted_address}")
                                address = reformatted_address
                                retries = 0  # Reset retries for the reformatted address
                                continue
                            else:
                                print(f"OpenAI suggestion already tried or same as original, skipping...")
                        except Exception as e:
                            print(f"OpenAI reformatting failed: {e}")
                    
                    # If we get here, the address couldn't be found even after reformatting
                    print(f"Address not found in database, skipping: {address}")
                    break  # Don't retry None responses
                
                # Rate limiting
                if success:
                    time.sleep(config.get('geocoding', {}).get('rate_limit_delay', 1))
                    
            except Exception as e:
                print(f"EXCEPTION while geocoding {address}: {type(e).__name__}: {str(e)}")
                if '429' in str(e):
                    print(f"429 Too Many Requests for {address}. Waiting 60 seconds before retrying...")
                    time.sleep(60)
                else:
                    wait_time = min(2 ** retries, max_wait)
                    print(f"Retrying in {wait_time} seconds... (attempt {retries + 1}/{max_retries})")
                    time.sleep(wait_time)
                retries += 1
        
        if not success:
            print(f"SKIPPING: Failed to geocode {original_address} after {max_retries} attempts!")
            print(f"Address appears to be invalid or not found in the database")
            # Don't add to results - this will be filtered out later
            cache[original_address] = (None, None)  # Cache the failure
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache, f)
    
    # Create new DataFrame with only successful geocodes
    if results:
        new_df = pd.DataFrame(results)
        new_df = new_df.set_index('index')
        print(f"Successfully processed {len(new_df)} addresses (skipped {len(df) - len(new_df)} invalid addresses)")
        return new_df
    else:
        print("No addresses could be geocoded!")
        return pd.DataFrame()

def reformat_address_with_openai(address, client):
    """Use OpenAI to reformat an address for better geocoding."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that reformats addresses to be more geocoding-friendly. Return only the reformatted address, nothing else."},
                {"role": "user", "content": f"Please reformat this address to be more geocoding-friendly: {address}. Remove unit numbers, simplify complex formatting, and make it more standard."}
            ],
            max_tokens=300,
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return None

def cluster_apartments(df, n_clusters=2):
    """Cluster apartments into n_clusters groups using K-means."""
    coords = df[['latitude', 'longitude']].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(coords)
    return df

def calculate_distance_matrix(coords):
    """Calculate distance matrix between all coordinates."""
    n = len(coords)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                distances[i][j] = geodesic(coords[i], coords[j]).miles
    return distances

def nearest_neighbor_route(coords):
    """Find a route using nearest neighbor algorithm."""
    n = len(coords)
    unvisited = list(range(n))
    route = [unvisited.pop(0)]  # Start with first apartment
    
    while unvisited:
        current = route[-1]
        nearest = min(unvisited, key=lambda x: geodesic(coords[current], coords[x]).miles)
        route.append(nearest)
        unvisited.remove(nearest)
    
    return route

def optimize_routes(df):
    """Optimize routes within each cluster."""
    optimized_df = df.copy()
    
    for cluster_id in df['cluster'].unique():
        cluster_data = df[df['cluster'] == cluster_id].copy()
        coords = cluster_data[['latitude', 'longitude']].values
        
        if len(coords) > 1:
            route = nearest_neighbor_route(coords)
            # Reorder the cluster data according to the optimized route
            cluster_data = cluster_data.iloc[route].reset_index(drop=True)
        
        # Add route order to the dataframe
        cluster_data['route_order'] = range(len(cluster_data))
        
        # Update the main dataframe with route orders
        for idx, row in cluster_data.iterrows():
            # Find the corresponding row in the main dataframe
            mask = (optimized_df['apartment'] == row['apartment']) & (optimized_df['address'] == row['address'])
            optimized_df.loc[mask, 'route_order'] = row['route_order']
    
    return optimized_df.sort_values(['cluster', 'route_order']).reset_index(drop=True)

def create_schedule(df, config=None):
    """Create a schedule for the two days."""
    if config is None:
        config = load_config()
    
    schedule = []
    
    # Define visit dates from config
    dates = config.get('visit_dates', ['July 16, 2024', 'July 17, 2024'])
    start_time = config.get('start_time', '09:00')
    visit_hours = config.get('default_visit_hours', 1)
    
    for cluster_id in df['cluster'].unique():
        cluster_data = df[df['cluster'] == cluster_id].copy()
        date = dates[cluster_id]
        
        # Start time for each day
        current_time = datetime.strptime(start_time, '%H:%M')
        
        for idx, row in cluster_data.iterrows():
            # Get custom buffer time for this apartment, or use default
            buffer_minutes = row.get('buffer_minutes', config.get('default_buffer_minutes', 30))
            
            # Add visit duration for the visit
            end_time = current_time + timedelta(hours=visit_hours)
            
            schedule.append({
                'Date': date,
                'Time': f"{current_time.strftime('%I:%M %p')} - {end_time.strftime('%I:%M %p')}",
                'Apartment': row['apartment'],
                'Address': row['address'],
                'Price Range': row['price_range'],
                'Buffer Time': f"{buffer_minutes} minutes",
                'Cluster': f"Day {cluster_id + 1}"
            })
            
            # Add buffer time + move to next apartment
            current_time = end_time + timedelta(minutes=buffer_minutes)
    
    return pd.DataFrame(schedule)

def generate_html_output(df, schedule_df, filename='apartment_schedule.html'):
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
        color = colors[cluster_id]
        
        popup_text = f"""
        <b>{row['Apartment']}</b><br>
        Address: {row['Address']}<br>
        Price: {row['Price Range']}<br>
        Day: {cluster_id + 1}
        """
        
        folium.Marker(
            [row['latitude'], row['longitude']],
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=f"{row['Apartment']} (Day {cluster_id + 1})",
            icon=folium.Icon(color=color, icon='home')
        ).add_to(m)
    
    # Convert map to HTML
    map_html = m._repr_html_()
    
    # Create schedule HTML
    schedule_html = schedule_df.to_html(classes='table table-striped table-bordered', index=False)
    
    # Combine into final HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Apartment Visitation Schedule</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            .container-fluid {{
                padding: 20px;
            }}
            .map-container {{
                height: 600px;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
            .schedule-container {{
                max-height: 600px;
                overflow-y: auto;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 15px;
            }}
            .table {{
                font-size: 14px;
            }}
            .table th {{
                background-color: #f8f9fa;
                position: sticky;
                top: 0;
                z-index: 1;
            }}
        </style>
    </head>
    <body>
        <div class="container-fluid">
            <h1 class="text-center mb-4">Apartment Visitation Schedule</h1>
            <div class="row">
                <div class="col-md-6">
                    <h3>Interactive Map</h3>
                    <div class="map-container">
                        {map_html}
                    </div>
                </div>
                <div class="col-md-6">
                    <h3>Visitation Schedule</h3>
                    <div class="schedule-container">
                        {schedule_html}
                    </div>
                </div>
            </div>
            <div class="row mt-3">
                <div class="col-12">
                    <div class="alert alert-info">
                        <strong>Legend:</strong> Red markers = Day 1 (July 16), Blue markers = Day 2 (July 17)
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML output saved to {filename}")

def main():
    # Check for spreadsheets folder and list available CSV files
    spreadsheets_dir = 'spreadsheets'
    csv_files = []
    
    if os.path.exists(spreadsheets_dir) and os.path.isdir(spreadsheets_dir):
        # Get all CSV files in the spreadsheets directory
        for file in os.listdir(spreadsheets_dir):
            if file.lower().endswith('.csv'):
                csv_files.append(os.path.join(spreadsheets_dir, file))
    
    if csv_files:
        print("Available CSV files in /spreadsheets/ folder:")
        for i, file_path in enumerate(csv_files, 1):
            file_name = os.path.basename(file_path)
            print(f"{i}. {file_name}")
        
        while True:
            try:
                choice = input(f"\nSelect a file (1-{len(csv_files)}) or press Enter for default: ").strip()
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
                        print(f"Please enter a number between 1 and {len(csv_files)}")
            except ValueError:
                print("Please enter a valid number")
    else:
        # No CSV files found in spreadsheets folder, use default
        if len(sys.argv) > 1:
            csv_file = sys.argv[1]
        else:
            csv_file = 'Updated_Madison_Area_Apartments.csv'
    
    print(f"Processing apartment data from: {csv_file}")
    
    # Step 1: Read CSV
    df = read_csv(csv_file)
    
    # Step 2: Geocode addresses
    df = geocode_addresses(df)
    
    # Step 3: Cluster apartments
    df = cluster_apartments(df, n_clusters=2)
    
    # Step 4: Optimize routes
    df = optimize_routes(df)
    
    # Step 5: Create schedule
    schedule_df = create_schedule(df)
    
    # Step 6: Generate HTML output
    generate_html_output(df, schedule_df)
    
    # Step 7: Also save schedule as CSV
    schedule_df.to_csv('apartment_schedule.csv', index=False)
    print("Schedule saved to apartment_schedule.csv")
    
    print("\nSummary:")
    print(f"Total apartments processed: {len(df)}")
    print(f"Day 1 (July 16): {len(df[df['cluster'] == 0])} apartments")
    print(f"Day 2 (July 17): {len(df[df['cluster'] == 1])} apartments")
    print("\nOpen 'apartment_schedule.html' in your browser to view the map and schedule!")

if __name__ == "__main__":
    main() 