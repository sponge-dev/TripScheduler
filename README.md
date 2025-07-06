# Location Visitation Scheduler

A Python tool that takes a CSV file of locations, geocodes addresses, clusters them into multiple days, optimizes routes, and generates interactive HTML maps with schedules. Features real driving times, turn-by-turn directions, and displays distance/time to the next location in your schedule.

## Quick Start

### Web Interface (Recommended)
1. **Install dependencies**: `pip install -r requirements.txt`
2. **Start the web interface**: `python start_web.py`
3. **Open your browser** to: http://localhost:5000
4. **Upload your CSV file** with location data
5. **Click "Generate Report"** and wait for processing
6. **Click "Open File"** to view your interactive schedule in the browser!

### Command Line Interface
1. **Install dependencies**: `pip install -r requirements.txt`
2. **Run with a specific CSV file**: `python main.py your_file.csv`
3. **Or use the default file**: `python main.py`
4. **Add debug output**: `python main.py your_file.csv --debug`

## Features

### 🗺️ **Interactive Scheduling**
- **Web interface**: Easy-to-use browser interface with file upload
- **Multi-day scheduling**: Automatically splits locations across multiple days
- **Route optimization**: Finds efficient routes between locations using nearest neighbor algorithm
- **Schedule display**: Shows visit times, locations, addresses, and prices in an organized layout

### 🚗 **Real Driving Information**
- **Real driving times**: Uses OpenRouteService for accurate travel times between locations
- **Distance display**: Shows exact driving distance to the next location in your schedule
- **Route visualization**: Interactive maps display actual driving routes, not straight lines
- **Turn-by-turn directions**: Detailed driving directions for each route segment
- **Schedule validation**: Ensures routes fit within time constraints using real travel times

### 🎯 **Smart Features**
- **Driving info in schedule**: Each location shows "Next: X.X km, XX min drive" to the next stop
- **Interactive maps**: Clickable markers with visit times, addresses, and route information
- **Smart geocoding**: Multiple services with AI-powered address fixing for difficult addresses
- **Flexible configuration**: Customizable start times, visit durations, and buffer times per day

### 📊 **Results Management**
- **Retroactive viewing**: Access ALL previously generated schedules from the Results page
- **Historical analysis**: View metadata for each result (location count, days, timestamps)
- **Multiple formats**: Both interactive HTML schedules and downloadable CSV data
- **One-click access**: "Open File" buttons open HTML schedules directly in your browser

## CSV File Format

Your CSV file should have columns for:
- **Location name** (apartment, event, location, name, venue, place)
- **Address** (full street address)
- **Price information** (optional - rent, cost, fee, rate)
- **Buffer time** (optional - custom wait time in minutes)

Place your CSV file in the `spreadsheets/` directory or upload through the web interface.

## Command Line Usage

For advanced users:
```bash
# Process default file
python main.py

# Process specific file (checks spreadsheets/ folder first)
python main.py your_file.csv

# Enable detailed debug output
python main.py your_file.csv --debug
```

## Configuration

Edit `config.json` or use the web interface Settings page to customize:

```json
{
  "start_times": ["12:00", "09:00"],
  "max_end_times": ["22:00", "21:00"],
  "visit_dates": ["July 16, 2024", "July 17, 2024"],
  "default_buffer_minutes": 15,
  "default_visit_hours": 0.75,
  "clustering": {
    "n_clusters": 2
  },
  "ors": {
    "enabled": true,
    "profile": "driving-car",
    "use_for_validation": true,
    "show_directions": true
  }
}
```

### Configuration Options:
- **start_times**: When to start each day (array for multiple days)
- **max_end_times**: Latest allowed end time for each day
- **visit_dates**: Actual dates for scheduling
- **default_visit_hours**: How long to spend at each location
- **default_buffer_minutes**: Travel time between locations
- **clustering.n_clusters**: Number of days to split locations across
- **ors.profile**: Transportation mode (driving-car, cycling-regular, foot-walking)

## API Keys (Optional)

The tool works without any API keys, but you can add them for enhanced features. Create an `api_keys.json` file:

```json
{
  "openai_api_key": "your-openai-api-key-here",
  "ors_api_key": "your-openrouteservice-api-key-here"
}
```

### API Benefits:
- **OpenAI** (optional): 
  - Fixes problematic addresses automatically
  - Suggests optimal visit times for unlocated addresses
  - Considers real travel times in scheduling recommendations

- **OpenRouteService** (optional): 
  - Provides real driving times and distances
  - Turn-by-turn directions for each route
  - Route visualization on maps
  - Schedule validation with actual travel times

Get your free ORS API key at [OpenRouteService](https://openrouteservice.org/).

**Without API keys**: The tool uses estimated travel times and basic geocoding.

## Output Files

The tool creates an `output/(tripname)/` directory with:

### 📱 **Interactive HTML Files**
- **`combined_map.html`**: Complete schedule with all days, interactive map, and driving directions
- **`location_schedule_day_X.html`**: Individual day schedules with maps and route details
- **Features in HTML files**:
  - Interactive maps with clickable markers
  - Driving distance and time to next location
  - Turn-by-turn directions dropdown
  - Route visualization with actual driving paths
  - Schedule validation warnings

### 📊 **Data Files**
- **`complete_location_schedule.csv`**: Full schedule data for all days
- **`location_schedule_day_X.csv`**: Individual day schedule data
- **Includes**: Visit times, locations, addresses, prices, buffer times

### 🗂️ **Results Page Features**
- **View all historical schedules**: Access any previously generated schedule
- **Rich metadata**: See location count, days, creation dates for each result
- **One-click access**: Open HTML files directly or download CSV data
- **Latest highlighting**: Most recent results are clearly marked

## How It Works

1. **Upload/Select**: Choose your CSV file with location data
2. **Geocoding**: Converts addresses to coordinates using multiple services
3. **Clustering**: Groups locations into optimal daily routes
4. **Route Optimization**: Finds efficient visiting order using nearest neighbor
5. **Real Travel Times**: Calculates actual driving times and distances (if ORS enabled)
6. **Schedule Creation**: Generates visit times considering travel and buffer time
7. **Validation**: Checks if schedule fits within time constraints
8. **Output Generation**: Creates interactive HTML maps and CSV schedules

## Troubleshooting

- **Web interface won't start**: Check if port 5000 is available, try `python start_web.py`
- **Geocoding failures**: Check internet connection, try debug mode, or verify address format
- **Large datasets**: Split into smaller files (50-100 locations) for better performance
- **Missing driving times**: Verify ORS API key in `api_keys.json` or check internet connection
- **Schedule validation warnings**: Increase buffer times or adjust max end times in settings

## Contributing

Issues and pull requests are welcome! This tool is designed to be flexible and user-friendly for real-world scheduling needs. 