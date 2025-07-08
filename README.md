# Location Visitation Scheduler

A Python tool that takes a CSV file of locations, geocodes addresses, clusters them into multiple days, optimizes routes, and generates interactive HTML maps with schedules. Features real driving times, turn-by-turn directions, and intelligent geographic proximity-based rescheduling for optimal route planning.

## Quick Start

### Web Interface (Recommended)
1. Install dependencies: `pip install -r requirements.txt`
2. Start the web interface: `python start_web.py`
3. Open your browser to: http://localhost:5000
4. Upload your CSV file with location data
5. Click "Generate Report" and wait for processing
6. Click "Open File" to view your interactive schedule

### Command Line Interface
1. Install dependencies: `pip install -r requirements.txt`
2. Run with a specific CSV file: `python main.py your_file.csv`
3. Or use the default file: `python main.py`
4. Add debug output: `python main.py your_file.csv --debug`

## Features

### Interactive Scheduling
- Web interface with file upload
- Multi-day scheduling that automatically splits locations
- Route optimization using nearest neighbor algorithm
- Schedule display with visit times, locations, addresses, and prices

### Real Driving Information
- Real driving times using OpenRouteService
- Exact driving distance to the next location
- Interactive maps with actual driving routes
- Turn-by-turn directions for each route segment
- Schedule validation with real travel times

### Intelligent Scheduling Algorithm
- Geographic proximity-based rescheduling: When locations don't fit in their assigned day, intelligently moves them to the best alternative day based on geographic proximity
- Cost-benefit analysis: Calculates relocation costs considering travel time impact, clustering efficiency, and schedule disruption
- Cluster optimization: Re-optimizes geographic clusters after initial scheduling to minimize travel times
- Balanced workload: Distributes locations across days to prevent overloading while maintaining geographic efficiency

### Additional Features
- Smart geocoding with multiple services and AI-powered address fixing
- Flexible configuration for start times, visit durations, and buffer times
- Historical results viewing with access to all previously generated schedules
- Multiple output formats: interactive HTML schedules and downloadable CSV data

## CSV File Format

Your CSV file should have columns for:
- Location name (apartment, event, location, name, venue, place)
- Address (full street address)
- Price information (optional - rent, cost, fee, rate)
- Buffer time (optional - custom travel time in minutes between locations)

Place your CSV file in the `spreadsheets/` directory or upload through the web interface.

## Configuration

Edit `config.json` or use the web interface Settings page:

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

## API Keys (Optional)

The tool works without any API keys, but you can add them for enhanced features. Create an `api_keys.json` file:

```json
{
  "openai_api_key": "your-openai-api-key-here",
  "ors_api_key": "your-openrouteservice-api-key-here"
}
```

### API Benefits:
- **OpenAI** (optional): Fixes problematic addresses automatically and suggests optimal visit times
- **OpenRouteService** (optional): Provides real driving times, distances, turn-by-turn directions, and route visualization

Get your free ORS API key at [OpenRouteService](https://openrouteservice.org/).

## Output Files

The tool creates an `output/(tripname)/` directory with:

### Interactive HTML Files
- `combined_map.html`: Complete schedule with all days, interactive map, and driving directions
- `location_schedule_day_X.html`: Individual day schedules with maps and route details

### Data Files
- `complete_location_schedule.csv`: Full schedule data for all days
- `location_schedule_day_X.csv`: Individual day schedule data

## How It Works

1. Upload/Select your CSV file with location data
2. Geocoding converts addresses to coordinates using multiple services
3. Initial clustering groups locations into daily routes using K-means clustering
4. Intelligent optimization analyzes cluster balance and uses geographic proximity to reassign locations
5. Route optimization finds efficient visiting order using nearest neighbor algorithm
6. Real travel times are calculated using actual driving times and distances
7. Schedule creation generates visit times considering travel and buffer time
8. Validation checks if schedule fits within time constraints
9. Output generation creates interactive HTML maps and CSV schedules

## Recent Improvements

### Enhanced Scheduling Algorithm
- Geographic proximity-based rescheduling: Locations that don't fit in their assigned day are moved to the best alternative day based on geographic cost analysis
- Cost-benefit analysis when moving locations considers travel time impact, clustering efficiency, and schedule disruption
- Cluster optimization after initial scheduling minimizes total travel time
- Balanced workload distribution prevents overloading of individual days

### Windows Compatibility
- Fixed console encoding issues by replacing Unicode symbols with ASCII equivalents
- Fixed color assignment errors in clustering visualization
- Improved debug output formatting for Windows Command Prompt and PowerShell

### Web Interface Improvements
- Fixed JSON parsing errors when CSV files contain empty cells
- Fixed file download functionality in the Results page
- Improved error handling and graceful handling of edge cases

## Troubleshooting

- **Web interface won't start**: Check if port 5000 is available
- **Geocoding failures**: Check internet connection, try debug mode, or verify address format
- **Large datasets**: Split into smaller files (50-100 locations) for better performance
- **Missing driving times**: Verify ORS API key in `api_keys.json` or check internet connection
- **Schedule validation warnings**: Increase buffer times or adjust max end times in settings
- **Unicode display issues on Windows**: The tool now uses ASCII symbols for console output
- **CSV preview errors**: The tool now properly handles empty cells in CSV files

## Contributing

Issues and pull requests are welcome! This tool is designed to be flexible and user-friendly for real-world scheduling needs. 