# Location Visitation Scheduler

A Python program that processes CSV files of location addresses, geocodes them, clusters visits into multiple days, optimizes routes, and generates interactive HTML maps with schedules.

## Features

- **Smart CSV Processing**: Automatically detects CSV files in `/spreadsheets/` folder
- **Multi-Service Geocoding**: Uses Nominatim with Photon API fallback for better success rates
- **OpenAI Integration**: Optional address reformatting for better geocoding success
- **Route Optimization**: Clusters locations into days and optimizes visit order
- **Flexible Scheduling**: Supports any number of days (2, 3, 5, etc.)
- **Interactive Output**: HTML map and schedule side by side with Bootstrap styling
- **Template System**: Customizable HTML templates in `/templates/` directory
- **Unlocated Handling**: Tracks addresses that couldn't be geocoded with suggested times
- **Auto-organization**: Groups output files by trip name in `/output/` directories
- **Colorful Terminal**: Colored output with loading animations and debug mode

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Optional setup:**
   - Create `spreadsheets/` folder for CSV files
   - Create `api_keys.json` for OpenAI integration
   - Customize `config.json` for scheduling preferences

3. **Run the program:**
   ```bash
   python main.py                    # Interactive CSV selection
   python main.py your_file.csv      # Direct file specification
   python main.py --debug            # Detailed output
   ```

## CSV Format

Your CSV needs these columns:
- **Name**: `Apartment`, `Event`, `Location`, `Name`, `Venue`, or `Place`
- **Address**: `Address`
- **Price/Cost** (optional): `Price`, `Cost`, `Fee`, or `Rate`
- **Buffer Time** (optional): `Time`, `Buffer`, `Duration`, or `Wait` (minutes)

**Example:**
```csv
Apartment,Address,Price Range,Buffer Time
Greenbriar Village,238 Randolph Dr,Madison,WI 53717,$1,240 - $1,895,45
Park Village,2305 S Park St,Madison,WI 53713,$1,130 - $1,640,30
```

## Configuration

Edit `config.json` to customize:
```json
{
  "default_buffer_minutes": 30,
  "default_visit_hours": 1,
  "start_times": ["09:00", "09:00"],
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
```

### Multiple Days

The script supports any number of days. Simply update the configuration:

**Example for 3 days:**
```json
{
  "clustering": {
    "n_clusters": 3
  },
  "start_times": ["09:00", "10:00", "11:00"],
  "visit_dates": ["July 16, 2024", "July 17, 2024", "July 18, 2024"]
}
```

**Example for 5 days:**
```json
{
  "clustering": {
    "n_clusters": 5
  },
  "start_times": ["09:00", "10:00", "11:00", "09:00", "10:00"],
  "visit_dates": ["July 16, 2024", "July 17, 2024", "July 18, 2024", "July 19, 2024", "July 20, 2024"]
}
```

**Important:** Array lengths must match (`n_clusters`, `start_times`, and `visit_dates`).

## Output

- **`output/(tripname)/location_schedule_combined.html`**: Main interactive map with schedule
- **`output/(tripname)/location_schedule_day_X.csv`**: Daily schedules
- **`output/(tripname)/complete_location_schedule.csv`**: Complete schedule
- **`geocode_cache.json`**: Cached geocoding results

The main HTML file opens automatically in your browser.

## Customization

### Templates
Edit `templates/schedule_template.html` to customize the HTML output. Available placeholders:
- `{{title}}`: Page title
- `{{map_html}}`: Interactive map
- `{{schedule_html}}`: Schedule display
- `{{unlocated_section}}`: Unlocated addresses section

### OpenAI Integration
Create `api_keys.json` for address reformatting:
```json
{
    "openai_api_key": "your-openai-api-key-here"
}
```

## Geocoding Services

The script uses multiple geocoding services for maximum success:

1. **Nominatim** (OpenStreetMap) - Primary service
2. **OpenAI Address Reformating** - Improves address format if available
3. **Photon API** - Fallback service for failed addresses
4. **Persistent Caching** - Saves results to avoid repeated API calls

## Troubleshooting

- **Geocoding failures**: Some addresses may not be found. The script tries multiple services and will skip only after all attempts fail.
- **Performance**: First run is slower due to geocoding. Subsequent runs use cached results.
- **Debug mode**: Use `--debug` flag for detailed output and troubleshooting.
- **Multiple days**: Ensure `n_clusters`, `start_times`, and `visit_dates` arrays have matching lengths. 