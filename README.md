# Apartment Visitation Scheduler

This program takes a CSV file of apartment addresses, geocodes them, clusters them into two days of visits, optimizes the visitation order, and generates both a schedule and an interactive HTML map.

## Features
- **Smart CSV Selection**: Automatically detects CSV files in `/spreadsheets/` folder and lets you choose which one to process
- **Intelligent Geocoding**: Uses Nominatim (OpenStreetMap) with OpenAI-powered address reformatting for better success rates
- **Persistent Caching**: Saves geocoding results to avoid re-querying the API for the same addresses
- **Smart Retry Logic**: Only retries on actual errors (network issues, rate limits), not on invalid addresses
- **Route Optimization**: Clusters apartments into two days to minimize travel distance
- **Optimized Scheduling**: 1 hour per apartment with 30-minute buffer between visits
- **Combined Output**: Interactive HTML map and schedule displayed side by side
- **Multiple Output Formats**: HTML map + schedule, plus separate CSV schedule file

## Installation

1. Install Python 3.7+
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Setup

### Optional: OpenAI Integration
For better address geocoding success, create an `api_keys.json` file:
```json
{
    "openai_api_key": "your-openai-api-key-here"
}
```

### Optional: Spreadsheets Folder
Create a `spreadsheets/` folder and place your CSV files there for easy selection.

## Usage

### Method 1: Interactive Selection (Recommended)
1. Place your CSV files in the `spreadsheets/` folder
2. Run the script:
   ```bash
   python main.py
   ```
3. Choose from the numbered list of available CSV files
4. Press Enter to use the default file

### Method 2: Direct File Specification
```bash
python main.py your_file.csv
```

### CSV Format
Your CSV should have these columns:
- `Apartment` (or `apartment name`)
- `Address`
- `Price Range` (or `price`)

## Output Files

- **`apartment_schedule.html`**: Interactive map with apartments colored by day (red = Day 1, blue = Day 2) and schedule side by side
- **`apartment_schedule.csv`**: Detailed visitation schedule with times and addresses
- **`geocode_cache.json`**: Cached geocoding results for faster future runs

## How It Works

1. **CSV Processing**: Reads your apartment data
2. **Smart Geocoding**: 
   - Checks cache first for existing results
   - Uses Nominatim for geocoding
   - If address fails, uses OpenAI to reformat and retry once
   - Skips invalid addresses after one reformat attempt
3. **Clustering**: Groups apartments into two days using K-means clustering
4. **Route Optimization**: Orders visits within each day to minimize travel distance
5. **Schedule Generation**: Creates detailed schedule with 1-hour visits and 30-minute buffers
6. **Map Creation**: Generates interactive HTML map with all apartments

## Schedule Details

- **Day 1**: July 16, 2024 (Red markers on map)
- **Day 2**: July 17, 2024 (Blue markers on map)
- **Visit Duration**: 1 hour per apartment
- **Buffer Time**: 30 minutes between visits
- **Start Time**: 9:00 AM each day

## Troubleshooting

### Geocoding Issues
- Some addresses may not be found in the geocoding database
- The script will skip these after one reformat attempt
- Check your address accuracy if many are being skipped

### OpenAI Integration
- If `api_keys.json` is missing or invalid, the script will work without OpenAI reformatting
- OpenAI is only used when addresses fail to geocode initially

### Performance
- First run may be slower due to geocoding
- Subsequent runs will be much faster due to caching
- Invalid addresses are cached to avoid repeated attempts

## Notes
- Geocoding uses the free Nominatim (OpenStreetMap) service
- For large numbers of addresses, consider using a paid geocoding API
- The script respects API rate limits and includes exponential backoff for errors 
- For large numbers of addresses, consider using a paid geocoding API. 