import os
import math
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
from datetime import datetime
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# Import functions from flexible_time_predictor
from flexible_time_predictor import parse_file, haversine, find_segment_index, calculate_time_to_destination

# Configuration
CONFIG = {
    'model_path': 'flexible_time_model.pkl',
    'tracks_dir': './filtered_tracks',
    'output_dir': './visualization_results',
    'split_points': [
        (81.25838, 30.99363),
        (81.25956, 31.01228),
        (81.26942, 31.03033),
        (81.28070, 31.05266),
        (81.28729, 31.08469)
    ]
}

def load_model():
    """Load the trained model"""
    print(f"Loading model from {CONFIG['model_path']}...")
    model_data = joblib.load(CONFIG['model_path'])
    return model_data

def list_track_files():
    """List all track files in the directory"""
    track_files = []
    for filename in os.listdir(CONFIG['tracks_dir']):
        if filename.endswith('track'):
            track_files.append(filename)
    return track_files

def select_track_file(track_files):
    """Select a track file with comprehensive data coverage"""
    # Try to find tracks with more comprehensive data
    track_candidates = []
    
    print("Analyzing tracks to find the best one...")
    
    # Examine up to 10 tracks to find a good candidate
    for filename in track_files[:30]:  # Limit to first 30 files for efficiency
        filepath = os.path.join(CONFIG['tracks_dir'], filename)
        try:
            coords, times = parse_file(filepath)
            
            # Skip very short tracks
            if len(coords) < 100:
                continue
                
            # Skip very long tracks (too much processing)
            if len(coords) > 2000:
                continue
                
            # Calculate track length
            track_length = 0
            for i in range(1, len(coords)):
                track_length += haversine(coords[i-1][0], coords[i-1][1], coords[i][0], coords[i][1])
            
            # Skip very short tracks in terms of distance
            if track_length < 5000:  # 5km
                continue
                
            # Calculate time span
            time_span = (times[-1] - times[0]).total_seconds()
            
            # Skip very short tracks in terms of time
            if time_span < 3600:  # 1 hour
                continue
                
            # Add to candidates with a score (higher is better)
            # Prefer tracks with moderate length and duration
            score = min(len(coords), 1000) * min(track_length/1000, 20) * min(time_span/3600, 10)
            
            track_candidates.append({
                'filename': filename,
                'points': len(coords),
                'length': track_length,
                'duration': time_span,
                'score': score
            })
            
            print(f"Analyzed {filename}: {len(coords)} points, {track_length/1000:.1f}km, {time_span/3600:.1f}h, score: {score:.1f}")
            
        except Exception as e:
            print(f"Error analyzing {filename}: {e}")
            continue
    
    # Sort candidates by score (highest first)
    track_candidates.sort(key=lambda x: x['score'], reverse=True)
    
    # Print top candidates
    if track_candidates:
        print("\nTop track candidates:")
        for i, track in enumerate(track_candidates[:5]):
            print(f"{i+1}. {track['filename']}: {track['points']} points, {track['length']/1000:.1f}km, {track['duration']/3600:.1f}h")
        
        # Select the highest scoring track
        selected_file = track_candidates[0]['filename']
        print(f"\nSelected track: {selected_file}")
        return selected_file
    
    # Fallback to a default track if no good candidates found
    print("No suitable tracks found, using first available")
    return track_files[0] if track_files else None

def prepare_track_data(track_file, model_data):
    """Prepare data for a single track"""
    filepath = os.path.join(CONFIG['tracks_dir'], track_file)
    coords, times = parse_file(filepath)
    
    print(f"Track {track_file} has {len(coords)} coordinate points")
    
    # Extract model components
    model = model_data['model']
    features = model_data['features']
    segment_encoder = model_data['segment_encoder']
    destination_encoder = model_data['destination_encoder']
    
    # Prepare data for each point in the track
    track_data = []
    
    # Get month and determine season
    start_time = times[0]
    month = start_time.month
    # Northern hemisphere seasons
    if 3 <= month <= 5:
        season = 0  # Spring
    elif 6 <= month <= 8:
        season = 1  # Summer
    elif 9 <= month <= 11:
        season = 2  # Fall
    else:
        season = 3  # Winter
    
    # Process each point
    for i in range(1, len(coords) - 1):
        lon, lat, ele = coords[i]
        lon_prev, lat_prev, ele_prev = coords[i-1]
        t_cur = times[i]
        t_prev = times[i-1]
        
        # Calculate speed
        delta_t = max((t_cur - t_prev).total_seconds(), 0.1)
        speed = haversine(lon_prev, lat_prev, lon, lat) / delta_t
        
        # Calculate gradient
        horiz_dist = haversine(lon_prev, lat_prev, lon, lat)
        epsilon = 0.001
        gradient = (ele - ele_prev) / max(horiz_dist, epsilon)
        gradient = max(min(gradient, 1.0), -1.0)  # Cap gradient
        
        # Get segment and difficulty
        seg_idx = find_segment_index(lon, lat)
        
        # For each destination (split points + endpoint)
        destinations = CONFIG['split_points'].copy()
        destinations.append((coords[-1][0], coords[-1][1]))  # Add endpoint
        
        for dest_idx, destination in enumerate(destinations):
            # Calculate distance to destination
            dest_lon, dest_lat = destination
            dist_to_dest = haversine(lon, lat, dest_lon, dest_lat)
            
            # Skip if destination is too far
            if dist_to_dest > 20000:
                continue
                
            # Calculate actual time to destination
            actual_time = calculate_time_to_destination(i, coords, times, destination)
            
            # Skip if we couldn't calculate time
            if actual_time is None:
                continue
                
            # Skip unreasonable times
            if actual_time > 12 * 3600:
                continue
                
            # Create feature vector for prediction
            feature_dict = {
                'speed': speed,
                'dist_to_dest': dist_to_dest,
                'gradient': gradient,
                'time_of_day': start_time.hour,
                'difficulty_level': min(seg_idx, 5),  # Ensure within bounds
                'season': season,
                'destination_idx': dest_idx
            }
            
            # One-hot encode segment index
            segment_ohe = segment_encoder.transform([[seg_idx]])
            for j, val in enumerate(segment_ohe[0]):
                feature_dict[f'segment_{j}'] = val
                
            # One-hot encode destination index
            dest_ohe = destination_encoder.transform([[dest_idx]])
            for j, val in enumerate(dest_ohe[0]):
                feature_dict[f'dest_{j}'] = val
                
            # Create DataFrame for prediction
            X = pd.DataFrame([feature_dict])[features]
            
            # Make prediction
            predicted_time = model.predict(X)[0]
            
            # Calculate error
            error = actual_time - predicted_time
            abs_error = abs(error)
            
            # Add to track data
            track_data.append({
                'lon': lon,
                'lat': lat,
                'ele': ele,
                'time': t_cur,
                'segment_index': seg_idx,
                'destination_idx': dest_idx,
                'destination': f"{'Split Point ' + str(dest_idx) if dest_idx < len(CONFIG['split_points']) else 'Endpoint'}",
                'distance_to_dest': dist_to_dest,
                'actual_time': actual_time,
                'predicted_time': predicted_time,
                'error': error,
                'abs_error': abs_error,
                'error_minutes': error / 60,
                'abs_error_minutes': abs_error / 60
            })
    
    return pd.DataFrame(track_data)

def create_track_error_map(track_data, track_file):
    """Create an interactive map showing prediction errors along the track"""
    # Calculate center of the track
    center_lat = track_data['lat'].mean()
    center_lon = track_data['lon'].mean()
    
    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
    
    # Add split points to the map
    for i, (lon, lat) in enumerate(CONFIG['split_points']):
        folium.Marker(
            location=[lat, lon],
            popup=f"Split Point {i}",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(m)
    
    # Normalize errors for color scaling
    max_error = track_data['abs_error_minutes'].max()
    
    # Group by destination for separate visualizations
    for dest_idx, dest_group in track_data.groupby('destination_idx'):
        dest_name = f"Split Point {dest_idx}" if dest_idx < len(CONFIG['split_points']) else "Endpoint"
        
        # Create a feature group for this destination
        fg = folium.FeatureGroup(name=f"Destination: {dest_name}")
        
        # Add points to the map
        for _, row in dest_group.iterrows():
            # Calculate color based on error (red for high error, green for low)
            error_ratio = min(row['abs_error_minutes'] / max_error, 1.0)
            color = f'#{int(255 * error_ratio):02x}{int(255 * (1-error_ratio)):02x}00'
            
            # Create popup content
            popup_html = f"""
            <b>Coordinates:</b> {row['lat']:.5f}, {row['lon']:.5f}<br>
            <b>Destination:</b> {row['destination']}<br>
            <b>Distance to Destination:</b> {row['distance_to_dest']/1000:.2f} km<br>
            <b>Actual Time:</b> {row['actual_time']/60:.2f} minutes<br>
            <b>Predicted Time:</b> {row['predicted_time']/60:.2f} minutes<br>
            <b>Error:</b> {row['error_minutes']:.2f} minutes<br>
            <b>Segment:</b> {row['segment_index']}
            """
            
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=5,
                popup=folium.Popup(popup_html, max_width=300),
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7
            ).add_to(fg)
        
        # Add the feature group to the map
        fg.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add title
    title_html = f'''
    <h3 align="center" style="font-size:16px"><b>Prediction Errors for Track: {track_file}</b></h3>
    <h4 align="center" style="font-size:14px">Red = High Error, Green = Low Error</h4>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Save the map
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    map_path = os.path.join(CONFIG['output_dir'], f'track_error_map_{track_file}.html')
    m.save(map_path)
    
    print(f"Interactive error map saved to {map_path}")
    return map_path

def create_track_error_summary(track_data, track_file):
    """Create summary visualizations for the track errors"""
    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # 1. Error distribution by destination
    plt.figure(figsize=(14, 8))
    
    # Group by destination
    dest_stats = track_data.groupby('destination')['abs_error_minutes'].agg(['mean', 'std']).reset_index()
    
    # Bar plot with error bars
    bars = plt.bar(dest_stats['destination'], dest_stats['mean'], 
                  yerr=dest_stats['std'], capsize=10, alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Destination', fontsize=14)
    plt.ylabel('Mean Absolute Error (minutes)', fontsize=14)
    plt.title(f'Mean Absolute Error by Destination for Track: {track_file}', fontsize=16)
    plt.xticks(rotation=45)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=12)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], f'track_error_by_dest_{track_file}.png'), dpi=300)
    plt.close()
    
    # 2. Error vs Distance scatter plot
    plt.figure(figsize=(14, 8))
    
    # Create scatter plot
    scatter = plt.scatter(
        track_data['distance_to_dest']/1000,  # Convert to km
        track_data['abs_error_minutes'],
        c=track_data['segment_index'],
        alpha=0.7,
        cmap='viridis'
    )
    
    # Add labels and title
    plt.xlabel('Distance to Destination (km)', fontsize=14)
    plt.ylabel('Absolute Error (minutes)', fontsize=14)
    plt.title(f'Prediction Error vs Distance for Track: {track_file}', fontsize=16)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Segment Index', fontsize=12)
    
    # Add trend line
    z = np.polyfit(track_data['distance_to_dest']/1000, track_data['abs_error_minutes'], 1)
    p = np.poly1d(z)
    plt.plot(
        sorted(track_data['distance_to_dest']/1000),
        p(sorted(track_data['distance_to_dest']/1000)),
        "r--", 
        alpha=0.7
    )
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], f'track_error_vs_distance_{track_file}.png'), dpi=300)
    plt.close()
    
    print(f"Track error summary visualizations saved to {CONFIG['output_dir']}")

def main():
    """Main function to visualize track predictions"""
    # Load model
    model_data = load_model()
    
    # List track files
    track_files = list_track_files()
    if not track_files:
        print("No track files found in directory")
        return
    
    # Select a track file
    selected_track = select_track_file(track_files)
    if not selected_track:
        print("No suitable track file found")
        return
    
    print(f"Selected track file: {selected_track}")
    
    # Prepare track data
    track_data = prepare_track_data(selected_track, model_data)
    
    # Create visualizations
    map_path = create_track_error_map(track_data, selected_track)
    create_track_error_summary(track_data, selected_track)
    
    print(f"\nTrack visualization complete!")
    print(f"Open {map_path} in a web browser to view the interactive map")

if __name__ == "__main__":
    main()
