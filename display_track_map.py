import os
import folium
from folium import Marker, PolyLine
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import numpy as np
from datetime import datetime

# Import parse_file function from flexible_time_predictor
from flexible_time_predictor import parse_file
 
# Define landmarks
LANDMARKS = [
    {"coords": (81.28668, 30.97381), "name": "Darchen", "icon": "home"},
    {"coords": (81.25838, 30.99363), "name": "First Prostration Point", "icon": "flag"},
    {"coords": (81.25956, 31.01228), "name": "Sershong", "icon": "flag"},
    {"coords": (81.26942, 31.03033), "name": "Chuku Temple Supply Point", "icon": "cutlery"},
    {"coords": (81.28070, 31.05266), "name": "Second Prostration Point", "icon": "flag"},
    {"coords": (81.28729, 31.08469), "name": "Dazhen Supply Point", "icon": "home"},
    {"coords": (81.32113, 31.10414), "name": "Gangjia/Zhire Temple", "icon": "building"},
]

def create_track_map(track_file, output_file="kora_route_map.html"):
    """Create an interactive map showing the track and landmarks"""
    # Parse the track file
    coords, times = parse_file(track_file)
    
    # Extract lat, lon for mapping
    lats = [coord[1] for coord in coords]  # Latitude is the second element
    lons = [coord[0] for coord in coords]  # Longitude is the first element
    
    # Calculate center of the track
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)
    
    # Create a map centered on the track
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12,
                   tiles='OpenStreetMap')
    
    # Create a colormap for the track based on time
    start_time = times[0]
    end_time = times[-1]
    total_duration = (end_time - start_time).total_seconds()
    
    # Create a simple track line without color gradient
    points = []
    
    for i in range(len(coords)):
        lat, lon = coords[i][1], coords[i][0]
        points.append((lat, lon))
    
    # Determine segment boundaries based on landmarks
    segment_boundaries = []
    for landmark in LANDMARKS:
        lon, lat = landmark["coords"]
        # Find the closest point on the track to this landmark
        min_dist = float('inf')
        closest_idx = 0
        for i, (point_lat, point_lon) in enumerate(points):
            dist = ((point_lat - lat)**2 + (point_lon - lon)**2)**0.5
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        segment_boundaries.append(closest_idx)
    
    # Sort segment boundaries
    segment_boundaries.sort()
    
    # Add a simple track line
    folium.PolyLine(
        points,
        color='blue',
        weight=3,
        opacity=0.7
    ).add_to(m)
    
    # Define segment names based on landmarks (with English translations)
    segment_names = [
        "s0: Darchen→First Prostration Point",
        "s1: First Prostration Point→Sershong",
        "s2: Sershong→Chuku Temple Supply Point",
        "s3: Chuku Temple→Second Prostration Point",
        "s4: Second Prostration Point→Dazhen Supply Point",
        "s5: Dazhen Supply Point→ Gangjia/Zhire Temple"
    ]
    
    # Add segment labels directly between landmarks
    for i in range(len(LANDMARKS) - 1):
        # Get coordinates of adjacent landmarks
        start_lon, start_lat = LANDMARKS[i]["coords"]
        end_lon, end_lat = LANDMARKS[i+1]["coords"]
        
        # Calculate midpoint between landmarks for label placement
        mid_lat = (start_lat + end_lat) / 2
        mid_lon = (start_lon + end_lon) / 2
        
        # Get segment name
        segment_label = segment_names[i] if i < len(segment_names) else f"s{i}"
        
        # Add segment label with horizontal alignment
        folium.map.Marker(
            [mid_lat, mid_lon],
            icon=folium.DivIcon(html=f'<div style="font-size: 10pt; color: black; background-color: white; padding: 2px 5px; border-radius: 3px; opacity: 0.8; text-align: center; white-space: nowrap;">{segment_label}</div>')
        ).add_to(m)
    
    # We're not adding start/end markers as requested
    
    # Add landmarks with permanent labels
    for landmark in LANDMARKS:
        lon, lat = landmark["coords"]
        # Add marker with icon
        folium.Marker(
            location=[lat, lon],
            popup=landmark["name"],
            icon=folium.Icon(color='blue', icon=landmark["icon"])
        ).add_to(m)
        
        # Add permanent label with horizontal alignment
        folium.map.Marker(
            [lat, lon],
            icon=folium.DivIcon(html=f'<div style="font-size: 11pt; color: blue; font-weight: bold; text-align: center; white-space: nowrap; background-color: rgba(255,255,255,0.7); padding: 1px 3px; border-radius: 2px;">{landmark["name"]}</div>')
        ).add_to(m)
    
    # Add track information
    track_info = f"""
    <h3>Track Information</h3>
    <ul>
        <li><b>File:</b> {os.path.basename(track_file)}</li>
        <li><b>Start Time:</b> {times[0].strftime('%Y-%m-%d %H:%M')}</li>
        <li><b>End Time:</b> {times[-1].strftime('%Y-%m-%d %H:%M')}</li>
        <li><b>Duration:</b> {end_time - start_time}</li>
        <li><b>Points:</b> {len(coords)}</li>
    </ul>
    """
    
    # Add a legend as an HTML element
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 200px; height: 120px; 
                border:2px solid grey; z-index:9999; font-size:14px;
                background-color:white; padding: 10px;
                border-radius: 5px;">
      <p style="margin-top: 0; margin-bottom: 5px;"><b>Time Progression</b></p>
      <div style="display: flex; align-items: center; margin-bottom: 5px;">
        <div style="background: blue; width: 30px; height: 15px; margin-right: 5px;"></div>
        <span>Start</span>
      </div>
      <div style="display: flex; align-items: center; margin-bottom: 5px;">
        <div style="background: green; width: 30px; height: 15px; margin-right: 5px;"></div>
        <span>Middle</span>
      </div>
      <div style="display: flex; align-items: center;">
        <div style="background: red; width: 30px; height: 15px; margin-right: 5px;"></div>
        <span>End</span>
      </div>
    </div>
    '''
    
    # Add title
    title_html = f'''
    <h3 align="center" style="font-size:20px"><b>冈仁波齐转山路线图</b></h3>
    <div align="center" style="font-size:16px">轨迹文件: {os.path.basename(track_file)}</div>
    '''
    
    # Add HTML elements to the map
    m.get_root().html.add_child(folium.Element(title_html))
    m.get_root().html.add_child(folium.Element(legend_html))
    m.get_root().html.add_child(folium.Element(track_info))
    
    # Save the map
    m.save(output_file)
    print(f"Map saved to {output_file}")
    return output_file

def main():
    # Path to the track file
    track_file = "./tracks/0827.track"
    
    # Check if file exists
    if not os.path.exists(track_file):
        print(f"Error: Track file {track_file} not found.")
        # Try to find the file in the directory
        tracks_dir = "./tracks"
        if os.path.exists(tracks_dir):
            files = os.listdir(tracks_dir)
            track_files = [f for f in files if f.endswith('.track')]
            if track_files:
                print(f"Available track files: {track_files[:10]}")
                if '0827.track' in track_files:
                    track_file = os.path.join(tracks_dir, '0827.track')
                    print(f"Found track file at {track_file}")
                else:
                    # Use the first available track file
                    track_file = os.path.join(tracks_dir, track_files[0])
                    print(f"Using alternative track file: {track_file}")
            else:
                print("No track files found in the directory.")
                return
        else:
            print(f"Directory {tracks_dir} not found.")
            return
    
    # Create the map
    output_file = create_track_map(track_file)
    
    print(f"Map created successfully. Open {output_file} in a web browser to view.")

if __name__ == "__main__":
    main()
