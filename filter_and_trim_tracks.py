#!/usr/bin/env python3
import os
import csv
import glob
import shutil
from datetime import datetime, timezone, timedelta

def format_datetime(dt_str):
    """Convert datetime string to datetime object"""
    if not dt_str or dt_str == "Unknown":
        return None
    # Parse the datetime string and make it timezone-aware with UTC+8
    dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
    utc_plus_8 = timezone(timedelta(hours=8))
    return dt.replace(tzinfo=utc_plus_8)

def main():
    # Path to the continuous_tracks.csv file
    csv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'continuous_tracks.csv')
    
    # Path to the original tracks directory
    original_tracks_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tracks')
    
    # Create a new directory for filtered and trimmed tracks
    filtered_tracks_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'filtered_tracks')
    os.makedirs(filtered_tracks_dir, exist_ok=True)
    
    # Read the CSV file
    filtered_tracks = []
    with open(csv_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert time_between_circles_hours to float
            time_between_circles = float(row['time_between_circles_hours']) if row['time_between_circles_hours'] else None
            
            # Filter tracks with time_between_circles_hours between 5 and 10
            if time_between_circles is not None and 5 <= time_between_circles <= 10:
                filtered_tracks.append(row)
    
    print(f"Found {len(filtered_tracks)} tracks with time_between_circles_hours between 5 and 10")
    
    # Process each filtered track
    for track_info in filtered_tracks:
        track_file = track_info['file']
        first_point_in_second_circle_time = format_datetime(track_info['first_point_in_second_circle_time'])
        
        # Skip if we couldn't parse the time
        if not first_point_in_second_circle_time:
            print(f"Skipping {track_file}: Could not parse first_point_in_second_circle_time")
            continue
        
        # Path to the original track file
        original_track_path = os.path.join(original_tracks_dir, track_file)
        
        # Path to the new filtered track file
        filtered_track_path = os.path.join(filtered_tracks_dir, track_file)
        
        # Read the original track file
        try:
            with open(original_track_path, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"Skipping {track_file}: File not found")
            continue
        
        # Extract the header lines (first two lines)
        header_lines = lines[:2]
        
        # Process track points
        track_points = []
        current_point = {}
        points_to_keep = []
        
        # Parse track points
        for line in lines[2:]:  # Skip header lines
            line = line.strip()
            if not line:
                continue
            
            if line.startswith("Coord1:"):
                # Format: "Coord1: 81.263788 30.989113 4718.121924"
                current_point = {'coord': line}
            elif line.startswith("Timestamp1:"):
                # Format: "Timestamp1: 2023-08-21T00:06:31+00:00"
                current_point['timestamp'] = line
                timestamp_str = line.split(':', 1)[1].strip()
                try:
                    # Parse the timestamp in UTC
                    timestamp = datetime.fromisoformat(timestamp_str)
                    
                    # Convert to UTC+8 (China Standard Time) to match the original script
                    utc_plus_8 = timezone(timedelta(hours=8))
                    timestamp = timestamp.astimezone(utc_plus_8)
                    
                    current_point['datetime'] = timestamp
                except ValueError:
                    current_point['datetime'] = None
            elif line == "---" and 'coord' in current_point and 'timestamp' in current_point:
                # End of a point entry, add to track points
                track_points.append(current_point)
                current_point = {}
        
        # Add the last point if it exists
        if 'coord' in current_point and 'timestamp' in current_point:
            track_points.append(current_point)
        
        # Find points to keep (all points up to and including the first point in second circle)
        found_first_in_second = False
        for point in track_points:
            points_to_keep.append(point)
            
            # Compare timestamps with a small tolerance for potential formatting differences
            if point.get('datetime') and first_point_in_second_circle_time:
                time_diff = abs((point['datetime'] - first_point_in_second_circle_time).total_seconds())
                # Allow a 1-second tolerance for comparison
                if time_diff < 1:
                    found_first_in_second = True
                    break
        
        if not found_first_in_second:
            print(f"Skipping {track_file}: Could not find first point in second circle")
            continue
        
        # Write the filtered track file
        with open(filtered_track_path, 'w') as f:
            # Write header
            f.writelines(header_lines)
            
            # Write filtered points
            for point in points_to_keep:
                f.write(point['coord'] + '\n')
                f.write(point['timestamp'] + '\n')
                f.write("---\n")
        
        print(f"Processed {track_file}: Kept {len(points_to_keep)} out of {len(track_points)} points")
    
    print(f"\nFiltered and trimmed tracks saved to: {filtered_tracks_dir}")

if __name__ == "__main__":
    main()
