import os
import xml.etree.ElementTree as ET
import math
from datetime import datetime
import re
import csv

def parse_timestamp(timestamp_str):
    """
    Parse timestamp with robust handling of different formats
    """
    # Handle Chinese AM/PM markers
    if '下午' in timestamp_str:
        # It's PM, need to adjust hours
        timestamp_str = timestamp_str.replace('下午', '')
        # Parse the time part to adjust hours
        time_parts = re.search(r'T(\d{1,2}):(\d{2}):(\d{2})', timestamp_str)
        if time_parts:
            hour = int(time_parts.group(1))
            # Add 12 to hours except when it's already 12 (noon)
            if hour != 12:
                new_hour = hour + 12
                timestamp_str = re.sub(r'T\d{1,2}:', f'T{new_hour}:', timestamp_str)
    else:
        # It's AM, just remove the marker
        timestamp_str = timestamp_str.replace('上午', '')
    
    try:
        # Try standard ISO format
        # return datetime.fromisoformat(timestamp_str.replace('Z', '+08:00'))
        return datetime.fromisoformat(timestamp_str)
    except ValueError:
        try:
            # Try alternative parsing
            return datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S%z')
        except ValueError:
            print(f"Could not parse timestamp: {timestamp_str}")
            return None

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

def process_kml_track(file_path):
    # Namespaces for KML parsing
    namespaces = {
        'kml': 'http://www.opengis.net/kml/2.2',
        'gx': 'http://www.google.com/kml/ext/2.2'
    }
    
    try:
        # Parse the XML tree
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Find Folder with TbuluTrackFolder ID
        track_folder = root.find(".//kml:Folder[@id='TbuluTrackFolder']", namespaces)
        
        if track_folder is None:
            print(f"No TbuluTrackFolder found in {file_path}")
            return None
        
        # Find all gx:Track within this folder
        tracks = track_folder.findall(".//gx:Track", namespaces)
        
        if not tracks:
            print(f"No gx:Track found in {file_path}")
            return None
        
        track_points = []
        total_distance = 0
        
        # Process each track
        for track in tracks:
            # Extract coordinates and timestamps
            coords = track.findall("gx:coord", namespaces)
            when_times = track.findall("kml:when", namespaces)
            
            # Take the minimum length to ensure alignment
            min_length = min(len(coords), len(when_times))
            coords = coords[:min_length]
            when_times = when_times[:min_length]
            
            # Process coordinates and timestamps            
            for i in range(1, len(coords)):
                # Parse coordinates
                coord1 = coords[i-1].text.split()
                coord2 = coords[i].text.split()
                
                lon1, lat1, alt1 = float(coord1[0]), float(coord1[1]), float(coord1[2])
                lon2, lat2, alt2 = float(coord2[0]), float(coord2[1]), float(coord2[2])
                
                # Calculate distance between points
                distance = haversine_distance(lat1, lon1, lat2, lon2)
                total_distance += distance
                
                # Parse timestamps
                timestamp1 = parse_timestamp(when_times[i-1].text)
                timestamp2 = parse_timestamp(when_times[i].text)
                
                # Skip if timestamp parsing failed
                if timestamp1 is None or timestamp2 is None:
                    continue
                
                # Store track point information
                track_points.append({
                    'lon1': lon1, 'lat1': lat1, 'alt1': alt1,
                    'lon2': lon2, 'lat2': lat2, 'alt2': alt2,
                    'distance': distance,
                    'timestamp1': timestamp1,
                    'timestamp2': timestamp2
                })
                    
        return {
            'total_distance': total_distance,
            'track_points': track_points
        }
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def write_track_lengths_to_csv(tracks_info, csv_file_path):
    """
    Write track lengths to a CSV file.
    """
    with open(csv_file_path, mode='w', newline='') as csvfile:
        fieldnames = ['filename', 'total_distance']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for info in tracks_info:
            writer.writerow(info)

def main():
    # Create tracks directory if it doesn't exist
    tracks_dir = './tracks'
    os.makedirs(tracks_dir, exist_ok=True)
    
    # KML source directory
    kml_directory = './kmls'
    
    # Counters
    total_files = 0
    processed_files = 0
    tracks_written = 0
    
    # List to store track lengths information
    tracks_info = []
    
    # Process each KML file
    for filename in sorted(os.listdir(kml_directory)):
        if not filename.endswith('.kml'):
            continue
        
        total_files += 1
        file_path = os.path.join(kml_directory, filename)
        
        # Process track
        track_data = process_kml_track(file_path)
        
        if track_data is None:
            print(f"Error processing {filename}")
            continue
        
        processed_files += 1
        
        tracks_info.append({'filename': filename, 'total_distance': track_data['total_distance']})
        
        # Check track length
        if 30 <= track_data['total_distance'] <= 60:
            # Generate track filename (remove .kml, add .track)
            track_filename = os.path.splitext(filename)[0] + '.track'
            track_filepath = os.path.join(tracks_dir, track_filename)
            
            # Write track data
            with open(track_filepath, 'w') as f:
                # Write total distance
                f.write(f"Total Distance: {track_data['total_distance']:.2f} km\n")
                f.write("Track Points:\n")
                
                # Write each track point
                for point in track_data['track_points']:
                    f.write(f"Coord1: {point['lon1']} {point['lat1']} {point['alt1']}\n")
                    f.write(f"Coord2: {point['lon2']} {point['lat2']} {point['alt2']}\n")
                    f.write(f"Distance: {point['distance']:.4f} km\n")
                    f.write(f"Timestamp1: {point['timestamp1'].isoformat()}\n")
                    f.write(f"Timestamp2: {point['timestamp2'].isoformat()}\n")
                    f.write("---\n")
            
            tracks_written += 1
            print(f"Writing {filename} with track length {track_data['total_distance']:.2f} km")
        else:
            print(f"Skipping {filename} with track length {track_data['total_distance']:.2f} km")
    
    # Write track lengths to CSV after processing all files
    csv_file_path = os.path.join(tracks_dir, 'track_lengths.csv')
    write_track_lengths_to_csv(tracks_info, csv_file_path)
    
    print(f"Total KML files: {total_files}")
    print(f"Processed files: {processed_files}")
    print(f"Tracks written: {tracks_written}")

if __name__ == "__main__":
    main()
