#!/usr/bin/env python3
import os
import glob
import math
import csv
from datetime import datetime, timedelta, timezone
import json

# Reference points
START_POINT_REF = {
    'lat': 30.97381664400325,
    'lon': 81.28668948086344,
    'radius_km': 2.0
}

TRACK_POINT_REF = {
    'lat': 31.104143826818795,
    'lon': 81.32113810070767,
    'radius_km': 1.0
}

# Minimum gap in hours to be considered a "long gap"
MIN_GAP_HOURS = 2.0

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

def is_point_in_circle(point, circle_ref):
    """Check if a point is within a circle defined by center and radius"""
    distance = haversine_distance(
        point['lat'], point['lon'],
        circle_ref['lat'], circle_ref['lon']
    )
    return distance <= circle_ref['radius_km']

def parse_track_file(track_file):
    """Parse a track file and check for continuous tracks between circles"""
    # Default return structure with all required keys to avoid KeyError
    default_result = {
        'file': os.path.basename(track_file),
        'first_point_in_radius': False,
        'has_point_in_second_radius': False,
        'has_long_gap_between_circles': False,
        'total_time_gap_hours': None,
        'all_points_in_season': False,
        'meets_criteria': False,
        'reason': "Error processing file"
    }
    
    try:
        with open(track_file, 'r') as f:
            lines = f.readlines()
        
        # Extract total distance from the first line
        total_distance = None
        if lines and "Total Distance:" in lines[0]:
            try:
                total_distance = float(lines[0].split(':')[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        
        # Initialize variables
        track_points = []
        current_point = {}
        
        # Parse track points
        for line in lines[2:]:  # Skip header lines
            line = line.strip()
            if not line:
                continue
            
            if line.startswith("Coord1:"):
                # Format: "Coord1: 81.263788 30.989113 4718.121924"
                parts = line.split(':')[1].strip().split()
                if len(parts) >= 3:
                    lon, lat, alt = map(float, parts[:3])
                    current_point = {
                        'lon': lon,
                        'lat': lat,
                        'alt': alt
                    }
            elif line.startswith("Timestamp1:"):
                # Format: "Timestamp1: 2023-08-21T00:06:31+00:00"
                timestamp_str = line.split(':', 1)[1].strip()
                try:
                    # Parse the timestamp in UTC
                    timestamp = datetime.fromisoformat(timestamp_str)
                    
                    # Convert to UTC+8 (China Standard Time)
                    utc_plus_8 = timezone(timedelta(hours=8))
                    timestamp = timestamp.astimezone(utc_plus_8)
                    
                    current_point['timestamp'] = timestamp
                except ValueError:
                    pass
            elif line == "---" and 'lon' in current_point and 'lat' in current_point and 'timestamp' in current_point:
                # End of a point entry, add to track points
                track_points.append(current_point)
                current_point = {}
        
        # Add the last point if it exists
        if 'lon' in current_point and 'lat' in current_point and 'timestamp' in current_point:
            track_points.append(current_point)
        
        if not track_points:
            print(f"Warning: No valid track points found in {track_file}")
            return None
        
        # Get first and last points
        first_point = track_points[0]
        last_point = track_points[-1]
        
        # Check if first point is within radius of START_POINT_REF
        first_point_in_radius = is_point_in_circle(first_point, START_POINT_REF)
        
        # Mark each point with its circle membership and find first point in second circle
        has_point_in_second_radius = False
        first_point_in_second_circle_index = -1
        
        for i, point in enumerate(track_points):
            # Check if point is in first reference circle
            point['in_first_circle'] = is_point_in_circle(point, START_POINT_REF)
            
            # Check if point is in second reference circle
            if is_point_in_circle(point, TRACK_POINT_REF):
                point['in_second_circle'] = True
                has_point_in_second_radius = True
                if first_point_in_second_circle_index == -1:
                    first_point_in_second_circle_index = i
            else:
                point['in_second_circle'] = False
        
        # If no points in second circle, return early
        if first_point_in_second_circle_index == -1:
            return {
                'file': os.path.basename(track_file),
                'first_point_in_radius': first_point_in_radius,
                'has_point_in_second_radius': has_point_in_second_radius,
                'total_time_gap_hours': None,
                'all_points_in_season': True,
                'meets_criteria': False,
                'reason': "No points in second circle"
            }
        
        # Find last point in first circle
        last_point_in_first_circle_index = -1
        for i in range(len(track_points) - 1, -1, -1):
            if track_points[i]['in_first_circle']:
                last_point_in_first_circle_index = i
                break
        
        # If no points in first circle (shouldn't happen if first_point_in_radius is True)
        if last_point_in_first_circle_index == -1:
            return {
                'file': os.path.basename(track_file),
                'first_point_in_radius': first_point_in_radius,
                'has_point_in_second_radius': has_point_in_second_radius,
                'total_time_gap_hours': None,
                'all_points_in_season': True,
                'meets_criteria': False,
                'reason': "No points in first circle"
            }
        
        # If last point in first circle is after first point in second circle,
        # it means the track went back to the first circle after visiting the second
        if last_point_in_first_circle_index > first_point_in_second_circle_index:
            # Find the first point in first circle after visiting second circle
            first_point_back_in_first_circle_index = -1
            for i in range(first_point_in_second_circle_index + 1, len(track_points)):
                if track_points[i]['in_first_circle']:
                    first_point_back_in_first_circle_index = i
                    break
            
            if first_point_back_in_first_circle_index != -1:
                # We'll use this as our endpoint for the journey to second circle
                last_point_in_first_circle_index = first_point_back_in_first_circle_index - 1
        
        # Now check if there are any long gaps between leaving first circle and reaching second circle
        has_long_gap_between_circles = False
        long_gaps = []
        
        # Find the last point in first circle before reaching second circle
        last_point_in_first_circle_before_second = -1
        for i in range(first_point_in_second_circle_index - 1, -1, -1):
            if track_points[i]['in_first_circle']:
                last_point_in_first_circle_before_second = i
                break
        
        # If we found a valid last point in first circle
        if last_point_in_first_circle_before_second != -1:
            # Check for long gaps between leaving first circle and reaching second circle
            for i in range(last_point_in_first_circle_before_second + 1, first_point_in_second_circle_index):
                prev_point = track_points[i-1]
                curr_point = track_points[i]
                
                if 'timestamp' in prev_point and 'timestamp' in curr_point:
                    time_diff = curr_point['timestamp'] - prev_point['timestamp']
                    gap_hours = time_diff.total_seconds() / 3600
                    
                    if gap_hours >= MIN_GAP_HOURS:
                        has_long_gap_between_circles = True
                        long_gaps.append({
                            'index': i,
                            'prev_point': prev_point,
                            'curr_point': curr_point,
                            'gap_hours': gap_hours
                        })
        
        # Calculate time from first point to reaching second circle
        time_between_circles = None
        if first_point_in_second_circle_index != -1 and 'timestamp' in first_point and 'timestamp' in track_points[first_point_in_second_circle_index]:
            time_diff = track_points[first_point_in_second_circle_index]['timestamp'] - first_point['timestamp']
            time_between_circles = time_diff.total_seconds() / 3600
        
        # Calculate total time gap between first and last point
        total_time_gap_hours = None
        if 'timestamp' in first_point and 'timestamp' in last_point:
            time_diff = last_point['timestamp'] - first_point['timestamp']
            total_time_gap_hours = time_diff.total_seconds() / 3600
        
        # Check if all points are between May and October (months 5-10)
        all_points_in_season = True
        for point in track_points:
            if 'timestamp' in point:
                month = point['timestamp'].month
                if month < 5 or month > 10:
                    all_points_in_season = False
                    break
        
        # Check if first point's timestamp is in the range of 6am to 2pm
        first_point_in_time_range = False
        if 'timestamp' in first_point:
            hour = first_point['timestamp'].hour
            first_point_in_time_range = 6 <= hour < 14  # 6am to 2pm (14:00)
        
        # Check if the track meets our criteria:
        # 1. First point in first circle
        # 2. At least one point in second circle
        # 3. No long gaps between leaving first circle and reaching second circle
        # 4. Total time between 5-72 hours
        # 5. All points in season (May-Oct)
        # 6. First point's timestamp is in the range of 6am to 2pm
        meets_criteria = (
            first_point_in_radius and 
            has_point_in_second_radius and 
            not has_long_gap_between_circles and
            total_time_gap_hours is not None and 
            5 <= total_time_gap_hours <= 72 and
            all_points_in_season and
            first_point_in_time_range
        )
        
        reason = None
        if not meets_criteria:
            if not first_point_in_radius:
                reason = "First point not in first circle"
            elif not has_point_in_second_radius:
                reason = "No points in second circle"
            elif has_long_gap_between_circles:
                reason = f"Has {len(long_gaps)} long gaps between circles"
            elif total_time_gap_hours is None or not (5 <= total_time_gap_hours <= 72):
                reason = f"Total time gap not between 5-72 hours: {total_time_gap_hours}"
            elif not all_points_in_season:
                reason = "Not all points in season (May-Oct)"
            elif not first_point_in_time_range:
                reason = "First point not in time range (6am-2pm)"
        
        return {
            'file': os.path.basename(track_file),
            'total_distance': total_distance,
            'first_point': first_point,
            'last_point': last_point,
            'first_point_in_radius': first_point_in_radius,
            'has_point_in_second_radius': has_point_in_second_radius,
            'first_point_in_second_circle_index': first_point_in_second_circle_index,
            'first_point_in_second_circle': track_points[first_point_in_second_circle_index] if first_point_in_second_circle_index != -1 else None,
            'last_point_in_first_circle_before_second': last_point_in_first_circle_before_second,
            'last_point_in_first_circle_before_second_data': track_points[last_point_in_first_circle_before_second] if last_point_in_first_circle_before_second != -1 else None,
            'has_long_gap_between_circles': has_long_gap_between_circles,
            'long_gaps': long_gaps,
            'time_between_circles': time_between_circles,
            'total_time_gap_hours': total_time_gap_hours,
            'all_points_in_season': all_points_in_season,
            'first_point_in_time_range': first_point_in_time_range,
            'num_points': len(track_points),
            'meets_criteria': meets_criteria,
            'reason': reason
        }
    
    except Exception as e:
        print(f"Error processing {track_file}: {e}")
        return None

def format_datetime(dt):
    """Format datetime object to string"""
    if dt is None:
        return "Unknown"
    return dt.strftime('%Y-%m-%d %H:%M:%S')

def main():
    # Directory containing track files
    track_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tracks')
    
    # Find all track files
    track_files = glob.glob(os.path.join(track_dir, '*.track'))
    print(f"Found {len(track_files)} track files")
    print(f"Note: All times are converted to UTC+8 (China Standard Time)")
    
    # Create output file
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'continuous_tracks.csv')
    
    # Process all track files
    all_results = []
    filtered_results = []
    
    print(f"Filtering tracks with the following criteria:")
    print(f"1. First point within {START_POINT_REF['radius_km']}KM of ({START_POINT_REF['lat']}, {START_POINT_REF['lon']})")
    print(f"2. At least one track point within {TRACK_POINT_REF['radius_km']}KM of ({TRACK_POINT_REF['lat']}, {TRACK_POINT_REF['lon']})")
    print(f"3. No long gaps (>{MIN_GAP_HOURS} hours) between leaving first circle and reaching second circle")
    print(f"4. Total time between first and last point: 5-72 hours")
    print(f"5. All track points recorded between May and October")
    print(f"6. First point's timestamp is in the range of 6am to 2pm")
    print("Processing files...")
    
    for i, track_file in enumerate(track_files):
        if i % 50 == 0:
            print(f"Processed {i}/{len(track_files)} files...")
        
        track_info = parse_track_file(track_file)
        if track_info:
            all_results.append(track_info)
            
            # Check if track meets all criteria
            if track_info['meets_criteria']:
                filtered_results.append(track_info)
    
    # Write filtered results to CSV
    with open(output_file, 'w', newline='') as csv_file:
        fieldnames = [
            'file', 'total_distance', 
            'first_point_lat', 'first_point_lon', 'first_point_time',
            'last_point_in_first_circle_lat', 'last_point_in_first_circle_lon', 'last_point_in_first_circle_time',
            'first_point_in_second_circle_lat', 'first_point_in_second_circle_lon', 'first_point_in_second_circle_time',
            'time_between_circles_hours',
            'last_point_lat', 'last_point_lon', 'last_point_time',
            'total_time_gap_hours', 'num_points'
        ]
        
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in filtered_results:
            last_point_in_first = result['last_point_in_first_circle_before_second_data']
            first_point_in_second = result['first_point_in_second_circle']
            
            writer.writerow({
                'file': result['file'],
                'total_distance': result['total_distance'],
                'first_point_lat': result['first_point']['lat'],
                'first_point_lon': result['first_point']['lon'],
                'first_point_time': format_datetime(result['first_point'].get('timestamp')),
                'last_point_in_first_circle_lat': last_point_in_first['lat'] if last_point_in_first else None,
                'last_point_in_first_circle_lon': last_point_in_first['lon'] if last_point_in_first else None,
                'last_point_in_first_circle_time': format_datetime(last_point_in_first.get('timestamp')) if last_point_in_first else None,
                'first_point_in_second_circle_lat': first_point_in_second['lat'] if first_point_in_second else None,
                'first_point_in_second_circle_lon': first_point_in_second['lon'] if first_point_in_second else None,
                'first_point_in_second_circle_time': format_datetime(first_point_in_second.get('timestamp')) if first_point_in_second else None,
                'time_between_circles_hours': result['time_between_circles'],
                'last_point_lat': result['last_point']['lat'],
                'last_point_lon': result['last_point']['lon'],
                'last_point_time': format_datetime(result['last_point'].get('timestamp')),
                'total_time_gap_hours': result['total_time_gap_hours'],
                'num_points': result['num_points']
            })
    
    # Print summary
    print(f"\nSummary:")
    print(f"Total track files: {len(track_files)}")
    print(f"Files with valid data: {len(all_results)}")
    print(f"Files with first point in radius: {len([r for r in all_results if r['first_point_in_radius']])}")
    print(f"Files with track point in second radius: {len([r for r in all_results if r['has_point_in_second_radius']])}")
    
    # Count files with long gaps between circles
    files_with_long_gaps_between_circles = len([r for r in all_results if r.get('has_long_gap_between_circles')])
    print(f"Files with long gaps between circles (>{MIN_GAP_HOURS} hours): {files_with_long_gaps_between_circles}")
    
    print(f"Files with time gap between 5-72 hours: {len([r for r in all_results if r.get('total_time_gap_hours') is not None and 5 <= r['total_time_gap_hours'] <= 72])}")
    print(f"Files with all points in season (May-Oct): {len([r for r in all_results if r.get('all_points_in_season', False)])}")
    print(f"Files with first point in time range (6am-2pm): {len([r for r in all_results if r.get('first_point_in_time_range', False)])}")
    
    print(f"Files meeting all criteria: {len(filtered_results)}")
    
    if filtered_results:
        # Calculate statistics on the time between circles
        times_between_circles = [r['time_between_circles'] for r in filtered_results if r['time_between_circles'] is not None]
        if times_between_circles:
            avg_time = sum(times_between_circles) / len(times_between_circles)
            max_time = max(times_between_circles)
            min_time = min(times_between_circles)
            
            print(f"\nTime between circles statistics:")
            print(f"Average time: {avg_time:.2f} hours")
            print(f"Maximum time: {max_time:.2f} hours")
            print(f"Minimum time: {min_time:.2f} hours")
        
        # Sort by time between circles
        filtered_results.sort(key=lambda x: x.get('time_between_circles', float('inf')))
        
        print("\nTop 5 tracks with shortest time between circles:")
        for i, result in enumerate(filtered_results[:5]):
            time = result.get('time_between_circles')
            if time is not None:
                print(f"{i+1}. {result['file']}: {time:.2f} hours between circles")
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
