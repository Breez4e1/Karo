import os
import math
import logging
import joblib
from datetime import datetime
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 分段坐标：5个经纬度点，将轨迹分为6段
split_points = [
    (81.25838, 30.99363),
    (81.25956, 31.01228),
    (81.26942, 31.03033),
    (81.28070, 31.05266),
    (81.28729, 31.08469)
]

# 每段的难度等级（数值越大越困难）
segment_difficulties = [1, 0, 2, 4, 5, 3]

# Configuration parameters
CONFIG = {
    'data_dir': './filtered_tracks',
    'model_output_path': 'flexible_time_model.pkl',
    'processed_data_path': 'processed_hiking_data.csv',
    'test_size': 0.2,
    'random_state': 42,
    'typical_hike_duration': 6 * 3600,  # 6 hours in seconds
    'xgb_params': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'reg:squarederror'
    }
}

def haversine(lon1, lat1, lon2, lat2):
    """Calculate the great-circle distance between two points on Earth.
    
    Args:
        lon1, lat1, lon2, lat2: Coordinates in decimal degrees
        
    Returns:
        Distance in meters
    """
    R = 6371000  # Earth radius in meters
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    return 2 * R * math.asin(math.sqrt(a))

def find_segment_index(lon, lat):
    """Find which segment a coordinate belongs to based on proximity to split points.
    
    Args:
        lon: Longitude in decimal degrees
        lat: Latitude in decimal degrees
        
    Returns:
        Segment index (0-5)
    """
    for i, (s_lon, s_lat) in enumerate(split_points):
        dist = haversine(lon, lat, s_lon, s_lat)
        if dist < 1000:  # 若距离小于1公里，归为该段
            return i
    # If no segment is found, return the index for the last segment
    return len(split_points)  # This will be 5 for the current configuration

def parse_file(filepath):
    """Parse track file to extract coordinates and timestamps.
    
    Args:
        filepath: Path to the track file
        
    Returns:
        Tuple of (coordinates, timestamps) lists
    """
    coords, times = [], []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                try:
                    if line.startswith("Coord1:"):
                        parts = line.split()
                        if len(parts) >= 4:  # Ensure we have enough parts
                            lon, lat, ele = float(parts[1]), float(parts[2]), float(parts[3])
                            coords.append((lon, lat, ele))
                    elif line.startswith("Timestamp1:"):
                        t = line.split("Timestamp1:")[1].strip()
                        times.append(datetime.fromisoformat(t.replace("Z", "+00:00")))
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing line in {filepath}: {line}. Error: {e}")
                    continue
    except Exception as e:
        logger.error(f"Error reading file {filepath}: {e}")
        return [], []
        
    # Ensure coordinates and timestamps are aligned
    if len(coords) != len(times):
        logger.warning(f"Mismatched coordinates and timestamps in {filepath}: {len(coords)} coords vs {len(times)} timestamps")
        # Use the shorter length to avoid index errors
        min_len = min(len(coords), len(times))
        coords = coords[:min_len]
        times = times[:min_len]
        
    return coords, times

def calculate_time_to_destination(current_idx, coords, times, destination):
    """Calculate the actual time it took to reach a destination from the current position.
    
    Args:
        current_idx: Index of current position in coords/times lists
        coords: List of coordinates for the track
        times: List of timestamps for the track
        destination: (lon, lat) tuple of destination
        
    Returns:
        Time in seconds to reach destination, or None if destination not reached
    """
    # Add optimizations to avoid excessive calculations for very long tracks
    max_points_to_check = 500  # Don't check more than 500 points ahead
    max_search_distance = 10000  # Don't look beyond 10km from current position
    dest_lon, dest_lat = destination
    
    # Find the closest point to the destination in the remaining track
    min_dist = float('inf')
    closest_idx = None
    
    # Limit the search range
    end_idx = min(current_idx + max_points_to_check, len(coords))
    
    for i in range(current_idx + 1, end_idx):
        lon, lat, _ = coords[i]
        
        # First check if we're too far from current position
        if i > current_idx + 10:  # Skip this check for the first few points
            dist_from_current = haversine(coords[current_idx][0], coords[current_idx][1], lon, lat)
            if dist_from_current > max_search_distance:
                break  # Stop searching if we're too far from current position
        
        # Calculate distance to destination
        dist = haversine(lon, lat, dest_lon, dest_lat)
        if dist < min_dist:
            min_dist = dist
            closest_idx = i
            # If we're very close to destination, stop searching
            if dist < 100:  # Within 100 meters
                break
    
    # If we found a point close to destination
    if closest_idx is not None and min_dist < 1000:  # Within 1km
        time_diff = (times[closest_idx] - times[current_idx]).total_seconds()
        # Sanity check - time shouldn't be more than 24 hours
        if time_diff > 24*3600:
            return None
        return max(time_diff, 0)  # Ensure non-negative
    
    return None  # Destination not reached

def build_dataset(directory):
    """Build a dataset from track files for model training with flexible destinations.
    
    Args:
        directory: Directory containing track files
        
    Returns:
        DataFrame with features and target variable
    """
    all_rows = []
    file_count = 0
    processed_count = 0
    total_points = 0
    progress_interval = 10  # Show progress more frequently
    
    # Set maximum processing time per file to avoid getting stuck
    max_file_processing_time = 300  # seconds
    
    # Set a limit on the number of points to process per file
    max_points_per_file = 5000
    
    try:
        files = os.listdir(directory)
    except FileNotFoundError:
        logger.error(f"Directory not found: {directory}")
        return pd.DataFrame()
        
    logger.info(f"Found {len(files)} files in {directory}")
    
    # Create a list of all possible destinations (split points + endpoint)
    # We'll add the endpoint as we process each track
    
    for filename in files:
        if not filename.endswith('track'):
            continue
            
        file_count += 1
        # Show progress periodically
        if file_count % progress_interval == 0 or file_count == len(files):
            logger.info(f"Processing file {file_count}/{len(files)} ({(file_count/len(files)*100):.1f}%) - {filename}")
            
        filepath = os.path.join(directory, filename)
        
        # Set a timeout for processing this file
        file_start_time = datetime.now()
        
        coords, times = parse_file(filepath)
        
        if len(coords) < 10:
            logger.debug(f"Skipping {filename}: too few coordinates ({len(coords)})")
            continue
        
        # Limit number of points for very large files
        if len(coords) > max_points_per_file:
            logger.info(f"Limiting {filename} from {len(coords)} to {max_points_per_file} points")
            # Take evenly spaced points
            indices = np.linspace(0, len(coords)-1, max_points_per_file, dtype=int)
            coords = [coords[i] for i in indices]
            times = [times[i] for i in indices]
            
        processed_count += 1
        start_time = times[0]
        start_hour = start_time.hour
        
        # Extract month and determine season
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
            
        # We're not using weekend information as per user request

        # Create all possible destinations for this track
        # For each possible destination
        # Use only split points that are ahead of the starting point
        destinations = []
        start_lon, start_lat = coords[0][0], coords[0][1]
        
        # Add only split points that are at least 1km ahead of the start
        for sp in split_points:
            if haversine(start_lon, start_lat, sp[0], sp[1]) > 1000:
                destinations.append(sp)
                
        # Always add the endpoint
        destinations.append((coords[-1][0], coords[-1][1]))
        
        # Process each point in the track
        # Check if we've spent too much time on this file
        if (datetime.now() - file_start_time).total_seconds() > max_file_processing_time:
            logger.warning(f"Timeout processing {filename} after {max_file_processing_time} seconds")
            continue
            
        # Use a subset of points for very long tracks to speed up processing
        point_step = max(1, len(coords) // 500)  # Process at most 500 points per track
        
        for i in range(1, len(coords) - 1, point_step):
            try:
                # Check again for timeout within the loop
                if (datetime.now() - file_start_time).total_seconds() > max_file_processing_time:
                    logger.warning(f"Timeout processing point {i} in {filename}")
                    break
                    
                lon, lat, ele = coords[i]
                lon_prev, lat_prev, ele_prev = coords[i-1]
                t_cur = times[i]
                t_prev = times[i-1]

                # Calculate time difference, ensure it's at least 0.1 seconds to avoid division by zero
                delta_t = max((t_cur - t_prev).total_seconds(), 0.1)
                speed = haversine(lon_prev, lat_prev, lon, lat) / delta_t

                # Calculate gradient with a small epsilon to avoid division by zero
                horiz_dist = haversine(lon_prev, lat_prev, lon, lat)
                epsilon = 0.001  # 1mm, much smaller than the previous 1m threshold
                gradient = (ele - ele_prev) / max(horiz_dist, epsilon)

                # Cap gradient to reasonable values (-1.0 to 1.0 or -45° to 45°)
                gradient = max(min(gradient, 1.0), -1.0)

                seg_idx = find_segment_index(lon, lat)
                # Ensure segment index is within bounds of segment_difficulties
                if seg_idx < len(segment_difficulties):
                    diff_level = segment_difficulties[seg_idx]
                else:
                    diff_level = segment_difficulties[-1]  # Use the last difficulty level as default
                
                # Skip destinations that are behind us
                # For each possible destination
                for dest_idx, destination in enumerate(destinations):
                    # Calculate distance to this destination
                    dest_lon, dest_lat = destination
                    dist_to_dest = haversine(lon, lat, dest_lon, dest_lat)
                    
                    # Skip if destination is too far (over 20km) - likely not relevant
                    if dist_to_dest > 20000:
                        continue
                    
                    # Calculate actual time to reach this destination
                    time_to_dest = calculate_time_to_destination(i, coords, times, destination)
                    
                    # Skip if we couldn't calculate time to destination
                    if time_to_dest is None:
                        continue
                        
                    # Skip unreasonable times (more than 12 hours to reach destination)
                    if time_to_dest > 12 * 3600:
                        continue
                    
                    # Create row with destination-specific features
                    row = {
                        'speed': speed,
                        'dist_to_dest': dist_to_dest,
                        'gradient': gradient,
                        'time_of_day': start_hour,
                        'difficulty_level': diff_level,
                        'segment_index': seg_idx,
                        'season': season,
                        'destination_idx': dest_idx,  # Which destination (0-5 for split points, 6 for endpoint)
                        'time_to_destination': time_to_dest,
                        'file': filename  # Add filename for debugging
                    }
                    all_rows.append(row)
                    total_points += 1
                    
                    # Show progress for large number of points
                    if total_points % 100000 == 0:
                        logger.info(f"Generated {total_points} data points so far from {processed_count} files")
            except Exception as e:
                logger.warning(f"Error processing point {i} in {filename}: {e}")
                continue

    logger.info(f"Processed {processed_count}/{file_count} track files, generated {len(all_rows)} data points")
    
    if not all_rows:
        logger.warning("No data points were generated. Check your input files.")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(all_rows)
    
    # Perform some basic data cleaning
    logger.info("Cleaning dataset...")
    
    # Remove extreme outliers in time_to_destination
    q1 = df['time_to_destination'].quantile(0.01)
    q3 = df['time_to_destination'].quantile(0.99)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Filter out extreme outliers
    df_filtered = df[(df['time_to_destination'] >= max(0, lower_bound)) & 
                      (df['time_to_destination'] <= upper_bound)]
    
    logger.info(f"Removed {len(df) - len(df_filtered)} outliers from dataset")
    
    return df_filtered

def evaluate_model_quality(model, X_test, y_test, typical_hike_duration=6*3600):
    """Comprehensive model evaluation for hiking time prediction.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target values
        typical_hike_duration: Typical duration of a hike in seconds (default: 6 hours)
        
    Returns:
        Dictionary of evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    # Basic metrics
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': math.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred)
    }
    
    # Error as percentage of typical duration
    metrics['error_percentage'] = (metrics['MAE'] / typical_hike_duration) * 100
    
    # Directional bias
    errors = y_test - y_pred
    metrics['mean_error'] = np.mean(errors)  # Positive means underestimation
    
    # Time threshold accuracy
    abs_errors = np.abs(errors)
    for minutes in [5, 10, 15, 30, 60]:
        metrics[f'within_{minutes}min'] = np.mean(abs_errors < (minutes * 60)) * 100
    
    # Percentage threshold accuracy
    for pct in [5, 10, 15, 20]:
        threshold = (pct/100) * y_test
        metrics[f'within_{pct}pct'] = np.mean(abs_errors < threshold) * 100
    
    # Baseline comparison - simple distance-based prediction
    if 'dist_to_dest' in X_test.columns:
        # Convert to numpy array if it's a pandas Series
        dist_to_dest = X_test['dist_to_dest'].values if hasattr(X_test['dist_to_dest'], 'values') else X_test['dist_to_dest']
        
        # Calculate average speed (m/s) for non-zero distances
        valid_indices = dist_to_dest > 0
        if np.any(valid_indices):
            avg_speed = np.mean(y_test[valid_indices] / dist_to_dest[valid_indices])
            baseline_pred = dist_to_dest * avg_speed
            metrics['baseline_MAE'] = mean_absolute_error(y_test, baseline_pred)
            metrics['improvement_over_baseline'] = (1 - metrics['MAE']/metrics['baseline_MAE']) * 100
    
    return metrics

def evaluate_by_destination(model, X_test, y_test):
    """Evaluate model performance separately for each destination.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target values
        
    Returns:
        Dictionary of metrics by destination
    """
    metrics_by_dest = {}
    
    # Get unique destination indices
    dest_indices = X_test['destination_idx'].unique()
    
    for dest_idx in dest_indices:
        # Filter data for this destination
        mask = X_test['destination_idx'] == dest_idx
        X_dest = X_test[mask]
        y_dest = y_test[mask]
        
        # Skip if not enough data
        if len(y_dest) < 10:
            logger.warning(f"Not enough data for destination {dest_idx}, skipping evaluation")
            continue
        
        # Predict and evaluate
        y_pred = model.predict(X_dest)
        
        dest_name = f"split_point_{dest_idx}" if dest_idx < len(split_points) else "endpoint"
        
        metrics_by_dest[dest_name] = {
            'count': len(y_dest),
            'MAE': mean_absolute_error(y_dest, y_pred),
            'RMSE': math.sqrt(mean_squared_error(y_dest, y_pred)),
            'R2': r2_score(y_dest, y_pred)
        }
    
    return metrics_by_dest

def main():
    """Main function to build dataset, train model and evaluate performance."""
    start_time = datetime.now()
    logger.info(f"Starting flexible time prediction model training at {start_time.strftime('%H:%M:%S')}")
    
    # Get configuration parameters
    directory = CONFIG['data_dir']
    test_size = CONFIG['test_size']
    random_state = CONFIG['random_state']
    xgb_params = CONFIG['xgb_params']
    model_output_path = CONFIG['model_output_path']
    processed_data_path = CONFIG['processed_data_path']
    typical_hike_duration = CONFIG['typical_hike_duration']
    
    # Build dataset
    logger.info(f"Building dataset from {directory}")
    logger.info("This may take some time depending on the number of files...")
    df = build_dataset(directory)
    
    dataset_time = datetime.now()
    logger.info(f"Dataset building completed in {(dataset_time - start_time).total_seconds():.1f} seconds")
    
    if df.empty:
        logger.error("Dataset is empty. Exiting.")
        return
        
    logger.info(f"Dataset shape: {df.shape}")
    
    # Save the processed dataset to CSV
    logger.info(f"Saving processed dataset to {processed_data_path}")
    try:
        df.to_csv(processed_data_path, index=False)
        logger.info(f"Successfully saved processed data with {df.shape[0]} rows and {df.shape[1]} columns")
    except Exception as e:
        logger.error(f"Error saving processed data: {e}")
        # Continue with the script even if saving fails
    
    # Check for NaN values
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        logger.warning(f"Found {nan_count} NaN values in dataset. Cleaning...")
        df = df.dropna()
        logger.info(f"After cleaning, dataset shape: {df.shape}")
    
    # One-hot encode segment index
    logger.info("One-hot encoding segment index")
    encoder_segment = OneHotEncoder(sparse_output=False)
    seg_ohe = encoder_segment.fit_transform(df[['segment_index']])
    seg_ohe_df = pd.DataFrame(seg_ohe, columns=[f'segment_{i}' for i in range(seg_ohe.shape[1])])
    
    # One-hot encode destination index
    logger.info("One-hot encoding destination index")
    encoder_dest = OneHotEncoder(sparse_output=False)
    dest_ohe = encoder_dest.fit_transform(df[['destination_idx']])
    dest_ohe_df = pd.DataFrame(dest_ohe, columns=[f'dest_{i}' for i in range(dest_ohe.shape[1])])
    
    # Combine all features
    df_combined = pd.concat([
        df.reset_index(drop=True), 
        seg_ohe_df, 
        dest_ohe_df
    ], axis=1)

    # Build feature matrix
    features = [
        'speed', 'dist_to_dest', 'gradient', 'time_of_day', 
        'difficulty_level', 'season', 'destination_idx'
    ] + list(seg_ohe_df.columns) + list(dest_ohe_df.columns)
    
    logger.info(f"Using features: {features}")
    
    X = df_combined[features]
    y = df_combined['time_to_destination']

    # Train-test split
    logger.info(f"Splitting data with test_size={test_size}, random_state={random_state}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    logger.info(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

    # Train model
    logger.info(f"Training XGBoost model with parameters: {xgb_params}")
    logger.info("Starting model training...")
    
    # Use simple model initialization for maximum compatibility
    model = XGBRegressor(**xgb_params)
    
    # Use the most basic fit method to ensure compatibility
    logger.info("Using basic XGBoost training method for maximum compatibility")
    model.fit(X_train, y_train)
    
    training_time = datetime.now()
    logger.info(f"Model training completed in {(training_time - dataset_time).total_seconds():.1f} seconds")

    # Evaluate model
    logger.info("Evaluating overall model performance")
    metrics = evaluate_model_quality(model, X_test, y_test, typical_hike_duration)
    
    # Print metrics
    logger.info("Model evaluation results:")
    for metric_name, metric_value in metrics.items():
        if 'within_' in metric_name or metric_name == 'error_percentage' or metric_name == 'improvement_over_baseline':
            logger.info(f"  {metric_name}: {metric_value:.2f}%")
        else:
            logger.info(f"  {metric_name}: {metric_value:.2f}")
    
    # Evaluate by destination
    logger.info("Evaluating model performance by destination")
    dest_metrics = evaluate_by_destination(model, X_test, y_test)
    
    logger.info("Performance by destination:")
    for dest_name, dest_metric in dest_metrics.items():
        logger.info(f"  {dest_name} (n={dest_metric['count']}):")
        logger.info(f"    MAE: {dest_metric['MAE']:.2f} seconds")
        logger.info(f"    RMSE: {dest_metric['RMSE']:.2f} seconds")
        logger.info(f"    R2: {dest_metric['R2']:.2f}")
    
    # Feature importance
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    logger.info("Top 10 most important features:")
    for _, row in feature_importance.head(10).iterrows():
        logger.info(f"  {row['Feature']}: {row['Importance']:.4f}")
    
    # Save model and encoders
    logger.info(f"Saving model to {model_output_path}")
    model_data = {
        'model': model,
        'segment_encoder': encoder_segment,
        'destination_encoder': encoder_dest,
        'features': features
    }
    joblib.dump(model_data, model_output_path)
    end_time = datetime.now()
    total_runtime = (end_time - start_time).total_seconds()
    logger.info(f"Model training completed successfully in {total_runtime:.1f} seconds ({total_runtime/60:.1f} minutes)")

def predict_time_to_point(model_path, current_position, destination_idx, current_speed=None):
    """Predict time to reach a specific point.
    
    Args:
        model_path: Path to the saved model
        current_position: Dictionary with current position data
            {
                'lon': longitude,
                'lat': latitude,
                'ele': elevation,
                'time_of_day': hour of day (0-23),
                'season': season (0-3)
            }
        destination_idx: Index of the destination (0-5 for split points, 6 for endpoint)
        current_speed: Current speed in m/s (optional)
        
    Returns:
        Predicted time in seconds
    """
    # Load model data
    model_data = joblib.load(model_path)
    model = model_data['model']
    segment_encoder = model_data['segment_encoder']
    destination_encoder = model_data['destination_encoder']
    features = model_data['features']
    
    # Get destination coordinates
    if destination_idx < len(split_points):
        dest_lon, dest_lat = split_points[destination_idx]
    else:
        # This is just a placeholder - in real usage you'd need the actual endpoint
        logger.warning("Using placeholder endpoint coordinates - replace with actual endpoint")
        dest_lon, dest_lat = 81.29, 31.09
    
    # Calculate distance to destination
    dist_to_dest = haversine(
        current_position['lon'], 
        current_position['lat'], 
        dest_lon, 
        dest_lat
    )
    
    # Find segment index
    segment_idx = find_segment_index(current_position['lon'], current_position['lat'])
    
    # Get difficulty level
    if segment_idx < len(segment_difficulties):
        diff_level = segment_difficulties[segment_idx]
    else:
        diff_level = segment_difficulties[-1]
    
    # Create feature vector
    feature_dict = {
        'speed': current_speed if current_speed is not None else 1.0,  # Default to 1 m/s if not provided
        'dist_to_dest': dist_to_dest,
        'gradient': 0.0,  # Would need previous point to calculate
        'time_of_day': current_position['time_of_day'],
        'difficulty_level': diff_level,
        'season': current_position['season'],
        'is_weekend': current_position['is_weekend'],
        'destination_idx': destination_idx
    }
    
    # One-hot encode segment index
    segment_ohe = segment_encoder.transform([[segment_idx]])
    for i, val in enumerate(segment_ohe[0]):
        feature_dict[f'segment_{i}'] = val
    
    # One-hot encode destination index
    dest_ohe = destination_encoder.transform([[destination_idx]])
    for i, val in enumerate(dest_ohe[0]):
        feature_dict[f'dest_{i}'] = val
    
    # Create feature array in the correct order
    X = pd.DataFrame([feature_dict])[features]
    
    # Make prediction
    predicted_time = model.predict(X)[0]
    
    return max(predicted_time, 0)  # Ensure non-negative

def load_and_train_from_csv(csv_path=None):
    """Load processed data from CSV and train the model.
    
    This function can be called separately if you've already processed the data
    and want to skip directly to model training.
    
    Args:
        csv_path: Path to the processed CSV file. If None, uses the default from CONFIG.
    """
    if csv_path is None:
        csv_path = CONFIG['processed_data_path']
        
    start_time = datetime.now()
    logger.info(f"Loading processed data from {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns from CSV")
        
        # Continue with model training (copy of the training code from main)
        test_size = CONFIG['test_size']
        random_state = CONFIG['random_state']
        xgb_params = CONFIG['xgb_params']
        model_output_path = CONFIG['model_output_path']
        typical_hike_duration = CONFIG['typical_hike_duration']
        
        # One-hot encode segment index
        logger.info("One-hot encoding segment index")
        encoder_segment = OneHotEncoder(sparse_output=False)
        seg_ohe = encoder_segment.fit_transform(df[['segment_index']])
        seg_ohe_df = pd.DataFrame(seg_ohe, columns=[f'segment_{i}' for i in range(seg_ohe.shape[1])])
        
        # One-hot encode destination index
        logger.info("One-hot encoding destination index")
        encoder_dest = OneHotEncoder(sparse_output=False)
        dest_ohe = encoder_dest.fit_transform(df[['destination_idx']])
        dest_ohe_df = pd.DataFrame(dest_ohe, columns=[f'dest_{i}' for i in range(dest_ohe.shape[1])])
        
        # Combine all features
        df_combined = pd.concat([df.reset_index(drop=True), seg_ohe_df, dest_ohe_df], axis=1)

        # Build feature matrix
        features = [
            'speed', 'dist_to_dest', 'gradient', 'time_of_day', 
            'difficulty_level', 'season', 'destination_idx'
        ] + list(seg_ohe_df.columns) + list(dest_ohe_df.columns)
        
        logger.info(f"Using features: {features}")
        
        X = df_combined[features]
        y = df_combined['time_to_destination']

        # Train-test split
        logger.info(f"Splitting data with test_size={test_size}, random_state={random_state}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        logger.info(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

        # Train model
        logger.info(f"Training XGBoost model with parameters: {xgb_params}")
        logger.info("Starting model training...")
        
        # Use simple model initialization for maximum compatibility
        model = XGBRegressor(**xgb_params)
        
        # Use the most basic fit method to ensure compatibility
        logger.info("Using basic XGBoost training method for maximum compatibility")
        model.fit(X_train, y_train)
        
        # Evaluate model
        logger.info("Evaluating overall model performance")
        metrics = evaluate_model_quality(model, X_test, y_test, typical_hike_duration)
        
        # Print metrics
        logger.info("Model evaluation results:")
        for metric_name, metric_value in metrics.items():
            if 'within_' in metric_name or metric_name == 'error_percentage' or metric_name == 'improvement_over_baseline':
                logger.info(f"  {metric_name}: {metric_value:.2f}%")
            else:
                logger.info(f"  {metric_name}: {metric_value:.2f}")
        
        # Evaluate by destination
        logger.info("Evaluating model performance by destination")
        dest_metrics = evaluate_by_destination(model, X_test, y_test)
        
        logger.info("Performance by destination:")
        for dest_name, dest_metric in dest_metrics.items():
            logger.info(f"  {dest_name} (n={dest_metric['count']}):")
            logger.info(f"    MAE: {dest_metric['MAE']:.2f} seconds")
            logger.info(f"    RMSE: {dest_metric['RMSE']:.2f} seconds")
            logger.info(f"    R2: {dest_metric['R2']:.2f}")
        
        # Feature importance
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        logger.info("Top 10 most important features:")
        for _, row in feature_importance.head(10).iterrows():
            logger.info(f"  {row['Feature']}: {row['Importance']:.4f}")
        
        # Save model and encoders
        logger.info(f"Saving model to {model_output_path}")
        model_data = {
            'model': model,
            'segment_encoder': encoder_segment,
            'destination_encoder': encoder_dest,
            'features': features
        }
        joblib.dump(model_data, model_output_path)
        logger.info("Model training completed successfully")
        
        end_time = datetime.now()
        total_runtime = (end_time - start_time).total_seconds()
        logger.info(f"Training from CSV completed in {total_runtime:.1f} seconds ({total_runtime/60:.1f} minutes)")
        
        return model, metrics
    except Exception as e:
        logger.error(f"Error in load_and_train_from_csv: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    try:
        # Check if processed data file already exists
        if os.path.exists(CONFIG['processed_data_path']):
            logger.info(f"Found existing processed data at {CONFIG['processed_data_path']}")
            user_input = input("Use existing processed data? (y/n): ").strip().lower()
            if user_input == 'y':
                load_and_train_from_csv()
            else:
                main()
        else:
            main()
    except Exception as e:
        logger.error(f"An error occurred in main execution: {e}", exc_info=True)
        print(f"Error: {e}. See log for details.")
        
    print("\nTo train using the saved processed data without reprocessing all files, run:")
    print("python -c 'from flexible_time_predictor import load_and_train_from_csv; load_and_train_from_csv()'")
