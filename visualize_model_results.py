import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12

# Configuration
CONFIG = {
    'processed_data_path': 'processed_hiking_data.csv',
    'model_path': 'flexible_time_model.pkl',
    'output_dir': './visualization_results',
    'test_size': 0.2,
    'random_state': 42
}

def load_data_and_model():
    """Load the processed data and trained model"""
    print(f"Loading data from {CONFIG['processed_data_path']}...")
    df = pd.read_csv(CONFIG['processed_data_path'])
    
    print(f"Loading model from {CONFIG['model_path']}...")
    model_data = joblib.load(CONFIG['model_path'])
    
    return df, model_data

def prepare_data(df, model_data):
    """Prepare data for visualization"""
    # Get model and features
    model = model_data['model']
    features = model_data['features']
    
    # One-hot encode segment index
    segment_encoder = model_data['segment_encoder']
    seg_ohe = segment_encoder.transform(df[['segment_index']])
    seg_ohe_df = pd.DataFrame(seg_ohe, columns=[f'segment_{i}' for i in range(seg_ohe.shape[1])])
    
    # One-hot encode destination index
    destination_encoder = model_data['destination_encoder']
    dest_ohe = destination_encoder.transform(df[['destination_idx']])
    dest_ohe_df = pd.DataFrame(dest_ohe, columns=[f'dest_{i}' for i in range(dest_ohe.shape[1])])
    
    # Combine all features
    df_combined = pd.concat([df.reset_index(drop=True), seg_ohe_df, dest_ohe_df], axis=1)
    
    # Get features and target
    X = df_combined[features]
    y = df_combined['time_to_destination']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG['test_size'], random_state=CONFIG['random_state']
    )
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Create result dataframe
    results_df = pd.DataFrame({
        'actual': y_test.values,
        'predicted': y_pred,
        'error': y_test.values - y_pred,
        'abs_error': np.abs(y_test.values - y_pred),
        'segment_index': X_test['segment_index'].values if 'segment_index' in X_test.columns else df.loc[y_test.index, 'segment_index'].values,
        'destination_idx': X_test['destination_idx'].values if 'destination_idx' in X_test.columns else df.loc[y_test.index, 'destination_idx'].values
    })
    
    return results_df

def create_error_distribution_histogram(results_df):
    """Create histogram of error distribution"""
    plt.figure(figsize=(14, 8))
    
    # Convert errors to minutes for better readability
    errors_minutes = results_df['error'] / 60
    
    # Plot histogram
    sns.histplot(errors_minutes, bins=50, kde=True)
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Error (minutes)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Distribution of Prediction Errors', fontsize=16)
    
    # Add statistics
    mean_error = errors_minutes.mean()
    std_error = errors_minutes.std()
    plt.text(0.02, 0.95, f'Mean Error: {mean_error:.2f} minutes\nStd Dev: {std_error:.2f} minutes', 
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Save figure
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], 'error_distribution.png'), dpi=300)
    plt.close()
    
    print(f"Error distribution histogram saved to {CONFIG['output_dir']}/error_distribution.png")

def create_segment_error_barplot(results_df):
    """Create bar plot of errors by segment"""
    # Group by segment and calculate statistics
    segment_stats = results_df.groupby('segment_index')['abs_error'].agg(['mean', 'std']).reset_index()
    
    # Convert to minutes
    segment_stats['mean'] = segment_stats['mean'] / 60
    segment_stats['std'] = segment_stats['std'] / 60
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Bar plot with error bars
    bars = plt.bar(segment_stats['segment_index'], segment_stats['mean'], 
                  yerr=segment_stats['std'], capsize=10, alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Segment Index', fontsize=14)
    plt.ylabel('Mean Absolute Error (minutes)', fontsize=14)
    plt.title('Mean Absolute Error by Segment with Standard Deviation', fontsize=16)
    plt.xticks(segment_stats['segment_index'])
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=12)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], 'segment_error_barplot.png'), dpi=300)
    plt.close()
    
    print(f"Segment error bar plot saved to {CONFIG['output_dir']}/segment_error_barplot.png")

def create_destination_error_barplot(results_df):
    """Create bar plot of errors by destination"""
    # Group by destination and calculate statistics
    dest_stats = results_df.groupby('destination_idx')['abs_error'].agg(['mean', 'std']).reset_index()
    
    # Convert to minutes
    dest_stats['mean'] = dest_stats['mean'] / 60
    dest_stats['std'] = dest_stats['std'] / 60
    
    # Create destination names
    dest_names = [f"Split Point {int(i)}" if i < 5 else "Endpoint" for i in dest_stats['destination_idx']]
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Bar plot with error bars
    bars = plt.bar(dest_names, dest_stats['mean'], 
                  yerr=dest_stats['std'], capsize=10, alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Destination', fontsize=14)
    plt.ylabel('Mean Absolute Error (minutes)', fontsize=14)
    plt.title('Mean Absolute Error by Destination with Standard Deviation', fontsize=16)
    plt.xticks(rotation=45)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=12)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], 'destination_error_barplot.png'), dpi=300)
    plt.close()
    
    print(f"Destination error bar plot saved to {CONFIG['output_dir']}/destination_error_barplot.png")

def create_prediction_vs_actual_scatter(results_df):
    """Create scatter plot of predicted vs actual times"""
    plt.figure(figsize=(14, 10))
    
    # Convert to minutes for better readability
    actual_minutes = results_df['actual'] / 60
    predicted_minutes = results_df['predicted'] / 60
    
    # Calculate max value for plot limits
    max_val = max(actual_minutes.max(), predicted_minutes.max()) * 1.05
    
    # Create scatter plot
    scatter = plt.scatter(actual_minutes, predicted_minutes, 
                         alpha=0.5, c=results_df['segment_index'], cmap='viridis')
    
    # Add reference line (y=x)
    plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Actual Time (minutes)', fontsize=14)
    plt.ylabel('Predicted Time (minutes)', fontsize=14)
    plt.title('Predicted vs Actual Hiking Time', fontsize=16)
    
    # Add colorbar for segments
    cbar = plt.colorbar(scatter)
    cbar.set_label('Segment Index', fontsize=12)
    
    # Add statistics
    mae = mean_absolute_error(actual_minutes, predicted_minutes)
    r2 = np.corrcoef(actual_minutes, predicted_minutes)[0, 1]**2
    
    plt.text(0.02, 0.95, f'MAE: {mae:.2f} minutes\nR²: {r2:.3f}', 
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Set equal aspect ratio
    plt.axis('equal')
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], 'prediction_vs_actual.png'), dpi=300)
    plt.close()
    
    print(f"Prediction vs actual scatter plot saved to {CONFIG['output_dir']}/prediction_vs_actual.png")

def create_prediction_vs_actual_by_destination(results_df):
    """Create scatter plots of predicted vs actual times for each destination"""
    # Get unique destinations
    destinations = sorted(results_df['destination_idx'].unique())
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharex=True, sharey=True)
    axes = axes.flatten()
    
    # Max value for plot limits
    max_val = max(results_df['actual'].max(), results_df['predicted'].max()) / 60 * 1.05
    
    # Create scatter plot for each destination
    for i, dest in enumerate(destinations):
        if i >= len(axes):
            break
            
        # Filter data for this destination
        dest_data = results_df[results_df['destination_idx'] == dest]
        
        # Convert to minutes
        actual_minutes = dest_data['actual'] / 60
        predicted_minutes = dest_data['predicted'] / 60
        
        # Create scatter plot
        axes[i].scatter(actual_minutes, predicted_minutes, alpha=0.5)
        
        # Add reference line
        axes[i].plot([0, max_val], [0, max_val], 'r--', alpha=0.7)
        
        # Add title and stats
        dest_name = f"Split Point {int(dest)}" if dest < 5 else "Endpoint"
        mae = mean_absolute_error(actual_minutes, predicted_minutes)
        r2 = np.corrcoef(actual_minutes, predicted_minutes)[0, 1]**2
        
        axes[i].set_title(f"{dest_name}\nMAE: {mae:.2f} min, R²: {r2:.3f}")
        
        # Set equal aspect ratio
        axes[i].set_aspect('equal')
        axes[i].set_xlim(0, max_val)
        axes[i].set_ylim(0, max_val)
    
    # Add labels to figure
    fig.text(0.5, 0.04, 'Actual Time (minutes)', ha='center', fontsize=14)
    fig.text(0.04, 0.5, 'Predicted Time (minutes)', va='center', rotation='vertical', fontsize=14)
    fig.suptitle('Predicted vs Actual Hiking Time by Destination', fontsize=16)
    
    # Save figure
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    plt.savefig(os.path.join(CONFIG['output_dir'], 'prediction_vs_actual_by_destination.png'), dpi=300)
    plt.close()
    
    print(f"Prediction vs actual by destination saved to {CONFIG['output_dir']}/prediction_vs_actual_by_destination.png")

def create_feature_importance_chart(model_data):
    """Create a horizontal bar chart of feature importance"""
    # Get model and features
    model = model_data['model']
    features = model_data['features']
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    # Limit to top 15 features for readability
    if len(importance_df) > 15:
        importance_df = importance_df.head(15)
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Horizontal bar chart
    bars = plt.barh(importance_df['Feature'], importance_df['Importance'])
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.002, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', va='center')
    
    # Add labels and title
    plt.xlabel('Importance (Gain)', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.title('Feature Importance (XGBoost Gain)', fontsize=16)
    
    # Adjust layout for feature names
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(CONFIG['output_dir'], 'feature_importance.png'), dpi=300)
    plt.close()
    
    print(f"Feature importance chart saved to {CONFIG['output_dir']}/feature_importance.png")

def main():
    """Main function to create visualizations"""
    # Create output directory if it doesn't exist
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Load data and model
    df, model_data = load_data_and_model()
    
    # Prepare data for visualization
    results_df = prepare_data(df, model_data)
    
    # Create visualizations
    create_error_distribution_histogram(results_df)
    create_segment_error_barplot(results_df)
    create_destination_error_barplot(results_df)
    create_prediction_vs_actual_scatter(results_df)
    create_prediction_vs_actual_by_destination(results_df)
    create_feature_importance_chart(model_data)
    
    print("All visualizations completed!")

if __name__ == "__main__":
    main()
