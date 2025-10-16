import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
import logging
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'model_path': 'flexible_time_model.pkl',
    'processed_data_path': 'processed_hiking_data.csv',
    'output_dir': 'visualization_results',
    'sample_size': 1000  # Number of samples to use for SHAP analysis (for speed)
}

def load_data_and_model():
    """Load the processed data and trained model"""
    logger.info(f"Loading data from {CONFIG['processed_data_path']}...")
    if not os.path.exists(CONFIG['processed_data_path']):
        logger.error(f"Data file not found: {CONFIG['processed_data_path']}")
        return None, None
    
    df = pd.read_csv(CONFIG['processed_data_path'])
    logger.info(f"Loaded {len(df)} data points")
    
    logger.info(f"Loading model from {CONFIG['model_path']}...")
    if not os.path.exists(CONFIG['model_path']):
        logger.error(f"Model file not found: {CONFIG['model_path']}")
        return df, None
    
    model_data = joblib.load(CONFIG['model_path'])
    logger.info("Model loaded successfully")
    
    return df, model_data

def prepare_data_for_shap(df, model_data):
    """Prepare data for SHAP analysis by recreating one-hot encoded features"""
    # Get model components
    features = model_data['features']
    segment_encoder = model_data['segment_encoder']
    destination_encoder = model_data['destination_encoder']
    
    logger.info("Recreating one-hot encoded features...")
    
    # Check if we need to recreate one-hot encoded features
    missing_features = [f for f in features if f not in df.columns]
    
    if missing_features:
        logger.info(f"Recreating {len(missing_features)} missing features")
        
        # Create segment one-hot encoded features
        if 'segment_index' in df.columns:
            # Get unique segment indices
            segment_indices = df['segment_index'].values.reshape(-1, 1)
            # Transform using the encoder from the model
            segment_encoded = segment_encoder.transform(segment_indices)
            # Add to dataframe
            for i in range(segment_encoded.shape[1]):
                df[f'segment_{i}'] = segment_encoded[:, i]
        else:
            logger.error("'segment_index' column not found in dataframe")
            return None
        
        # Create destination one-hot encoded features
        if 'destination_idx' in df.columns:
            # Get unique destination indices
            dest_indices = df['destination_idx'].values.reshape(-1, 1)
            # Transform using the encoder from the model
            dest_encoded = destination_encoder.transform(dest_indices)
            # Add to dataframe
            for i in range(dest_encoded.shape[1]):
                df[f'dest_{i}'] = dest_encoded[:, i]
        else:
            logger.error("'destination_idx' column not found in dataframe")
            return None
    
    # Check again if all features are available
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        logger.error(f"Still missing features after encoding: {missing_features}")
        return None
    
    # Select features used by the model
    X = df[features]
    
    # Sample data for faster SHAP analysis
    if len(X) > CONFIG['sample_size']:
        X = X.sample(CONFIG['sample_size'], random_state=42)
        logger.info(f"Sampled {CONFIG['sample_size']} data points for SHAP analysis")
    
    return X

def create_shap_summary_plot(model, X, output_path):
    """Create and save SHAP summary plot"""
    logger.info("Calculating SHAP values...")
    
    # Create explainer
    explainer = shap.Explainer(model)
    
    # Calculate SHAP values
    shap_values = explainer(X)
    
    # Create summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.title("SHAP Feature Importance", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"SHAP summary plot saved to {output_path}")

def create_shap_dependence_plots(model, X, output_dir):
    """Create and save SHAP dependence plots for key features"""
    logger.info("Creating SHAP dependence plots...")
    
    # Create explainer
    explainer = shap.Explainer(model)
    
    # Calculate SHAP values
    shap_values = explainer(X)
    
    # Key features to analyze
    key_features = [
        'dist_to_dest',
        'difficulty_level',
        'gradient',
        'speed',
        'time_of_day',
        'season',
        'destination_idx'
    ]
    
    # Create dependence plots for each key feature
    for feature in key_features:
        if feature in X.columns:
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(
                feature, 
                shap_values.values, 
                X, 
                show=False,
                interaction_index=None
            )
            plt.title(f"SHAP Dependence Plot for {feature}", fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"shap_dependence_{feature}.png"), dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"SHAP dependence plot for {feature} saved")

def create_segment_specific_shap_analysis(df, model_data):
    """Create SHAP analysis for each segment separately"""
    logger.info("Performing segment-specific SHAP analysis...")
    
    # Get model and features
    model = model_data['model']
    features = model_data['features']
    
    # Create output directory
    os.makedirs(os.path.join(CONFIG['output_dir'], 'segment_shap'), exist_ok=True)
    
    # Analyze each segment
    for segment in range(6):  # 6 segments (0-5)
        logger.info(f"Analyzing segment {segment}...")
        
        # Filter data for this segment
        segment_data = df[df['segment_index'] == segment]
        
        # Skip if not enough data
        if len(segment_data) < 100:
            logger.warning(f"Not enough data for segment {segment}, skipping")
            continue
        
        # Sample data if too large
        if len(segment_data) > CONFIG['sample_size']:
            segment_data = segment_data.sample(CONFIG['sample_size'], random_state=42)
        
        # Select features
        X_segment = segment_data[features]
        
        # Create explainer
        explainer = shap.Explainer(model)
        
        # Calculate SHAP values
        shap_values = explainer(X_segment)
        
        # Create summary plot for this segment
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, 
            X_segment, 
            show=False, 
            max_display=10
        )
        plt.title(f"SHAP Feature Importance for Segment {segment}", fontsize=16)
        plt.tight_layout()
        plt.savefig(
            os.path.join(CONFIG['output_dir'], 'segment_shap', f"shap_summary_segment_{segment}.png"), 
            dpi=300, 
            bbox_inches='tight'
        )
        plt.close()
        
        logger.info(f"SHAP summary plot for segment {segment} saved")

def create_shap_waterfall_plot(model, X, output_dir):
    """Create waterfall plots for individual predictions to show feature contributions"""
    logger.info("Creating SHAP waterfall plots for sample predictions...")
    
    # Create explainer
    explainer = shap.Explainer(model)
    
    # Calculate SHAP values
    shap_values = explainer(X)
    
    # Create waterfall plots for a few sample instances
    for i in range(min(5, len(X))):
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(shap_values[i], max_display=10, show=False)
        plt.title(f"SHAP Waterfall Plot for Sample Prediction {i+1}", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"shap_waterfall_sample_{i+1}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
    logger.info(f"SHAP waterfall plots saved to {output_dir}")

def create_shap_force_plot(model, X, output_dir):
    """Create force plots to show feature contributions for individual predictions"""
    logger.info("Creating SHAP force plots...")
    
    try:
        # Try TreeExplainer first (better for tree models like XGBoost)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Create individual force plots for each sample
        for i in range(min(5, len(X))):
            plt.figure(figsize=(20, 3))
            shap.force_plot(explainer.expected_value, shap_values[i], X.iloc[i], 
                          matplotlib=True, show=False)
            plt.title(f"SHAP Force Plot for Sample {i+1}", fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"shap_force_plot_sample_{i+1}.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"SHAP force plots saved as individual PNG files to {output_dir}")
        
    except Exception as e:
        logger.warning(f"Could not create force plots using TreeExplainer: {e}")
        logger.info("Trying alternative approach with KernelExplainer...")
        
        try:
            # Try KernelExplainer as fallback
            background = X.iloc[:100]  # Use first 100 instances as background
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(X.iloc[:5])  # Just use 5 samples
            
            for i in range(min(5, len(shap_values))):
                plt.figure(figsize=(20, 3))
                shap.force_plot(explainer.expected_value, shap_values[i], X.iloc[i], 
                              matplotlib=True, show=False)
                plt.title(f"SHAP Force Plot for Sample {i+1}", fontsize=14)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"shap_force_plot_sample_{i+1}.png"), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
            logger.info(f"SHAP force plots saved as individual PNG files to {output_dir}")
        except Exception as e:
            logger.error(f"Failed to create force plots with KernelExplainer: {e}")
            logger.info("Skipping force plots generation")


def main():
    """Main function to run SHAP analysis"""
    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Load data and model
    df, model_data = load_data_and_model()
    if df is None or model_data is None:
        logger.error("Failed to load data or model")
        return
    
    # Get model
    model = model_data['model']
    
    # Prepare data for SHAP analysis
    X = prepare_data_for_shap(df, model_data)
    if X is None:
        logger.error("Failed to prepare data for SHAP analysis")
        return
    
    # Create SHAP summary plot
    create_shap_summary_plot(
        model, 
        X, 
        os.path.join(CONFIG['output_dir'], "shap_summary_plot.png")
    )
    
    # Create SHAP dependence plots
    create_shap_dependence_plots(model, X, CONFIG['output_dir'])
    
    # Create segment-specific SHAP analysis
    create_segment_specific_shap_analysis(df, model_data)
    
    # Create waterfall plots for individual predictions
    create_shap_waterfall_plot(model, X.iloc[:5], CONFIG['output_dir'])
    
    # Create force plots
    create_shap_force_plot(model, X.iloc[:10], CONFIG['output_dir'])
    
    logger.info("SHAP analysis completed successfully")

if __name__ == "__main__":
    main()
