import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'data_path': 'processed_hiking_data.csv',
    'model_output_path': 'dense_nn_model.h5',
    'results_output_path': 'dense_nn_results.pkl',
    'test_size': 0.2,
    'random_state': 42,
    'batch_size': 32,
    'epochs': 20,
    'patience': 5
}

def load_and_prepare_data():
    """Load and prepare data for model training"""
    logger.info(f"Loading data from {CONFIG['data_path']}...")
    
    if not os.path.exists(CONFIG['data_path']):
        logger.error(f"Data file not found: {CONFIG['data_path']}")
        return None, None, None, None
    
    # Load data
    df = pd.read_csv(CONFIG['data_path'])
    logger.info(f"Loaded {len(df)} data points")
    
    # Basic features that should be available
    numerical_features = ['dist_to_dest', 'gradient', 'speed', 'time_of_day', 'season', 'difficulty_level']
    
    # Check if all required columns exist
    missing_columns = [col for col in numerical_features if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing columns in dataframe: {missing_columns}")
        return None, None, None, None
    
    # Ensure all data is numeric
    for col in numerical_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Ensure segment_index and destination_idx are integers
    df['segment_index'] = pd.to_numeric(df['segment_index'], errors='coerce').fillna(0).astype(int)
    df['destination_idx'] = pd.to_numeric(df['destination_idx'], errors='coerce').fillna(0).astype(int)
    
    # Ensure target is numeric
    df['time_to_destination'] = pd.to_numeric(df['time_to_destination'], errors='coerce')
    
    # Drop rows with NaN values
    df = df.dropna(subset=numerical_features + ['time_to_destination'])
    logger.info(f"After cleaning, {len(df)} data points remain")
    
    # Create segment one-hot encoding manually
    segment_dummies = pd.get_dummies(df['segment_index'], prefix='segment')
    
    # Create destination one-hot encoding manually
    dest_dummies = pd.get_dummies(df['destination_idx'], prefix='dest')
    
    # Combine features
    X = pd.concat([df[numerical_features], segment_dummies, dest_dummies], axis=1)
    
    # Target variable
    y = df['time_to_destination'].values  # Convert to numpy array
    
    # Scale numerical features
    scaler = StandardScaler()
    X[numerical_features] = scaler.fit_transform(X[numerical_features])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG['test_size'], random_state=CONFIG['random_state']
    )
    
    logger.info(f"Data prepared: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    
    return X_train, X_test, y_train, y_test

def create_model(input_dim):
    """Create a simple neural network model"""
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)  # Output layer for regression
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    logger.info("Model created")
    model.summary()
    
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    """Train the model"""
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=CONFIG['patience'],
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Save model
    model.save(CONFIG['model_output_path'])
    logger.info(f"Model saved to {CONFIG['model_output_path']}")
    
    return model, history

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    # Make predictions
    y_pred = model.predict(X_test).flatten()
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"Model performance: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.2f}")
    
    # Create results dictionary
    results = {
        'y_test': y_test,
        'y_pred': y_pred,
        'metrics': {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
    }
    
    # Save results
    joblib.dump(results, CONFIG['results_output_path'])
    logger.info(f"Results saved to {CONFIG['results_output_path']}")
    
    return results

def plot_results(history, results):
    """Plot training history and predictions"""
    # Create output directory if it doesn't exist
    os.makedirs('nn_results', exist_ok=True)
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('nn_results/training_history.png')
    plt.close()
    
    # Plot predictions
    plt.figure(figsize=(10, 6))
    
    plt.scatter(results['y_test'], results['y_pred'], alpha=0.5)
    
    # Plot perfect prediction line
    min_val = min(np.min(results['y_test']), np.min(results['y_pred']))
    max_val = max(np.max(results['y_test']), np.max(results['y_pred']))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Add metrics text
    metrics = results['metrics']
    plt.text(
        0.05, 0.95,
        f"MAE: {metrics['mae']:.2f}\nRMSE: {metrics['rmse']:.2f}\nR²: {metrics['r2']:.2f}",
        transform=plt.gca().transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
    )
    
    plt.title('Neural Network Model: Actual vs Predicted Hiking Times')
    plt.xlabel('Actual Time (minutes)')
    plt.ylabel('Predicted Time (minutes)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('nn_results/predictions.png')
    plt.close()
    
    logger.info("Plots saved to 'nn_results/' directory")

def compare_with_xgboost():
    """Compare Neural Network with XGBoost results"""
    # Load NN results
    nn_results_path = CONFIG['results_output_path']
    if not os.path.exists(nn_results_path):
        logger.error(f"Neural Network results not found at {nn_results_path}")
        return
    
    nn_results = joblib.load(nn_results_path)
    
    # Load XGBoost model to extract results
    xgboost_model_path = 'flexible_time_model.pkl'
    if not os.path.exists(xgboost_model_path):
        logger.warning(f"XGBoost model not found at {xgboost_model_path}")
        return
    
    try:
        # Try to load the model
        xgboost_model = joblib.load(xgboost_model_path)
        
        # Check if test data is available in the model
        if not isinstance(xgboost_model, dict) or 'test_data' not in xgboost_model:
            logger.warning("XGBoost test data not found in model, skipping comparison")
            return
        
        # Extract XGBoost test data
        xgboost_test_data = xgboost_model['test_data']
        xgboost_y_test = xgboost_test_data['y_test']
        xgboost_y_pred = xgboost_test_data['y_pred']
        
        # Calculate XGBoost metrics
        xgboost_mae = mean_absolute_error(xgboost_y_test, xgboost_y_pred)
        xgboost_rmse = np.sqrt(mean_squared_error(xgboost_y_test, xgboost_y_pred))
        xgboost_r2 = r2_score(xgboost_y_test, xgboost_y_pred)
        
        # Create comparison bar chart
        plt.figure(figsize=(10, 6))
        
        metrics = ['MAE', 'RMSE', '1 - R²']
        nn_values = [
            nn_results['metrics']['mae'],
            nn_results['metrics']['rmse'],
            1 - nn_results['metrics']['r2']
        ]
        xgboost_values = [
            xgboost_mae,
            xgboost_rmse,
            1 - xgboost_r2
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, nn_values, width, label='Neural Network', color='blue', alpha=0.7)
        plt.bar(x + width/2, xgboost_values, width, label='XGBoost', color='green', alpha=0.7)
        
        plt.title('Model Performance Comparison (Lower is Better)')
        plt.xticks(x, metrics)
        plt.ylabel('Error')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('nn_results/model_comparison.png')
        plt.close()
        
        logger.info("Model comparison plot saved to 'nn_results/model_comparison.png'")
        
        # Save comparison results
        comparison_results = {
            'neural_network': {
                'metrics': nn_results['metrics']
            },
            'xgboost': {
                'metrics': {
                    'mae': xgboost_mae,
                    'rmse': xgboost_rmse,
                    'r2': xgboost_r2
                }
            }
        }
        
        joblib.dump(comparison_results, 'nn_results/model_comparison.pkl')
        logger.info("Comparison results saved to 'nn_results/model_comparison.pkl'")
        
    except Exception as e:
        logger.error(f"Error comparing with XGBoost: {e}")

def main():
    """Main function"""
    logger.info("Starting Neural Network model training and evaluation")
    
    # Load and prepare data
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    if X_train is None:
        return
    
    # Create validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=CONFIG['random_state']
    )
    
    # Create model
    input_dim = X_train.shape[1]
    model = create_model(input_dim)
    
    # Train model
    model, history = train_model(model, X_train, y_train, X_val, y_val)
    
    # Evaluate model
    results = evaluate_model(model, X_test, y_test)
    
    # Plot results
    plot_results(history, results)
    
    # Compare with XGBoost
    compare_with_xgboost()
    
    logger.info("Neural Network model training and evaluation completed")

if __name__ == "__main__":
    main()
