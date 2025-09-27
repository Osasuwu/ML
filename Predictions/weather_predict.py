import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pickle
import json
from datetime import datetime, timedelta
import os
from weather_data_generator import WeatherDataGenerator

class WeatherLSTM(nn.Module):
    """LSTM model for weather prediction - same architecture as training"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(WeatherLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the output from the last time step
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class WeatherPredictor:
    def __init__(self, model_name=None, model_path=None, scaler_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None
        self.model_name = model_name
        self.sequence_length = 30  # Will be updated from model name
        self.feature_names = ['temperature', 'humidity', 'pressure', 'wind_speed', 'precipitation']
        
        # If model_name is provided, construct paths automatically
        if model_name:
            model_dir = f'Predictions/models/{model_name}'
            model_path = f'{model_dir}/{model_name}_best.pth'
            scaler_path = f'{model_dir}/{model_name}_scaler.pkl'
            # Parse model parameters from name
            self._parse_model_parameters(model_name)
        
        # Load model and scaler
        self.load_model(model_path, scaler_path)
    
    def _parse_model_parameters(self, model_name):
        """Parse model parameters from model name"""
        try:
            # Parse format: weather_seq30_h128_l2_lr0.001_drop0.3
            parts = model_name.split('_')
            
            # Set defaults
            self.sequence_length = 30
            self.hidden_size = 64
            self.num_layers = 2
            self.dropout = 0.2
            
            for part in parts:
                if part.startswith('seq') and len(part) > 3:
                    self.sequence_length = int(part[3:])
                elif part.startswith('h') and len(part) > 1 and part[1:].isdigit():
                    self.hidden_size = int(part[1:])
                elif part.startswith('l') and len(part) > 1 and part[1:].isdigit():
                    self.num_layers = int(part[1:])
                elif part.startswith('drop') and len(part) > 4:
                    self.dropout = float(part[4:])
            
            print(f"Parsed parameters: seq={self.sequence_length}, hidden={self.hidden_size}, layers={self.num_layers}, dropout={self.dropout}")
            
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse parameters from model name '{model_name}': {e}")
            print("Using defaults: seq=30, hidden=64, layers=2, dropout=0.2")
            self.sequence_length = 30
            self.hidden_size = 64
            self.num_layers = 2
            self.dropout = 0.2
    
    def load_model(self, model_path, scaler_path):
        """Load trained model and scaler"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Initialize model with correct architecture from parsed parameters
        input_size = len(self.feature_names)  # Number of features
        self.model = WeatherLSTM(
            input_size=input_size,
            hidden_size=getattr(self, 'hidden_size', 64),
            num_layers=getattr(self, 'num_layers', 2),
            output_size=1,
            dropout=getattr(self, 'dropout', 0.2)
        ).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Scaler loaded from {scaler_path}")
        print(f"Using device: {self.device}")
    
    def prepare_input_data(self, weather_data):
        """Prepare input data for prediction"""
        if len(weather_data) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} days of data for prediction")
        
        # Ensure we have all required features
        for feature in self.feature_names:
            if feature not in weather_data.columns:
                raise ValueError(f"Missing required feature: {feature}")
        
        # Get the last sequence_length days
        recent_data = weather_data[self.feature_names].tail(self.sequence_length).values
        
        # Scale the data
        scaled_data = self.scaler.transform(recent_data)
        
        # Reshape for model input (batch_size=1, sequence_length, features)
        input_tensor = torch.FloatTensor(scaled_data).unsqueeze(0).to(self.device)
        
        return input_tensor
    
    def predict_temperature(self, weather_data, days_ahead=1):
        """Predict temperature for upcoming days"""
        predictions = []
        current_data = weather_data.copy()
        
        for day in range(days_ahead):
            # Prepare input
            input_tensor = self.prepare_input_data(current_data)
            
            # Make prediction
            with torch.no_grad():
                scaled_prediction = self.model(input_tensor).cpu().numpy()[0, 0]
            
            # Denormalize prediction
            # Create dummy array for inverse transform
            dummy_features = np.zeros((1, len(self.feature_names)))
            dummy_features[0, 0] = scaled_prediction  # Temperature is first feature
            denorm_prediction = self.scaler.inverse_transform(dummy_features)[0, 0]
            
            predictions.append(denorm_prediction)
            
            # For multi-step prediction, append prediction to data
            # (This is a simplified approach - could be improved)
            if day < days_ahead - 1:
                # Create a new row with predicted temperature and estimated other features
                last_row = current_data.iloc[-1].copy()
                last_row['temperature'] = denorm_prediction
                
                # Simple estimation for other features (could be more sophisticated)
                last_row['humidity'] = max(20, min(95, last_row['humidity'] + np.random.normal(0, 2)))
                last_row['pressure'] = last_row['pressure'] + np.random.normal(0, 2)
                last_row['wind_speed'] = max(0, last_row['wind_speed'] + np.random.normal(0, 1))
                last_row['precipitation'] = max(0, np.random.exponential(1) if np.random.random() < 0.3 else 0)
                
                # Add new row
                new_row = pd.DataFrame([last_row])
                current_data = pd.concat([current_data, new_row], ignore_index=True)
        
        return predictions
    
    def predict_with_confidence(self, weather_data, days_ahead=1, num_samples=50):
        """Predict with confidence intervals using Monte Carlo dropout"""
        self.model.train()  # Enable dropout for uncertainty estimation
        
        all_predictions = []
        
        for _ in range(num_samples):
            predictions = self.predict_temperature(weather_data, days_ahead)
            all_predictions.append(predictions)
        
        self.model.eval()  # Back to evaluation mode
        
        # Calculate statistics
        predictions_array = np.array(all_predictions)
        mean_predictions = np.mean(predictions_array, axis=0)
        std_predictions = np.std(predictions_array, axis=0)
        
        # Confidence intervals (95%)
        lower_bound = mean_predictions - 1.96 * std_predictions
        upper_bound = mean_predictions + 1.96 * std_predictions
        
        return {
            'predictions': mean_predictions,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'std': std_predictions
        }

# Генератор данных вынесен в отдельный модуль weather_data_generator.py

def find_available_models():
    """Find all available trained models"""
    models_dir = 'Predictions/models'
    available_models = []
    
    if not os.path.exists(models_dir):
        return available_models
    
    for model_folder in os.listdir(models_dir):
        model_path = os.path.join(models_dir, model_folder)
        if os.path.isdir(model_path):
            # Check if required files exist
            model_file = os.path.join(model_path, f'{model_folder}_best.pth')
            scaler_file = os.path.join(model_path, f'{model_folder}_scaler.pkl')
            
            if os.path.exists(model_file) and os.path.exists(scaler_file):
                available_models.append(model_folder)
    
    return sorted(available_models)

def select_model():
    """Interactive model selection"""
    available_models = find_available_models()
    
    if not available_models:
        print("No trained models found. Please run weather_train.py first.")
        return None
    
    print("Available trained models:")
    print("=" * 40)
    
    for i, model_name in enumerate(available_models, 1):
        # Parse and display model parameters
        try:
            parts = model_name.split('_')
            seq_len = 'N/A'
            hidden = 'N/A'
            layers = 'N/A'
            lr = 'N/A'
            dropout = 'N/A'
            
            for part in parts:
                if part.startswith('seq') and len(part) > 3:
                    seq_len = int(part[3:])
                elif part.startswith('h') and len(part) > 1 and part[1:].isdigit():
                    hidden = int(part[1:])
                elif part.startswith('l') and len(part) > 1 and part[1:].isdigit():
                    layers = int(part[1:])
                elif part.startswith('lr') and len(part) > 2:
                    lr = part[2:]
                elif part.startswith('drop') and len(part) > 4:
                    dropout = part[4:]
            
            print(f"{i}. {model_name}")
            print(f"   Sequence: {seq_len} days, Hidden: {hidden}, Layers: {layers}")
            print(f"   Learning Rate: {lr}, Dropout: {dropout}")
            print()
        except Exception as e:
            print(f"{i}. {model_name}")
            print(f"   (Parameters could not be parsed: {e})")
            print()
    
    while True:
        try:
            choice = input(f"Select model (1-{len(available_models)}) or press Enter for the first one: ").strip()
            
            if not choice:  # Default to first model
                return available_models[0]
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(available_models):
                return available_models[choice_idx]
            else:
                print(f"Please enter a number between 1 and {len(available_models)}")
        except ValueError:
            print("Please enter a valid number")

def extract_data_type_from_model_name(model_name):
    """Extract data type from model name"""
    data_types = ["realistic", "linear_trend", "random_walk", "step_changes", "noisy", "polynomial"]
    
    for data_type in data_types:
        if data_type in model_name:
            return data_type
    
    return "realistic"  # Default fallback

def select_prediction_data_type(model_data_type):
    """Select data type for prediction input"""
    data_types = [("same", f"Тот же тип что и модель ({model_data_type})")] + WeatherDataGenerator.get_available_data_types()
    
    print(f"\nМодель была обучена на данных типа: {model_data_type}")
    print("Выберите тип данных для предсказания:")
    print("=" * 50)
    
    for i, (data_type, description) in enumerate(data_types, 1):
        print(f"{i}. {description}")
    
    while True:
        try:
            choice = input(f"\nВыберите тип данных (1-{len(data_types)}) или Enter для того же типа: ").strip()
            
            if not choice:  # Default to same as model
                return model_data_type
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(data_types):
                selected_type = data_types[choice_idx][0]
                if selected_type == "same":
                    return model_data_type
                else:
                    return selected_type
            else:
                print(f"Пожалуйста, введите число от 1 до {len(data_types)}")
        except ValueError:
            print("Пожалуйста, введите корректное число")

def plot_predictions(historical_data, predictions_result, days_ahead, model_name):
    """Plot historical data and predictions"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Historical temperature
    last_30_days = historical_data.tail(30)
    axes[0].plot(range(len(last_30_days)), last_30_days['temperature'], 
                'b-', label='Historical Temperature', linewidth=2)
    
    # Predictions
    pred_start = len(last_30_days)
    pred_range = range(pred_start, pred_start + days_ahead)
    
    predictions = predictions_result['predictions']
    lower_bound = predictions_result['lower_bound']
    upper_bound = predictions_result['upper_bound']
    
    axes[0].plot(pred_range, predictions, 'r-', label='Predicted Temperature', 
                linewidth=2, marker='o')
    axes[0].fill_between(pred_range, lower_bound, upper_bound, 
                        alpha=0.3, color='red', label='95% Confidence Interval')
    
    axes[0].axvline(x=pred_start-1, color='gray', linestyle='--', alpha=0.7)
    axes[0].set_title('Temperature Prediction')
    axes[0].set_xlabel('Days')
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Uncertainty visualization
    axes[1].bar(range(days_ahead), predictions_result['std'], alpha=0.7, color='orange')
    axes[1].set_title('Prediction Uncertainty (Standard Deviation)')
    axes[1].set_xlabel('Days Ahead')
    axes[1].set_ylabel('Uncertainty (°C)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save to model-specific results directory
    results_dir = f'Predictions/results/{model_name}'
    os.makedirs(results_dir, exist_ok=True)
    
    plt.savefig(f'{results_dir}/{model_name}_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main prediction function"""
    print("Weather Prediction System")
    print("=" * 30)
    
    try:
        # Select model
        selected_model = select_model()
        if not selected_model:
            return
        
        # Initialize predictor with selected model
        print(f"\nLoading model: {selected_model}")
        predictor = WeatherPredictor(model_name=selected_model)
        
        # Extract data type from model name
        model_data_type = extract_data_type_from_model_name(selected_model)
        
        # Select data type for prediction
        prediction_data_type = select_prediction_data_type(model_data_type)
        
        # Simulate current weather conditions (in real use, this would be actual data)
        print(f"\nGenerating weather data ({prediction_data_type})...")
        data_generator = WeatherDataGenerator.for_prediction(days=35, data_type=prediction_data_type)
        weather_data = data_generator.generate_data()
        
        print("Current weather data preview:")
        print(weather_data.tail().to_string(index=False))
        
        # Make predictions
        days_to_predict = 7
        print(f"\nPredicting temperature for the next {days_to_predict} days...")
        
        # Simple predictions
        simple_predictions = predictor.predict_temperature(weather_data, days_ahead=days_to_predict)
        
        # Predictions with confidence intervals
        confidence_predictions = predictor.predict_with_confidence(
            weather_data, days_ahead=days_to_predict, num_samples=30
        )
        
        # Display results
        print("\nPrediction Results:")
        print("-" * 50)
        
        future_dates = pd.date_range(
            start=weather_data['date'].iloc[-1] + timedelta(days=1),
            periods=days_to_predict,
            freq='D'
        )
        
        for i, date in enumerate(future_dates):
            pred_temp = confidence_predictions['predictions'][i]
            lower = confidence_predictions['lower_bound'][i]
            upper = confidence_predictions['upper_bound'][i]
            uncertainty = confidence_predictions['std'][i]
            
            print(f"{date.strftime('%Y-%m-%d')}: "
                  f"{pred_temp:.1f}°C (±{uncertainty:.1f}) "
                  f"[{lower:.1f}°C - {upper:.1f}°C]")
        
        # Create visualization
        print(f"\nGenerating prediction plots...")
        plot_predictions(weather_data, confidence_predictions, days_to_predict, selected_model)
        
        # Create results directory
        results_dir = f'Predictions/results/{selected_model}'
        os.makedirs(results_dir, exist_ok=True)
        
        # Save predictions to file
        results = {
            'model_name': selected_model,
            'model_data_type': model_data_type,
            'prediction_data_type': prediction_data_type,
            'prediction_date': datetime.now().isoformat(),
            'model_parameters': {
                'sequence_length': predictor.sequence_length,
                'hidden_size': getattr(predictor, 'hidden_size', 64),
                'num_layers': getattr(predictor, 'num_layers', 2),
                'dropout': getattr(predictor, 'dropout', 0.2)
            },
            'predictions': {
                str(date): {
                    'temperature': float(pred),
                    'lower_bound': float(lower),
                    'upper_bound': float(upper),
                    'uncertainty': float(std)
                }
                for date, pred, lower, upper, std in zip(
                    future_dates,
                    confidence_predictions['predictions'],
                    confidence_predictions['lower_bound'],
                    confidence_predictions['upper_bound'],
                    confidence_predictions['std']
                )
            }
        }
        
        results_file = f'{results_dir}/{selected_model}_predictions.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nPrediction completed successfully!")
        print("Files generated:")
        print(f"- {results_dir}/{selected_model}_predictions.png (visualization)")
        print(f"- {results_file} (detailed results)")
        print(f"\nModel used: {selected_model}")
        print(f"Model trained on: {model_data_type} data")
        print(f"Prediction made on: {prediction_data_type} data")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run weather_train.py first to train the model.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()