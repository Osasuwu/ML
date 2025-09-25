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
    def __init__(self, model_path='Predictions/saves/weather_model.pth', scaler_path='Predictions/saves/weather_scaler.pkl'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None
        self.sequence_length = 30  # Should match training
        self.feature_names = ['temperature', 'humidity', 'pressure', 'wind_speed', 'precipitation']
        
        # Load model and scaler
        self.load_model(model_path, scaler_path)
    
    def load_model(self, model_path, scaler_path):
        """Load trained model and scaler"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Initialize model with correct architecture
        input_size = len(self.feature_names)  # Number of features
        self.model = WeatherLSTM(
            input_size=input_size,
            hidden_size=64,  # Should match training
            num_layers=2,    # Should match training
            output_size=1
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

class WeatherDataSimulator:
    """Simulate current weather conditions for demonstration"""
    
    @staticmethod
    def generate_current_conditions(days=30):
        """Generate current weather conditions for the last 30 days"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days-1)
        
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        day_of_year = dates.dayofyear
        
        # Generate realistic weather data
        temperature = (
            20 + 15 * np.sin(2 * np.pi * day_of_year / 365.25) +
            5 * np.sin(2 * np.pi * day_of_year / 7) +
            np.random.normal(0, 2, len(dates))
        )
        
        humidity = (
            70 - 0.5 * (temperature - 20) + 
            10 * np.sin(2 * np.pi * day_of_year / 365.25 + np.pi/2) +
            np.random.normal(0, 3, len(dates))
        )
        humidity = np.clip(humidity, 20, 95)
        
        pressure = (
            1013 + 10 * np.sin(2 * np.pi * day_of_year / 365.25 + np.pi/4) +
            np.random.normal(0, 3, len(dates))
        )
        
        wind_speed = (
            10 + 5 * np.sin(2 * np.pi * day_of_year / 365.25) +
            np.random.exponential(1.5, len(dates))
        )
        wind_speed = np.clip(wind_speed, 0, 25)
        
        precipitation_prob = 0.3 + 0.2 * np.sin(2 * np.pi * day_of_year / 365.25 + np.pi)
        precipitation = np.random.binomial(1, precipitation_prob) * np.random.exponential(3, len(dates))
        
        df = pd.DataFrame({
            'date': dates,
            'temperature': temperature,
            'humidity': humidity,
            'pressure': pressure,
            'wind_speed': wind_speed,
            'precipitation': precipitation
        })
        
        return df

def plot_predictions(historical_data, predictions_result, days_ahead):
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
    plt.savefig('Predictions/saves/weather_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main prediction function"""
    print("Weather Prediction System")
    print("=" * 30)
    
    try:
        # Initialize predictor
        print("Loading trained model...")
        predictor = WeatherPredictor()
        
        # Simulate current weather conditions (in real use, this would be actual data)
        print("Loading weather data...")
        weather_data = WeatherDataSimulator.generate_current_conditions(days=35)
        
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
        plot_predictions(weather_data, confidence_predictions, days_to_predict)
        
        # Save predictions to file
        results = {
            'prediction_date': datetime.now().isoformat(),
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
        
        with open('weather_predictions.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nPrediction completed successfully!")
        print("Files generated:")
        print("- weather_predictions.png (visualization)")
        print("- weather_predictions.json (detailed results)")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run weather_train.py first to train the model.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()