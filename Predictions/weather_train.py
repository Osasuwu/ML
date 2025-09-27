import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import requests
import json
from datetime import datetime, timedelta
import pickle
import os
from weather_data_generator import WeatherDataGenerator

class WeatherLSTM(nn.Module):
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

# Генератор данных вынесен в отдельный модуль weather_data_generator.py

class WeatherPredictor:
    def __init__(self, sequence_length=30, hidden_size=64, num_layers=2, learning_rate=0.001, dropout=0.2, data_type="realistic"):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.data_type = data_type
        self.scaler = MinMaxScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = self._generate_model_name()
    
    def _generate_model_name(self):
        """Generate automatic model name based on hyperparameters and data type"""
        return f"weather_{self.data_type}_seq{self.sequence_length}_h{self.hidden_size}_l{self.num_layers}_lr{self.learning_rate}_drop{self.dropout}"
        
    def prepare_data(self, df, target_column='temperature'):
        """Prepare data for training"""
        # Select features for training
        feature_columns = ['temperature', 'humidity', 'pressure', 'wind_speed', 'precipitation']
        data = df[feature_columns].values
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        target_idx = feature_columns.index(target_column)
        
        for i in range(len(scaled_data) - self.sequence_length):
            X.append(scaled_data[i:(i + self.sequence_length)])
            y.append(scaled_data[i + self.sequence_length, target_idx])
        
        return np.array(X), np.array(y)
    
    def create_dataloaders(self, X, y, train_split=0.8, batch_size=32):
        """Create train and validation dataloaders"""
        split_idx = int(len(X) * train_split)
        
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        X_val = torch.FloatTensor(X_val)
        y_val = torch.FloatTensor(y_val)
        
        # Create datasets
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, (X_val, y_val)
    
    def train_model(self, train_loader, val_loader, val_data, epochs=100):
        """Train the LSTM model"""
        input_size = train_loader.dataset[0][0].shape[1]  # Number of features
        
        self.model = WeatherLSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=1,
            dropout=self.dropout
        ).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model with automatic naming
                model_dir = f'Predictions/models/{self.model_name}'
                os.makedirs(model_dir, exist_ok=True)
                best_model_path = f'{model_dir}/{self.model_name}_best.pth'
                torch.save(self.model.state_dict(), best_model_path)
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            if patience_counter >= 30:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        # Load best model
        model_dir = f'Predictions/models/{self.model_name}'
        best_model_path = f'{model_dir}/{self.model_name}_best.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return train_losses, val_losses
    
    def evaluate_model(self, val_data):
        """Evaluate model performance"""
        X_val, y_val = val_data
        X_val = torch.FloatTensor(X_val).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_val).cpu().numpy().squeeze()
        
        # Denormalize predictions and actual values
        # Create dummy array for inverse transform
        dummy_features = np.zeros((len(predictions), 5))
        dummy_features[:, 0] = predictions  # Temperature is first feature
        pred_denorm = self.scaler.inverse_transform(dummy_features)[:, 0]
        
        dummy_features[:, 0] = y_val
        actual_denorm = self.scaler.inverse_transform(dummy_features)[:, 0]
        
        mse = mean_squared_error(actual_denorm, pred_denorm)
        mae = mean_absolute_error(actual_denorm, pred_denorm)
        rmse = np.sqrt(mse)
        
        print(f'Model Evaluation:')
        print(f'MSE: {mse:.4f}')
        print(f'MAE: {mae:.4f}')
        print(f'RMSE: {rmse:.4f}')
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'predictions': pred_denorm,
            'actual': actual_denorm
        }
    
    def save_model(self, model_path=None, scaler_path=None):
        """Save trained model and scaler with automatic naming in separate folders"""
        # Create model-specific directory
        model_dir = f'Predictions/models/{self.model_name}'
        os.makedirs(model_dir, exist_ok=True)
        
        if model_path is None:
            model_path = f'{model_dir}/{self.model_name}.pth'
        if scaler_path is None:
            scaler_path = f'{model_dir}/{self.model_name}_scaler.pkl'
            
        torch.save(self.model.state_dict(), model_path)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f'Model saved to {model_path}')
        print(f'Scaler saved to {scaler_path}')

def plot_results(train_losses, val_losses, evaluation_results):
    """Plot training results and predictions"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training losses
    axes[0, 0].plot(train_losses, label='Train Loss')
    axes[0, 0].plot(val_losses, label='Validation Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Predictions vs Actual
    predictions = evaluation_results['predictions']
    actual = evaluation_results['actual']
    
    axes[0, 1].scatter(actual, predictions, alpha=0.5)
    axes[0, 1].plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
    axes[0, 1].set_xlabel('Actual Temperature')
    axes[0, 1].set_ylabel('Predicted Temperature')
    axes[0, 1].set_title('Predictions vs Actual')
    axes[0, 1].grid(True)
    
    # Time series comparison (last 100 points)
    last_n = min(100, len(predictions))
    axes[1, 0].plot(actual[-last_n:], label='Actual', linewidth=2)
    axes[1, 0].plot(predictions[-last_n:], label='Predicted', linewidth=2, alpha=0.8)
    axes[1, 0].set_title('Temperature Prediction (Last 100 days)')
    axes[1, 0].set_xlabel('Days')
    axes[1, 0].set_ylabel('Temperature')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Residuals
    residuals = actual - predictions
    axes[1, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Prediction Residuals')
    axes[1, 1].set_xlabel('Residual (Actual - Predicted)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    return fig  # Return figure for saving

def select_data_type():
    """Function to select data generation type"""
    data_types = WeatherDataGenerator.get_available_data_types()
    
    print("\nДоступные типы данных для обучения:")
    print("=" * 50)
    for i, (data_type, description) in enumerate(data_types, 1):
        print(f"{i}. {description}")
    
    while True:
        try:
            choice = input(f"\nВыберите тип данных (1-{len(data_types)}) или Enter для реалистичных: ").strip()
            
            if not choice:  # Default to realistic
                return "realistic"
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(data_types):
                selected_type = data_types[choice_idx][0]
                print(f"Выбран тип данных: {data_types[choice_idx][1]}")
                return selected_type
            else:
                print(f"Пожалуйста, введите число от 1 до {len(data_types)}")
        except ValueError:
            print("Пожалуйста, введите корректное число")

def visualize_data_types():
    """Visualize different data types for comparison"""
    WeatherDataGenerator.visualize_data_types(days=365, save_path='Predictions/data_types_comparison.png')

def main():
    """Main training function"""
    print("Weather Prediction Model Training")
    print("=" * 40)
    
    # Option to visualize data types
    show_viz = input("\nПоказать сравнение типов данных? (y/n): ").strip().lower()
    if show_viz in ['y', 'yes', 'да']:
        visualize_data_types()
    
    # Select data type
    data_type = select_data_type()
    
    # Generate synthetic weather data
    print(f"\nGenerating synthetic weather data ({data_type})...")
    data_generator = WeatherDataGenerator.for_training(days=365*3, data_type=data_type)  # 3 years of data
    weather_data = data_generator.generate_data()
    
    print(f"Generated {len(weather_data)} days of weather data")
    print("\nData preview:")
    print(weather_data.head())
    print(f"\nData statistics:")
    print(weather_data.describe())
    
    # Initialize predictor with default configuration
    # Для экспериментов можно менять эти параметры:
    predictor = WeatherPredictor(
        sequence_length=30,     # Длина последовательности (дни)
        hidden_size=64,         # Размер скрытого слоя
        num_layers=2,           # Количество LSTM слоев
        learning_rate=0.001,    # Скорость обучения
        dropout=0.2,            # Dropout для регуляризации
        data_type=data_type     # Тип данных для именования модели
    )
    
    # Prepare data
    print("\nPreparing data for training...")
    X, y = predictor.prepare_data(weather_data, target_column='temperature')
    train_loader, val_loader, val_data = predictor.create_dataloaders(X, y, batch_size=32)
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Input shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Train model
    print(f"\nTraining model on {predictor.device}...")
    train_losses, val_losses = predictor.train_model(
        train_loader, val_loader, val_data, epochs=365
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluation_results = predictor.evaluate_model(val_data)
    
    # Save model
    print("\nSaving model...")
    predictor.save_model()
    
    # Plot results
    print("\nGenerating plots...")
    model_dir = f'Predictions/models/{predictor.model_name}'
    fig = plot_results(train_losses, val_losses, evaluation_results)
    
    # Save the plot to model directory
    fig.savefig(f'{model_dir}/{predictor.model_name}_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()  # Show after saving
    plt.close(fig)  # Close figure to free memory
    
    print("\nTraining completed successfully!")
    print("Files generated:")
    print(f"- {predictor.model_name}.pth (trained model)")
    print(f"- {predictor.model_name}_best.pth (best model)")
    print(f"- {predictor.model_name}_scaler.pkl (data scaler)")
    print(f"- {predictor.model_name}_training_results.png (training visualization)")
    print(f"\nAll files saved in: {model_dir}/")

if __name__ == "__main__":
    main()