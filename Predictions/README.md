# Weather Prediction Neural Network

A time sequence LSTM neural network for weather prediction using PyTorch.

## Overview

This project implements a Long Short-Term Memory (LSTM) neural network that predicts temperature based on historical weather patterns. The system consists of two main components:

1. **Training Module** (`weather_train.py`): Trains the LSTM model on synthetic weather data
2. **Prediction Module** (`weather_predict.py`): Uses the trained model to make weather predictions

## Features

- **LSTM Architecture**: Multi-layer LSTM with dropout for regularization
- **Multi-variate Input**: Uses temperature, humidity, pressure, wind speed, and precipitation
- **Sequence Learning**: Uses 30 days of historical data to predict next day's temperature
- **Uncertainty Estimation**: Provides confidence intervals using Monte Carlo dropout
- **Synthetic Data Generation**: Creates realistic weather patterns for training
- **Visualization**: Comprehensive plotting of training results and predictions
- **Early Stopping**: Prevents overfitting with patience-based early stopping
- **Model Persistence**: Save and load trained models

## Installation

1. Install Python 3.7 or later
2. Install required packages:

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install torch numpy pandas scikit-learn matplotlib requests
```

## Usage

### Training the Model

Run the training script to create and train the LSTM model:

```bash
python weather_train.py
```

This will:
- Generate 3 years of synthetic weather data
- Train an LSTM model to predict temperature
- Save the trained model and data scaler
- Generate training visualization plots
- Display training progress and evaluation metrics

Output files:
- `weather_model.pth`: Trained PyTorch model
- `weather_model_best.pth`: Best model checkpoint
- `weather_scaler.pkl`: Data preprocessing scaler
- `weather_training_results.png`: Training visualization

### Making Predictions

After training, use the prediction script:

```bash
python weather_predict.py
```

This will:
- Load the trained model and scaler
- Generate current weather conditions (simulated)
- Predict temperature for the next 7 days
- Provide confidence intervals
- Generate prediction visualizations
- Save results to JSON file

Output files:
- `weather_predictions.png`: Prediction visualization
- `weather_predictions.json`: Detailed prediction results

## Model Architecture

The LSTM neural network consists of:

- **Input Layer**: 5 features (temperature, humidity, pressure, wind speed, precipitation)
- **LSTM Layers**: 2 layers with 64 hidden units each
- **Dropout**: 20% dropout rate for regularization
- **Output Layer**: Single neuron for temperature prediction
- **Sequence Length**: 30 days of historical data

## Data Features

The model uses the following weather parameters:

1. **Temperature** (°C): Target variable for prediction
2. **Humidity** (%): Relative humidity
3. **Pressure** (hPa): Atmospheric pressure
4. **Wind Speed** (m/s): Wind velocity
5. **Precipitation** (mm): Daily rainfall

## Prediction Capabilities

- **Single-step Prediction**: Predict next day's temperature
- **Multi-step Prediction**: Predict multiple days ahead
- **Uncertainty Quantification**: 95% confidence intervals
- **Trend Analysis**: Capture seasonal and weekly patterns

## Performance Metrics

The model is evaluated using:
- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error  
- **RMSE**: Root Mean Squared Error

## Customization

### Training Parameters

Modify in `weather_train.py`:
```python
predictor = WeatherPredictor(
    sequence_length=30,    # Days of history to use
    hidden_size=64,        # LSTM hidden units
    num_layers=2,          # Number of LSTM layers
    learning_rate=0.001    # Learning rate
)
```

### Data Generation

Adjust synthetic data in `WeatherDataGenerator`:
```python
data_generator = WeatherDataGenerator(days=365*3)  # Training data duration
```

### Prediction Horizon

Change prediction period in `weather_predict.py`:
```python
days_to_predict = 7  # Number of days to predict
```

## Model Files

After training, the following files are created:

- `weather_model.pth`: Complete trained model
- `weather_scaler.pkl`: Data normalization scaler (required for predictions)

## Real Data Integration

To use real weather data instead of synthetic data:

1. Replace `WeatherDataGenerator` with actual weather API calls
2. Ensure data format matches: columns = ['date', 'temperature', 'humidity', 'pressure', 'wind_speed', 'precipitation']
3. Maintain consistent units and data quality

Example weather APIs:
- OpenWeatherMap
- WeatherAPI
- National Weather Service

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all packages are installed via `pip install -r requirements.txt`
2. **CUDA Errors**: Model automatically detects GPU; works on CPU if CUDA unavailable
3. **Memory Issues**: Reduce batch_size in training parameters
4. **Poor Predictions**: Try longer training (more epochs) or more training data

### File Not Found Errors

Make sure to run `weather_train.py` before `weather_predict.py` to generate the required model files.

## Future Enhancements

Potential improvements:
- Integration with real weather APIs
- Multi-location prediction
- Additional weather variables (UV index, visibility, etc.)
- Ensemble methods for improved accuracy
- Web interface for easy interaction
- Real-time prediction updates

## Dependencies

- **PyTorch**: Neural network framework
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **Scikit-learn**: Data preprocessing and metrics
- **Matplotlib**: Visualization
- **Requests**: HTTP requests (for future API integration)

## License

This project is open-source and available under the MIT License.