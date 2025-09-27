"""
Weather Data Generator
Общий модуль для генерации различных типов синтетических данных о погоде
для обучения и тестирования моделей машинного обучения.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class WeatherDataGenerator:
    """Generate synthetic weather data with different patterns for ML training and testing"""
    
    def __init__(self, days=365*3, start_date=None, data_type="realistic"):
        """
        Initialize weather data generator
        
        Args:
            days (int): Number of days to generate data for
            start_date (str or datetime): Start date for data generation
            data_type (str): Type of data pattern to generate
        """
        self.days = days
        self.data_type = data_type
        
        if start_date is None:
            # For training: use historical dates
            self.start_date = pd.to_datetime("2020-01-01")
        elif isinstance(start_date, str):
            self.start_date = pd.to_datetime(start_date)
        else:
            self.start_date = start_date
    
    @classmethod
    def for_training(cls, days=365*3, data_type="realistic"):
        """Create generator for training data (historical dates)"""
        return cls(days=days, start_date="2020-01-01", data_type=data_type)
    
    @classmethod
    def for_prediction(cls, days=35, data_type="realistic"):
        """Create generator for prediction input data (recent dates)"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days-1)
        return cls(days=days, start_date=start_date, data_type=data_type)
    
    def generate_data(self):
        """Generate synthetic weather data based on selected type"""
        if self.data_type == "realistic":
            return self._generate_realistic_data()
        elif self.data_type == "linear_trend":
            return self._generate_linear_trend_data()
        elif self.data_type == "random_walk":
            return self._generate_random_walk_data()
        elif self.data_type == "step_changes":
            return self._generate_step_changes_data()
        elif self.data_type == "noisy":
            return self._generate_noisy_data()
        elif self.data_type == "polynomial":
            return self._generate_polynomial_data()
        else:
            print(f"Warning: Unknown data type '{self.data_type}', using 'realistic'")
            return self._generate_realistic_data()
    
    def _generate_realistic_data(self):
        """Generate realistic weather data with seasonal patterns"""
        dates = pd.date_range(start=self.start_date, periods=self.days, freq='D')
        day_of_year = dates.dayofyear
        
        # Temperature with seasonal pattern
        temperature = (
            20 + 15 * np.sin(2 * np.pi * day_of_year / 365.25) +  # Seasonal
            5 * np.sin(2 * np.pi * day_of_year / 7) +  # Weekly pattern
            np.random.normal(0, 3, len(dates))  # Random noise
        )
        
        # Humidity (inversely related to temperature with noise)
        humidity = (
            70 - 0.5 * (temperature - 20) + 
            10 * np.sin(2 * np.pi * day_of_year / 365.25 + np.pi/2) +
            np.random.normal(0, 5, len(dates))
        )
        humidity = np.clip(humidity, 20, 95)
        
        # Pressure with seasonal variation
        pressure = (
            1013 + 10 * np.sin(2 * np.pi * day_of_year / 365.25 + np.pi/4) +
            np.random.normal(0, 5, len(dates))
        )
        
        # Wind speed
        wind_speed = (
            10 + 5 * np.sin(2 * np.pi * day_of_year / 365.25) +
            np.random.exponential(2, len(dates))
        )
        wind_speed = np.clip(wind_speed, 0, 30)
        
        # Precipitation (more complex pattern)
        precipitation_prob = 0.3 + 0.2 * np.sin(2 * np.pi * day_of_year / 365.25 + np.pi)
        precipitation = np.random.binomial(1, precipitation_prob) * np.random.exponential(5, len(dates))
        
        return self._create_dataframe(dates, temperature, humidity, pressure, wind_speed, precipitation)
    
    def _generate_linear_trend_data(self):
        """Generate data with linear trends"""
        dates = pd.date_range(start=self.start_date, periods=self.days, freq='D')
        time_idx = np.arange(len(dates))
        
        # Adjust trend steepness based on data length
        trend_factor = 1.0 if len(dates) > 100 else 10.0
        
        # Linear temperature trend with noise
        temperature = 15 + (0.01 * trend_factor) * time_idx + np.random.normal(0, 2, len(dates))
        
        # Humidity with opposite trend
        humidity = 80 - (0.005 * trend_factor) * time_idx + np.random.normal(0, 3, len(dates))
        humidity = np.clip(humidity, 20, 95)
        
        # Pressure with slight trend
        pressure = 1013 + (0.002 * trend_factor) * time_idx + np.random.normal(0, 3, len(dates))
        
        # Wind speed with random variations
        wind_speed = 10 + np.random.exponential(1.5, len(dates))
        wind_speed = np.clip(wind_speed, 0, 25)
        
        # Precipitation random
        precipitation = np.random.exponential(2, len(dates)) * np.random.binomial(1, 0.3, len(dates))
        
        return self._create_dataframe(dates, temperature, humidity, pressure, wind_speed, precipitation)
    
    def _generate_random_walk_data(self):
        """Generate random walk data"""
        dates = pd.date_range(start=self.start_date, periods=self.days, freq='D')
        
        # Temperature as random walk
        temperature_changes = np.random.normal(0, 1, len(dates))
        temperature = np.cumsum(temperature_changes) + 20
        
        # Humidity as random walk with bounds
        humidity_changes = np.random.normal(0, 0.5, len(dates))
        humidity = np.cumsum(humidity_changes) + 60
        humidity = np.clip(humidity, 20, 95)
        
        # Pressure as random walk
        pressure_changes = np.random.normal(0, 0.3, len(dates))
        pressure = np.cumsum(pressure_changes) + 1013
        
        # Wind speed with trend
        wind_speed = 5 + np.abs(np.cumsum(np.random.normal(0, 0.2, len(dates))))
        wind_speed = np.clip(wind_speed, 0, 30)
        
        # Precipitation random
        precipitation = np.random.exponential(1.5, len(dates)) * np.random.binomial(1, 0.25, len(dates))
        
        return self._create_dataframe(dates, temperature, humidity, pressure, wind_speed, precipitation)
    
    def _generate_step_changes_data(self):
        """Generate data with sudden step changes"""
        dates = pd.date_range(start=self.start_date, periods=self.days, freq='D')
        
        # Adjust step frequency based on data length
        step_interval = max(10, len(dates) // 10) if len(dates) > 100 else 10
        
        # Temperature with step changes
        temperature = np.ones(len(dates)) * 20
        step_points = np.arange(step_interval, len(dates), step_interval)
        for step in step_points:
            temperature[step:] += np.random.normal(0, 5)
        temperature += np.random.normal(0, 1, len(dates))
        
        # Humidity with different step pattern
        humidity = np.ones(len(dates)) * 60
        step_points = np.arange(step_interval + 5, len(dates), step_interval + 5)
        for step in step_points:
            humidity[step:] += np.random.normal(0, 8)
        humidity = np.clip(humidity, 20, 95)
        
        # Pressure with small steps
        pressure = np.ones(len(dates)) * 1013
        step_points = np.arange(step_interval - 2, len(dates), step_interval - 2)
        for step in step_points:
            pressure[step:] += np.random.normal(0, 3)
        
        # Wind speed normal distribution
        wind_speed = np.abs(np.random.normal(10, 3, len(dates)))
        wind_speed = np.clip(wind_speed, 0, 25)
        
        # Precipitation random
        precipitation = np.random.exponential(2, len(dates)) * np.random.binomial(1, 0.3, len(dates))
        
        return self._create_dataframe(dates, temperature, humidity, pressure, wind_speed, precipitation)
    
    def _generate_noisy_data(self):
        """Generate very noisy data with minimal patterns"""
        dates = pd.date_range(start=self.start_date, periods=self.days, freq='D')
        
        # High noise temperature
        temperature = 20 + np.random.normal(0, 8, len(dates))
        
        # High noise humidity
        humidity = np.random.uniform(20, 95, len(dates))
        
        # High noise pressure
        pressure = 1013 + np.random.normal(0, 15, len(dates))
        
        # High noise wind speed
        wind_speed = np.random.exponential(5, len(dates))
        wind_speed = np.clip(wind_speed, 0, 30)
        
        # High variability precipitation
        precipitation = np.random.exponential(3, len(dates)) * np.random.binomial(1, 0.4, len(dates))
        
        return self._create_dataframe(dates, temperature, humidity, pressure, wind_speed, precipitation)
    
    def _generate_polynomial_data(self):
        """Generate data with polynomial trends"""
        dates = pd.date_range(start=self.start_date, periods=self.days, freq='D')
        time_idx = np.arange(len(dates)) / len(dates)  # Normalize to [0,1]
        
        # Polynomial temperature trend
        temperature = 20 + 10 * time_idx - 15 * time_idx**2 + 8 * time_idx**3 + np.random.normal(0, 2, len(dates))
        
        # Quadratic humidity trend
        humidity = 60 + 20 * time_idx - 10 * time_idx**2 + np.random.normal(0, 3, len(dates))
        humidity = np.clip(humidity, 20, 95)
        
        # Cubic pressure trend
        pressure = 1013 + 5 * time_idx - 8 * time_idx**2 + 4 * time_idx**3 + np.random.normal(0, 2, len(dates))
        
        # Oscillating wind speed
        wind_speed = 10 + 3 * np.sin(10 * np.pi * time_idx) + np.random.exponential(1, len(dates))
        wind_speed = np.clip(wind_speed, 0, 25)
        
        # Precipitation with pattern
        precipitation = np.random.exponential(1 + 2 * time_idx, len(dates)) * np.random.binomial(1, 0.3, len(dates))
        
        return self._create_dataframe(dates, temperature, humidity, pressure, wind_speed, precipitation)
    
    def _create_dataframe(self, dates, temperature, humidity, pressure, wind_speed, precipitation):
        """Helper method to create consistent DataFrame"""
        return pd.DataFrame({
            'date': dates,
            'temperature': temperature,
            'humidity': humidity,
            'pressure': pressure,
            'wind_speed': wind_speed,
            'precipitation': precipitation
        })
    
    @staticmethod
    def get_available_data_types():
        """Get list of available data types with descriptions"""
        return [
            ("realistic", "Реалистичные данные с сезонными паттернами (синусоиды)"),
            ("linear_trend", "Линейные тренды"),
            ("random_walk", "Случайное блуждание"),
            ("step_changes", "Резкие ступенчатые изменения"),
            ("noisy", "Очень шумные данные без паттернов"),
            ("polynomial", "Полиномиальные тренды")
        ]
    
    @staticmethod
    def visualize_data_types(days=365, save_path="Predictions/data_types_comparison.png"):
        """Visualize different data types for comparison"""
        import matplotlib.pyplot as plt
        
        data_types = [dt[0] for dt in WeatherDataGenerator.get_available_data_types()]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for i, data_type in enumerate(data_types):
            generator = WeatherDataGenerator.for_training(days=days, data_type=data_type)
            data = generator.generate_data()
            
            axes[i].plot(data['temperature'], label='Temperature', linewidth=1.5, alpha=0.8)
            axes[i].set_title(f'{data_type.replace("_", " ").title()}')
            axes[i].set_xlabel('Days')
            axes[i].set_ylabel('Temperature (°C)')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Сравнение типов данных сохранено в '{save_path}'")