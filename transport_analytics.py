import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report
import joblib
import os

class TransportAnalytics:
    def __init__(self):
        self.data = None
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        
    def load_data(self, file_path='datasets/transport_and_logistics_dataset.csv'):
        """Load and preprocess the transport dataset"""
        self.data = pd.read_csv(file_path)
        
        # Convert dates to datetime
        self.data['Date of Entry'] = pd.to_datetime(self.data['Date of Entry'])
        
        # Extract features from dates
        self.data['Entry Month'] = self.data['Date of Entry'].dt.month
        self.data['Entry Year'] = self.data['Date of Entry'].dt.year
        self.data['Entry Day'] = self.data['Date of Entry'].dt.day
        
        return self.data
    
    def preprocess_data(self):
        """Preprocess the data for model training"""
        # Create copies of data for different models
        delivery_time_data = self.data.copy()
        fuel_consumption_data = self.data.copy()
        maintenance_cost_data = self.data.copy()
        
        # Prepare features for delivery time prediction
        delivery_features = ['Number of Deliveries per Week', 'Vehicle Condition Score', 'Entry Month', 'Entry Year', 'Entry Day']
        delivery_target = 'Average Delivery Time (hrs)'
        
        # Prepare features for fuel consumption prediction
        fuel_features = ['Number of Deliveries per Week', 'Vehicle Condition Score', 'Entry Month', 'Entry Year', 'Entry Day']
        fuel_target = 'Fuel Consumption (L/week)'
        
        # Prepare features for maintenance cost prediction
        maintenance_features = ['Number of Deliveries per Week', 'Vehicle Condition Score', 'Fuel Consumption (L/week)', 'Entry Month', 'Entry Year', 'Entry Day']
        maintenance_target = 'Maintenance Cost ($/month)'
        
        # Encode categorical variables
        for feature in ['Role', 'Vehicle Type', 'Shift']:
            if feature not in self.label_encoders:
                self.label_encoders[feature] = LabelEncoder()
                self.data[feature] = self.label_encoders[feature].fit_transform(self.data[feature])
        
        # Scale numerical features
        numerical_features = ['Number of Deliveries per Week', 'Average Delivery Time (hrs)', 'Vehicle Condition Score', 
                            'Fuel Consumption (L/week)', 'Maintenance Cost ($/month)']
        for feature in numerical_features:
            if feature not in self.scalers:
                self.scalers[feature] = StandardScaler()
                self.data[feature] = self.scalers[feature].fit_transform(self.data[[feature]])
        
        return {
            'delivery_time': {
                'features': delivery_features,
                'target': delivery_target
            },
            'fuel_consumption': {
                'features': fuel_features,
                'target': fuel_target
            },
            'maintenance_cost': {
                'features': maintenance_features,
                'target': maintenance_target
            }
        }
    
    def train_models(self):
        """Train models for different predictions"""
        preprocessed_data = self.preprocess_data()
        model_metrics = {}
        
        for model_type, data_info in preprocessed_data.items():
            X = self.data[data_info['features']]
            y = self.data[data_info['target']]
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            self.models[model_type] = model
            
            # Evaluate model
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            model_metrics[model_type] = {
                'mse': round(mse, 2),
                'r2_score': round(r2, 2)
            }
            
            print(f"\n{model_type} Model Performance:")
            print(f"Mean Squared Error: {mse:.2f}")
            print(f"R2 Score: {r2:.2f}")
        
        return model_metrics
    
    def save_models(self, output_dir='models'):
        """Save trained models and preprocessing components"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save models
        for model_name, model in self.models.items():
            joblib.dump(model, f'{output_dir}/{model_name}_model.joblib')
        
        # Save scalers
        scaler_name_map = {
            'Number of Deliveries per Week': 'deliveries',
            'Average Delivery Time (hrs)': 'delivery_time',
            'Vehicle Condition Score': 'vehicle_condition',
            'Fuel Consumption (L/week)': 'fuel_consumption',
            'Maintenance Cost ($/month)': 'maintenance_cost'
        }
        for scaler_name, scaler in self.scalers.items():
            safe_name = scaler_name_map.get(scaler_name, scaler_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_'))
            joblib.dump(scaler, f'{output_dir}/{safe_name}_scaler.joblib')
        
        # Save label encoders
        for encoder_name, encoder in self.label_encoders.items():
            safe_name = encoder_name.lower().replace(' ', '_')
            joblib.dump(encoder, f'{output_dir}/{safe_name}_encoder.joblib')
    
    def load_models(self, input_dir='models'):
        """Load trained models and preprocessing components"""
        # Load models
        for model_name in ['delivery_time', 'fuel_consumption', 'maintenance_cost']:
            self.models[model_name] = joblib.load(f'{input_dir}/{model_name}_model.joblib')
        
        # Load scalers
        scaler_name_map = {
            'Number of Deliveries per Week': 'deliveries',
            'Average Delivery Time (hrs)': 'delivery_time',
            'Vehicle Condition Score': 'vehicle_condition',
            'Fuel Consumption (L/week)': 'fuel_consumption',
            'Maintenance Cost ($/month)': 'maintenance_cost'
        }
        for scaler_name in ['Number of Deliveries per Week', 'Average Delivery Time (hrs)', 'Vehicle Condition Score', 
                          'Fuel Consumption (L/week)', 'Maintenance Cost ($/month)']:
            safe_name = scaler_name_map.get(scaler_name, scaler_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_'))
            self.scalers[scaler_name] = joblib.load(f'{input_dir}/{safe_name}_scaler.joblib')
        
        # Load label encoders
        for encoder_name in ['Role', 'Vehicle Type', 'Shift']:
            safe_name = encoder_name.lower().replace(' ', '_')
            self.label_encoders[encoder_name] = joblib.load(f'{input_dir}/{safe_name}_encoder.joblib')
    
    def predict_delivery_time(self, deliveries, vehicle_condition, entry_date):
        """Predict delivery time for a new entry"""
        if 'delivery_time' not in self.models:
            raise ValueError("Delivery time model not trained. Please train the model first.")
        
        # Prepare input data
        entry_date = pd.to_datetime(entry_date)
        input_data = pd.DataFrame({
            'Number of Deliveries per Week': [deliveries],
            'Vehicle Condition Score': [vehicle_condition],
            'Entry Month': [entry_date.month],
            'Entry Year': [entry_date.year],
            'Entry Day': [entry_date.day]
        })
        
        # Scale input data
        for feature in ['Number of Deliveries per Week', 'Vehicle Condition Score']:
            input_data[feature] = self.scalers[feature].transform(input_data[[feature]])
        
        # Make prediction
        prediction = self.models['delivery_time'].predict(input_data)[0]
        
        return round(prediction, 2)
    
    def predict_fuel_consumption(self, deliveries, vehicle_condition, entry_date):
        """Predict fuel consumption for a new entry"""
        if 'fuel_consumption' not in self.models:
            raise ValueError("Fuel consumption model not trained. Please train the model first.")
        
        # Prepare input data
        entry_date = pd.to_datetime(entry_date)
        input_data = pd.DataFrame({
            'Number of Deliveries per Week': [deliveries],
            'Vehicle Condition Score': [vehicle_condition],
            'Entry Month': [entry_date.month],
            'Entry Year': [entry_date.year],
            'Entry Day': [entry_date.day]
        })
        
        # Scale input data
        for feature in ['Number of Deliveries per Week', 'Vehicle Condition Score']:
            input_data[feature] = self.scalers[feature].transform(input_data[[feature]])
        
        # Make prediction
        prediction = self.models['fuel_consumption'].predict(input_data)[0]
        
        return round(prediction, 2)
    
    def predict_maintenance_cost(self, deliveries, vehicle_condition, fuel_consumption, entry_date):
        """Predict maintenance cost for a new entry"""
        if 'maintenance_cost' not in self.models:
            raise ValueError("Maintenance cost model not trained. Please train the model first.")
        
        # Prepare input data
        entry_date = pd.to_datetime(entry_date)
        input_data = pd.DataFrame({
            'Number of Deliveries per Week': [deliveries],
            'Vehicle Condition Score': [vehicle_condition],
            'Fuel Consumption (L/week)': [fuel_consumption],
            'Entry Month': [entry_date.month],
            'Entry Year': [entry_date.year],
            'Entry Day': [entry_date.day]
        })
        
        # Scale input data
        for feature in ['Number of Deliveries per Week', 'Vehicle Condition Score', 'Fuel Consumption (L/week)']:
            input_data[feature] = self.scalers[feature].transform(input_data[[feature]])
        
        # Make prediction
        prediction = self.models['maintenance_cost'].predict(input_data)[0]
        
        return round(prediction, 2)

def run_transport_analytics():
    """Run the complete transport analytics pipeline and save summary to outputs."""
    analytics = TransportAnalytics()
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = analytics.load_data()
    
    # Calculate key metrics
    total_deliveries = int(df['Number of Deliveries per Week'].sum())
    avg_delivery_time = float(df['Average Delivery Time (hrs)'].mean())
    avg_on_time_rate = float(df['On-time Delivery Rate (%)'].mean())
    total_incidents = int(df['Incidents Reported'].sum())
    
    # Vehicle distribution
    vehicle_counts = {k: int(v) for k, v in df['Vehicle Type'].value_counts().to_dict().items()}
    
    # Role distribution
    role_counts = {k: int(v) for k, v in df['Role'].value_counts().to_dict().items()}
    
    # Performance by vehicle type
    vehicle_performance = {k: float(v) for k, v in df.groupby('Vehicle Type')['Average Delivery Time (hrs)'].mean().to_dict().items()}
    
    # Monthly trends
    df['Date of Entry'] = pd.to_datetime(df['Date of Entry'])
    monthly_deliveries = {k: int(v) for k, v in df.groupby(df['Date of Entry'].dt.strftime('%Y-%m'))['Number of Deliveries per Week'].sum().to_dict().items()}
    
    # Train models
    print("\nTraining models...")
    model_metrics = analytics.train_models()
    
    # Save models in outputs/models
    print("\nSaving models...")
    analytics.save_models(output_dir='outputs/models')
    
    # Collect summary info
    summary = {
        'status': 'success',
        'message': 'Transport analytics models trained and saved successfully',
        'metrics': {
            'total_deliveries': total_deliveries,
            'avg_delivery_time': round(avg_delivery_time, 1),
            'avg_on_time_rate': round(avg_on_time_rate, 1),
            'total_incidents': total_incidents
        },
        'distributions': {
            'vehicle_types': vehicle_counts,
            'roles': role_counts
        },
        'performance': {
            'by_vehicle_type': vehicle_performance,
            'monthly_trends': monthly_deliveries
        },
        'model_performance': model_metrics
    }
    
    # Save summary as JSON
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/transport_analytics_summary.json', 'w') as f:
        import json
        json.dump(summary, f, indent=2)
    
    return summary

if __name__ == "__main__":
    run_transport_analytics() 