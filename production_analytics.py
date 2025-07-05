import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
import json
from datetime import datetime

class ProductionAnalytics:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        
    def load_data(self):
        """Load and preprocess production operations data"""
        df = pd.read_csv('datasets/production_operations_dataset.csv')
        return self.preprocess_data(df)
    
    def preprocess_data(self, df):
        """Preprocess the data for analysis"""
        # Convert categorical variables to string
        df['Product_Name'] = df['Product_Name'].astype(str)
        df['Production_Line'] = df['Production_Line'].astype(str)
        df['Shift'] = df['Shift'].astype(str)
        df['Supervisor'] = df['Supervisor'].astype(str)
        
        # Convert date columns
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Calculate efficiency rate
        df['Efficiency_Rate'] = (df['Units_Produced'] - df['Defective_Units']) / df['Units_Produced'] * 100
        
        # Calculate quality score
        df['Quality_Score'] = (df['Units_Produced'] - df['Defective_Units']) / df['Units_Produced'] * 100
        
        # Ensure features for model training
        df['Month'] = df['Date'].dt.month
        df['Day_of_Week'] = df['Date'].dt.dayofweek
        
        return df
    
    def calculate_metrics(self, df):
        """Calculate key metrics"""
        metrics = {
            'total_operations': len(df),
            'avg_efficiency': df['Efficiency_Rate'].mean(),
            'avg_quality_score': df['Quality_Score'].mean(),
            'avg_machine_utilization': df['Machine_Utilization_%'].mean()
        }
        return metrics
    
    def analyze_distributions(self, df):
        """Analyze distributions of operations and statuses"""
        distributions = {
            'product_lines': df['Product_Name'].value_counts().to_dict(),
            'production_lines': df['Production_Line'].value_counts().to_dict(),
            'shifts': df['Shift'].value_counts().to_dict()
        }
        return distributions
    
    def analyze_trends(self, df):
        """Analyze monthly trends"""
        monthly_trends = df.groupby('Month')['Efficiency_Rate'].mean().to_dict()
        return monthly_trends
    
    def analyze_performance(self, df):
        """Analyze performance by category"""
        performance = {
            'by_product': df.groupby('Product_Name')['Efficiency_Rate'].mean().to_dict(),
            'by_line': df.groupby('Production_Line')['Efficiency_Rate'].mean().to_dict(),
            'by_shift': df.groupby('Shift')['Efficiency_Rate'].mean().to_dict()
        }
        return performance
    
    def train_models(self, df):
        """Train prediction models"""
        # Features for efficiency prediction
        X = df[['Product_Name', 'Production_Line', 'Shift', 'Month', 'Day_of_Week']]
        y = df['Efficiency_Rate']
        
        # One-hot encode categorical features
        X_encoded = pd.get_dummies(X, columns=['Product_Name', 'Production_Line', 'Shift'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Save model and scaler
        self.models['efficiency'] = model
        self.scalers['efficiency'] = scaler
        
        # Calculate model performance
        y_pred = model.predict(X_test_scaled)
        mse = np.mean((y_test - y_pred) ** 2)
        r2 = model.score(X_test_scaled, y_test)
        
        return {
            'mse': mse,
            'r2': r2
        }
    
    def generate_analysis(self, df, metrics, distributions, performance):
        """Generate insights and recommendations"""
        # Find most efficient product line
        most_efficient = max(performance['by_product'].items(), key=lambda x: x[1])[0]
        
        analysis = {
            'insights': [
                f"Average efficiency rate is {metrics['avg_efficiency']:.1f}%",
                f"Average quality score is {metrics['avg_quality_score']:.1f}%",
                f"Average machine utilization is {metrics['avg_machine_utilization']:.1f}%",
                f"Most efficient product is {most_efficient}"
            ],
            'recommendations': [
                "Implement lean manufacturing principles across all production lines",
                "Develop standardized operating procedures for common operations",
                "Enhance quality control measures and training programs",
                "Optimize shift scheduling based on performance metrics"
            ]
        }
        return analysis
    
    def get_recent_operations(self, df, n=10):
        """Get most recent operations"""
        recent = df.sort_values('Date', ascending=False).head(n)
        recent['Product_Name'] = recent['Product_Name'].astype(str)
        recent['Production_Line'] = recent['Production_Line'].astype(str)
        recent['Shift'] = recent['Shift'].astype(str)
        recent['Date'] = recent['Date'].dt.strftime('%Y-%m-%d')
        recent['Efficiency_Rate'] = recent['Efficiency_Rate'].astype(float)
        recent['Quality_Score'] = recent['Quality_Score'].astype(float)
        return recent.to_dict('records')

def run_production_analytics():
    """Run production analytics and save results"""
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Initialize analytics
    production_analytics = ProductionAnalytics()
    
    # Load and preprocess data
    df = production_analytics.load_data()
    
    # Calculate metrics
    metrics = production_analytics.calculate_metrics(df)
    
    # Analyze distributions
    distributions = production_analytics.analyze_distributions(df)
    
    # Analyze trends
    trends = production_analytics.analyze_trends(df)
    
    # Analyze performance
    performance = production_analytics.analyze_performance(df)
    
    # Train models and get performance
    model_performance = production_analytics.train_models(df)
    
    # Generate analysis
    analysis = production_analytics.generate_analysis(df, metrics, distributions, performance)
    
    # Get recent operations
    recent_operations = production_analytics.get_recent_operations(df)
    
    # Save models
    joblib.dump(production_analytics.models['efficiency'], 'models/efficiency_model.joblib')
    joblib.dump(production_analytics.scalers['efficiency'], 'models/efficiency_scaler.joblib')
    
    # Save summary
    summary = {
        'metrics': metrics,
        'distributions': distributions,
        'trends': trends,
        'performance': performance,
        'model_performance': model_performance,
        'analysis': analysis,
        'recent_operations': recent_operations
    }
    
    with open('outputs/production_analytics_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    
    return summary 