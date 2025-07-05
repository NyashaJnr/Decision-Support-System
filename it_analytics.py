import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
import json
from datetime import datetime

class ITAnalytics:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        
    def load_data(self):
        """Load and preprocess IT systems support data"""
        df = pd.read_csv('datasets/it_systems_support_dataset.csv')
        return self.preprocess_data(df)
    
    def preprocess_data(self, df):
        """Preprocess the data for analysis"""
        # Convert categorical variables to string
        df['System Name'] = df['System Name'].astype(str)
        df['Issue Reported'] = df['Issue Reported'].astype(str)
        df['Status'] = df['Status'].astype(str)
        
        # Convert date columns
        df['Reported Date'] = pd.to_datetime(df['Reported Date'])
        
        # Ensure features for model training
        df['Month'] = df['Reported Date'].dt.month
        df['Day_of_Week'] = df['Reported Date'].dt.dayofweek
        
        return df
    
    def calculate_metrics(self, df):
        """Calculate key metrics"""
        metrics = {
            'total_tickets': len(df),
            'avg_resolution_time': df['Resolution Time (hrs)'].mean(),
            'first_call_resolution': (df['Resolution Time (hrs)'] <= 1).mean() * 100,
            'system_uptime': 99.9  # This would typically come from system monitoring
        }
        return metrics
    
    def analyze_distributions(self, df):
        """Analyze distributions of ticket types and statuses"""
        distributions = {
            'ticket_types': df['Issue Reported'].value_counts().to_dict(),
            'systems': df['System Name'].value_counts().to_dict(),
            'status': df['Status'].value_counts().to_dict()
        }
        return distributions
    
    def analyze_trends(self, df):
        """Analyze monthly trends"""
        df['Month'] = df['Reported Date'].dt.month
        monthly_trends = df.groupby('Month')['Resolution Time (hrs)'].mean().to_dict()
        return monthly_trends
    
    def analyze_performance(self, df):
        """Analyze performance by category"""
        performance = {
            'by_system': df.groupby('System Name')['Resolution Time (hrs)'].mean().to_dict(),
            'by_issue': df.groupby('Issue Reported')['Resolution Time (hrs)'].mean().to_dict()
        }
        return performance
    
    def train_models(self, df):
        """Train prediction models"""
        # Features for resolution time prediction
        X = df[['System Name', 'Issue Reported', 'Month', 'Day_of_Week']]
        y = df['Resolution Time (hrs)']
        
        # One-hot encode categorical features
        X_encoded = pd.get_dummies(X, columns=['System Name', 'Issue Reported'])
        
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
        self.models['resolution_time'] = model
        self.scalers['resolution_time'] = scaler
        
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
        # Find most common issues
        most_common_issue = max(distributions['ticket_types'].items(), key=lambda x: x[1])[0]
        
        analysis = {
            'insights': [
                f"Average resolution time is {metrics['avg_resolution_time']:.1f} hours",
                f"First call resolution rate is {metrics['first_call_resolution']:.1f}%",
                f"Most common issue is {most_common_issue}"
            ],
            'recommendations': [
                "Implement automated ticket routing based on system and issue type",
                "Develop self-service portal for common issues",
                "Enhance knowledge base with frequently asked questions"
            ]
        }
        return analysis
    
    def get_recent_tickets(self, df, n=10):
        """Get most recent tickets as strings"""
        recent = df.sort_values('Reported Date', ascending=False).head(n)
        recent['System Name'] = recent['System Name'].astype(str)
        recent['Issue Reported'] = recent['Issue Reported'].astype(str)
        recent['Status'] = recent['Status'].astype(str)
        recent['Reported Date'] = recent['Reported Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        recent['Resolution Time (hrs)'] = recent['Resolution Time (hrs)'].astype(float)
        return recent.to_dict('records')

def run_it_analytics():
    """Run IT analytics and save results"""
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Initialize analytics
    it_analytics = ITAnalytics()
    
    # Load and preprocess data
    df = it_analytics.load_data()
    
    # Calculate metrics
    metrics = it_analytics.calculate_metrics(df)
    
    # Analyze distributions
    distributions = it_analytics.analyze_distributions(df)
    
    # Analyze trends
    trends = it_analytics.analyze_trends(df)
    
    # Analyze performance
    performance = it_analytics.analyze_performance(df)
    
    # Train models and get performance
    model_performance = it_analytics.train_models(df)
    
    # Generate analysis
    analysis = it_analytics.generate_analysis(df, metrics, distributions, performance)
    
    # Get recent tickets
    recent_tickets = it_analytics.get_recent_tickets(df)
    
    # Save models
    joblib.dump(it_analytics.models['resolution_time'], 'models/resolution_time_model.joblib')
    joblib.dump(it_analytics.scalers['resolution_time'], 'models/resolution_time_scaler.joblib')
    
    # Save summary
    summary = {
        'metrics': metrics,
        'distributions': distributions,
        'trends': trends,
        'performance': performance,
        'model_performance': model_performance,
        'analysis': analysis,
        'recent_tickets': recent_tickets
    }
    
    with open('outputs/it_analytics_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    
    return summary 