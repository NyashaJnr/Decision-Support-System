import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
import json
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score

class SalesAnalytics:
    def __init__(self):
        self.models = {}
        self.scalers = {}

    def load_data(self):
        df = pd.read_csv('datasets/sales_and_marketing_dataset.csv')
        return self.preprocess_data(df)

    def preprocess_data(self, df):
        # Convert categorical variables
        if 'Product' in df.columns:
            df['Product'] = df['Product'].astype(str)
        if 'Region' in df.columns:
            df['Region'] = df['Region'].astype(str)
        if 'Salesperson' in df.columns:
            df['Salesperson'] = df['Salesperson'].astype(str)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df['Month'] = df['Date'].dt.month
            df['Year'] = df['Date'].dt.year
        return df

    def calculate_metrics(self, df):
        metrics = {
            'total_sales': df['Revenue'].sum() if 'Revenue' in df.columns else 0,
            'total_units': df['Units Sold'].sum() if 'Units Sold' in df.columns else 0,
            'avg_unit_price': df['Revenue'].sum() / df['Units Sold'].sum() if 'Revenue' in df.columns and 'Units Sold' in df.columns and df['Units Sold'].sum() > 0 else 0,
            'num_transactions': len(df)
        }
        return metrics

    def analyze_distributions(self, df):
        distributions = {
            'by_product': df['Product'].value_counts().to_dict() if 'Product' in df.columns else {},
            'by_region': df['Region'].value_counts().to_dict() if 'Region' in df.columns else {},
            'by_salesperson': df['Salesperson'].value_counts().to_dict() if 'Salesperson' in df.columns else {}
        }
        return distributions

    def analyze_trends(self, df):
        if 'Date' in df.columns and 'Revenue' in df.columns:
            monthly_trend = df.groupby(df['Date'].dt.to_period('M'))['Revenue'].sum().to_dict()
            # Convert Period to string for JSON serialization
            monthly_trend = {str(k): float(v) for k, v in monthly_trend.items()}
        else:
            monthly_trend = {}
        return monthly_trend

    def analyze_performance(self, df):
        performance = {
            'top_products': df.groupby('Product')['Revenue'].sum().sort_values(ascending=False).head(5).to_dict() if 'Product' in df.columns and 'Revenue' in df.columns else {},
            'top_regions': df.groupby('Region')['Revenue'].sum().sort_values(ascending=False).head(5).to_dict() if 'Region' in df.columns and 'Revenue' in df.columns else {},
            'top_salespersons': df.groupby('Salesperson')['Revenue'].sum().sort_values(ascending=False).head(5).to_dict() if 'Salesperson' in df.columns and 'Revenue' in df.columns else {}
        }
        return performance

    def generate_analysis(self, df, metrics, distributions, performance):
        insights = [
            f"Total sales revenue: ${metrics['total_sales']:.2f}",
            f"Total units sold: {metrics['total_units']}",
            f"Average unit price: ${metrics['avg_unit_price']:.2f}",
            f"Number of transactions: {metrics['num_transactions']}"
        ]
        if performance['top_products']:
            top_product = max(performance['top_products'], key=performance['top_products'].get)
            insights.append(f"Top selling product: {top_product}")
        if performance['top_regions']:
            top_region = max(performance['top_regions'], key=performance['top_regions'].get)
            insights.append(f"Top performing region: {top_region}")
        if performance['top_salespersons']:
            top_salesperson = max(performance['top_salespersons'], key=performance['top_salespersons'].get)
            insights.append(f"Top salesperson: {top_salesperson}")
        recommendations = [
            "Focus marketing efforts on top-performing products and regions.",
            "Provide incentives for high-performing sales staff.",
            "Analyze underperforming products for improvement opportunities.",
            "Monitor monthly trends to adjust sales strategies."
        ]
        return {'insights': insights, 'recommendations': recommendations}

    def get_recent_sales(self, df, n=10):
        if 'Date' in df.columns:
            recent = df.sort_values('Date', ascending=False).head(n)
        else:
            recent = df.tail(n)
        return recent.to_dict('records')

def run_sales_analytics():
    if not os.path.exists('models'):
        os.makedirs('models')
    sales_analytics = SalesAnalytics()
    df = sales_analytics.load_data()
    metrics = sales_analytics.calculate_metrics(df)
    distributions = sales_analytics.analyze_distributions(df)
    trends = sales_analytics.analyze_trends(df)
    performance = sales_analytics.analyze_performance(df)
    analysis = sales_analytics.generate_analysis(df, metrics, distributions, performance)
    recent_sales = sales_analytics.get_recent_sales(df)

    # Regression: Predict Revenue
    features = ['Units Sold', 'Month', 'Year']
    regression_metrics = {}
    if all(f in df.columns for f in features + ['Revenue']):
        X = df[features]
        y = df['Revenue']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        regression_metrics = {'mse': mse, 'r2_score': r2}
        joblib.dump(model, 'models/sales_revenue_rf_model.pkl')
    summary = {
        'metrics': metrics,
        'distributions': distributions,
        'trends': trends,
        'performance': performance,
        'analysis': analysis,
        'recent_sales': recent_sales,
        'regression_metrics': regression_metrics
    }
    with open('outputs/sales_analytics_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    return summary 