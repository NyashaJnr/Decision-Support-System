import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import json
import sys

class SupplyChainAnalytics:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        
    def load_data(self):
        """Load and preprocess supply chain data"""
        try:
            df = pd.read_csv('datasets/Supply_Chain_Procurement_dataset.csv')
            df['Order Date'] = pd.to_datetime(df['Order Date'])
            df['Delivery Date'] = pd.to_datetime(df['Delivery Date'])
            return df
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
        
    def preprocess_data(self, df):
        """Preprocess data for analysis"""
        try:
            # Calculate delay days
            df['Delay_Days'] = (df['Delivery Date'] - df['Order Date']).dt.days
            df['Delayed'] = df['Delay_Days'] > 0
            
            # Convert categorical variables
            df['Category'] = pd.Categorical(df['Category'])
            df['Supplier'] = pd.Categorical(df['Supplier'])
            df['Procurement Status'] = pd.Categorical(df['Procurement Status'])
            
            return df
        except Exception as e:
            print(f"Error preprocessing data: {str(e)}")
            raise

    def train_ml_models(self, df):
        """Train machine learning models for predictions"""
        try:
            # Prepare features for classification (delivery delay prediction)
            X_cls = df[['Order Quantity', 'Unit Cost (USD)', 'Delay_Days']]
            y_cls = df['Delayed'].astype(int)
            X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

            # Train classification model
            clf = RandomForestClassifier(random_state=42)
            clf.fit(X_train_cls, y_train_cls)
            y_pred_cls = clf.predict(X_test_cls)
            classification_rep = classification_report(y_test_cls, y_pred_cls, output_dict=True)

            # Prepare features for regression (cost prediction)
            X_reg = df[['Order Quantity', 'Unit Cost (USD)', 'Delay_Days']]
            y_reg = df['Total Cost (USD)']
            X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

            # Train regression model
            reg = RandomForestRegressor(random_state=42)
            reg.fit(X_train_reg, y_train_reg)
            y_pred_reg = reg.predict(X_test_reg)
            mse = mean_squared_error(y_test_reg, y_pred_reg)
            r2 = r2_score(y_test_reg, y_pred_reg)

            # Perform clustering for supplier segmentation
            kmeans = KMeans(n_clusters=3, random_state=42)
            cluster_features = df[['Order Quantity', 'Unit Cost (USD)']]
            df['Cluster'] = kmeans.fit_predict(cluster_features)
            cluster_counts = df['Cluster'].value_counts().to_dict()

            # Save predictions
            df_cls = pd.DataFrame({'Predicted_Delayed': clf.predict(X_cls)})
            df_reg = pd.DataFrame({'Predicted_Total_Cost': reg.predict(X_reg)})
            
            output_dir = 'Outputs'
            os.makedirs(output_dir, exist_ok=True)
            
            df_cls.to_csv(os.path.join(output_dir, 'supply_delay_predictions.csv'), index=False)
            df_reg.to_csv(os.path.join(output_dir, 'supply_cost_predictions.csv'), index=False)

            # Create and save cluster visualization
            plt.figure(figsize=(8, 6))
            for i in range(3):
                cluster = df[df['Cluster'] == i]
                plt.scatter(cluster['Order Quantity'], cluster['Unit Cost (USD)'], 
                           label=f'Cluster {i}', alpha=0.6)
            plt.xlabel('Order Quantity')
            plt.ylabel('Unit Cost (USD)')
            plt.title('Supplier Clustering Analysis')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'supply_clusters.png'))
            plt.close()

            return {
                'classification_report': classification_rep,
                'regression_metrics': {'mse': round(mse, 2), 'r2_score': round(r2, 2)},
                'clustering': {'cluster_distribution': cluster_counts}
            }
        except Exception as e:
            print(f"Error in ML model training: {str(e)}")
            raise
    
    def calculate_metrics(self, df):
        """Calculate key supply chain metrics"""
        try:
            total_cost = df['Total Cost (USD)'].sum()
            avg_delay = df['Delay_Days'].mean()
            on_time_delivery = (df['Delay_Days'] <= 0).mean() * 100
            
            return {
                'total_cost': round(total_cost, 2),
                'avg_delay': round(avg_delay, 2),
                'on_time_delivery': round(on_time_delivery, 2),
                'total_orders': len(df),
                'avg_order_value': round(total_cost / len(df), 2)
            }
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            raise
    
    def analyze_distributions(self, df):
        """Analyze category and supplier distributions"""
        try:
            category_distribution = df.groupby('Category', observed=True)['Total Cost (USD)'].sum()
            supplier_distribution = df.groupby('Supplier', observed=True)['Total Cost (USD)'].sum()
            
            return {
                'category_costs': category_distribution.to_dict(),
                'supplier_costs': supplier_distribution.to_dict()
            }
        except Exception as e:
            print(f"Error analyzing distributions: {str(e)}")
            raise
    
    def analyze_trends(self, df):
        """Analyze monthly trends"""
        try:
            df['Month'] = df['Order Date'].dt.strftime('%Y-%m')
            monthly_cost = df.groupby('Month')['Total Cost (USD)'].sum()
            monthly_delays = df.groupby('Month')['Delay_Days'].mean()
            
            return {
                'monthly_costs': monthly_cost.to_dict(),
                'monthly_delays': monthly_delays.to_dict()
            }
        except Exception as e:
            print(f"Error analyzing trends: {str(e)}")
            raise
    
    def analyze_performance(self, df):
        """Analyze supplier performance"""
        try:
            supplier_performance = df.groupby('Supplier').agg({
                'Delay_Days': 'mean',
                'Total Cost (USD)': 'sum',
                'Procurement Status': lambda x: (x == 'Completed').mean()
            })
            
            return {
                'suppliers': supplier_performance.to_dict('index')
            }
        except Exception as e:
            print(f"Error analyzing performance: {str(e)}")
            raise
    
    def generate_analysis(self, df):
        """Generate supply chain analysis insights"""
        try:
            # Calculate average delay by category
            avg_delay_by_category = df.groupby('Category', observed=True)['Delay_Days'].mean()
            worst_category = avg_delay_by_category.idxmax()
            worst_delay = avg_delay_by_category.max()
            
            # Calculate cost efficiency
            avg_cost_per_order = df['Total Cost (USD)'].mean()
            cost_status = "High" if avg_cost_per_order > df['Total Cost (USD)'].median() else "Low"
            
            # Calculate delivery performance
            on_time_rate = (df['Delay_Days'] <= 0).mean() * 100
            delivery_status = "Good" if on_time_rate >= 90 else "Needs Improvement"
            
            return {
                'worst_performing_category': f"{worst_category} ({worst_delay:.1f} days)",
                'cost_efficiency': f"{cost_status} (${avg_cost_per_order:,.2f})",
                'delivery_performance': f"{delivery_status} ({on_time_rate:.1f}%)"
            }
        except Exception as e:
            print(f"Error generating analysis: {str(e)}")
            raise
    
    def get_recent_orders(self, df):
        """Get recent orders for display"""
        try:
            recent = df.sort_values('Order Date', ascending=False).head(10)
            orders = []
            
            for _, row in recent.iterrows():
                orders.append({
                    'date': row['Order Date'].strftime('%Y-%m-%d'),
                    'category': row['Category'],
                    'description': f"{row['Item Name']} - {row['Supplier']}",
                    'amount': round(row['Total Cost (USD)'], 2),
                    'status': row['Procurement Status']
                })
            
            return orders
        except Exception as e:
            print(f"Error getting recent orders: {str(e)}")
            raise

def run_supply_chain_analytics():
    """Run supply chain analytics and save results"""
    try:
        analytics = SupplyChainAnalytics()
        
        # Load and preprocess data
        df = analytics.load_data()
        df = analytics.preprocess_data(df)
        
        # Train ML models and get predictions
        ml_results = analytics.train_ml_models(df)
        
        # Calculate metrics and generate analysis
        metrics = analytics.calculate_metrics(df)
        distributions = analytics.analyze_distributions(df)
        trends = analytics.analyze_trends(df)
        performance = analytics.analyze_performance(df)
        analysis = analytics.generate_analysis(df)
        recent_orders = analytics.get_recent_orders(df)
        
        # Save results
        summary = {
            'metrics': metrics,
            'distributions': distributions,
            'trends': trends,
            'performance': performance,
            'analysis': analysis,
            'recent_orders': recent_orders,
            'ml_results': ml_results
        }
        
        # Create outputs directory if it doesn't exist
        output_dir = 'Outputs'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save summary to JSON file with proper encoding
        output_path = os.path.join(output_dir, 'supply_chain_analytics_summary.json')
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=4, ensure_ascii=False)
            print(f"Successfully saved analytics summary to {output_path}")
        except Exception as e:
            print(f"Error saving analytics summary: {str(e)}")
            raise
        
        return summary
    except Exception as e:
        print(f"Error in run_supply_chain_analytics: {str(e)}")
        raise

if __name__ == '__main__':
    try:
        results = run_supply_chain_analytics()
        print("Analytics completed successfully")
    except Exception as e:
        print(f"Error running analytics: {str(e)}")
        sys.exit(1)
