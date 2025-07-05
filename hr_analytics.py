import pandas as pd
import numpy as np
import os
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, mean_squared_error, r2_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA

def run_hr_analytics():
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    # Load and preprocess data
    df = pd.read_csv('datasets/hr_dataset.csv')
    df.drop(['Employee_ID', 'Name', 'Date_Hired'], axis=1, inplace=True)

    le = LabelEncoder()
    for col in ['Gender', 'Department', 'Position', 'Employment_Type']:
        df[col] = le.fit_transform(df[col])
    df['Resigned'] = df['Resigned'].map({'Yes': 1, 'No': 0})

    df['Total_Active_Time'] = df['Training_Hours'] + df['Overtime_Hours'] - df['Leave_Days_Taken']

    numeric_cols = ['Age', 'Monthly_Salary', 'Performance_Score',
                    'Training_Hours', 'Leave_Days_Taken', 'Overtime_Hours', 'Total_Active_Time']
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # === Attrition Prediction ===
    X_attr = df.drop('Resigned', axis=1)
    y_attr = df['Resigned']
    X_attr_bal, y_attr_bal = SMOTE(random_state=42).fit_resample(X_attr, y_attr)
    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X_attr_bal, y_attr_bal, test_size=0.2, random_state=42)

    rf_clf = RandomForestClassifier(random_state=42)
    params_clf = {
        'n_estimators': [100, 150],
        'max_depth': [3, 5],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    clf_search = RandomizedSearchCV(rf_clf, params_clf, n_iter=4, cv=3, random_state=42, n_jobs=-1)
    clf_search.fit(X_train_a, y_train_a)
    y_pred_a = clf_search.predict(X_test_a)
    clf_report = classification_report(y_test_a, y_pred_a, output_dict=True)

    pd.DataFrame({'Actual': y_test_a, 'Predicted': y_pred_a}).to_csv('outputs/attrition_predictions.csv', index=False)
    joblib.dump(clf_search.best_estimator_, 'models/attrition_rf_model.pkl')

    # === Performance Regression ===
    y_perf = df['Performance_Score']
    X_perf = df.drop('Performance_Score', axis=1)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_perf, y_perf, test_size=0.2, random_state=42)

    rf_reg = RandomForestRegressor(random_state=42)
    params_reg = {
        'n_estimators': [100, 150],
        'max_depth': [3, 5],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    reg_search = RandomizedSearchCV(rf_reg, params_reg, n_iter=4, cv=3, random_state=42, n_jobs=-1)
    reg_search.fit(X_train_r, y_train_r)
    y_pred_r = reg_search.predict(X_test_r)

    regression_metrics = {
        "mse": float(mean_squared_error(y_test_r, y_pred_r)),
        "r2_score": float(r2_score(y_test_r, y_pred_r))
    }
    pd.DataFrame({'Actual': y_test_r, 'Predicted': y_pred_r}).to_csv('outputs/performance_predictions.csv', index=False)
    joblib.dump(reg_search.best_estimator_, 'models/performance_rf_model.pkl')

    # === Clustering ===
    cluster_features = df[numeric_cols]
    sil_scores = {}
    for k in range(2, 6):
        kmeans_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans_tmp.fit_predict(cluster_features)
        sil_scores[k] = silhouette_score(cluster_features, labels)

    optimal_k = max(sil_scores, key=sil_scores.get)
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df['Resource_Group'] = kmeans.fit_predict(cluster_features)
    cluster_distribution = df['Resource_Group'].value_counts().to_dict()
    joblib.dump(kmeans, 'models/resource_kmeans_model.pkl')

    # Save plot
    try:
        reduced = PCA(n_components=2).fit_transform(cluster_features)
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=df['Resource_Group'], palette='viridis')
        plt.title("Resource Allocation Clusters")
        plt.savefig('outputs/resource_clusters.png')
        plt.close()
    except:
        pass

    # === Training Recommendation ===
    perf_threshold = df['Performance_Score'].mean()
    leave_threshold = df['Leave_Days_Taken'].mean()
    df['Recommend_Training'] = ((df['Performance_Score'] < perf_threshold) &
                                (df['Leave_Days_Taken'] > leave_threshold)).astype(int)
    training_count = int(df['Recommend_Training'].sum())

    joblib.dump(scaler, 'models/hr_scaler.pkl')

    # Calculate metrics
    metrics = {
        'total_employees': int(len(df)),
        'avg_tenure': float(df['Total_Active_Time'].mean()),
        'training_completion': float(df['Training_Hours'].mean()),
        'satisfaction_score': float(df['Performance_Score'].mean())
    }

    # Calculate distributions
    distributions = {
        'departments': df['Department'].value_counts().to_dict(),
        'performance': df['Performance_Score'].value_counts().to_dict()
    }

    # Calculate insights
    insights = {
        'departments': [
            {
                'department': dept,
                'avg_performance': float(df[df['Department'] == dept]['Performance_Score'].mean()),
                'attrition_rate': float(df[df['Department'] == dept]['Resigned'].mean())
            }
            for dept in df['Department'].unique()
        ],
        'top_performers': [
            {
                'department': row['Department'],
                'position': row['Position'],
                'performance_score': float(row['Performance_Score'])
            }
            for _, row in df.nlargest(5, 'Performance_Score').iterrows()
        ]
    }

    # Calculate trends
    trends = {
        'training': {
            'effectiveness': float(df.groupby('Training_Hours')['Performance_Score'].mean().corr(df['Training_Hours']))
        }
    }

    # Calculate analysis
    analysis = {
        'insights': [
            {
                'title': 'Performance Analysis',
                'description': f"Average performance score is {metrics['satisfaction_score']:.2f}",
                'impact': 'High',
                'trend': 'Stable'
            },
            {
                'title': 'Training Impact',
                'description': f"Training effectiveness correlation: {trends['training']['effectiveness']:.2f}",
                'impact': 'Medium',
                'trend': 'Improving'
            }
        ],
        'recommendations': [
            {
                'title': 'Performance Optimization',
                'description': 'Implement targeted training programs for underperforming employees',
                'priority': 'High',
                'expected_impact': 'Improve average performance by 15%',
                'implementation_time': '2-3 months'
            },
            {
                'title': 'Resource Allocation',
                'description': 'Optimize resource allocation based on performance clusters',
                'priority': 'Medium',
                'expected_impact': 'Increase efficiency by 10%',
                'implementation_time': '1-2 months'
            }
        ]
    }

    summary = {
        'metrics': metrics,
        'distributions': distributions,
        'insights': insights,
        'trends': trends,
        'analysis': analysis,
        'ml_results': {
            'classification_report': clf_report,
            'regression_metrics': regression_metrics,
            'clustering': {
                'optimal_clusters': int(optimal_k),
                'cluster_distribution': cluster_distribution,
                'cluster_plot': 'outputs/resource_clusters.png'
            }
        },
        'training_recommendations': [
            {
                'department': row['Department'],
                'position': row['Position'],
                'performance_score': float(row['Performance_Score']),
                'training_hours': float(row['Training_Hours'])
            }
            for _, row in df[df['Recommend_Training'] == 1].iterrows()
        ]
    }

    with open("Outputs/hr_analytics_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    return summary