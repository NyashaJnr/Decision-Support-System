from flask import Flask, render_template, request, jsonify, send_from_directory, flash, redirect, url_for, send_file
from pymongo import MongoClient
import json
import os
from flask import session
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from models.database import db
from models.user import User
from models.log import Log
from models.prediction import Prediction
from models.uploaded_dataset import UploadedDataset
import time
from supply_chain_analytics import run_supply_chain_analytics
from routes.auth import auth, admin_required, department_access_required
from routes.uploads import uploads_bp
from routes.user_management import user_management_bp
import pandas as pd
from io import BytesIO

client = MongoClient('mongodb://localhost:27017/')
db_mongo = client['DSSdb']

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for session management

# Configure SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy with the app
db.init_app(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'

@login_manager.user_loader
def load_user(user_id):
    return User.get_by_id(user_id)

# Register blueprints
app.register_blueprint(auth)
app.register_blueprint(uploads_bp)
app.register_blueprint(user_management_bp)

def create_initial_admin():
    with app.app_context():
        # Check if admin user exists
        admin = User.get_by_email('admin@example.com')
        if not admin:
            # Create admin user
            admin = User.create_user(
                email='admin@example.com',
                password='admin123',  # Change this in production!
                role='admin',
                department='IT'
            )
            if admin:
                print("Initial admin user created successfully!")
            else:
                print("Failed to create initial admin user!")

# Create database tables if they don't exist
with app.app_context():
    db.create_all()
    create_initial_admin()

# Serve files from Outputs directory
@app.route('/static/outputs/<path:filename>')
def serve_output(filename):
    return send_from_directory('Outputs', filename)

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('welcome.html')

@app.route('/dashboard')
@login_required
def dashboard():
    if not current_user.is_authenticated:
        return redirect(url_for('auth.login'))
    
    # Check if this is a fresh login by checking the session
    if session.get('_fresh', False):
        flash('You are logged in successfully!', 'success')
        session['_fresh'] = False
    
    return render_template('dashboard.html')

@app.route('/analytics')
@login_required
def analytics():
    return render_template('analytics.html')

# Department Dashboard Routes
@app.route('/supply-chain-dashboard')
@login_required
@department_access_required('Supply Chain')
def supply_chain_dashboard():
    try:
        json_path = os.path.join('Outputs', 'supply_chain_analytics_summary.json')
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return render_template('supply_chain_dashboard.html', results=data)
        else:
            return render_template('supply_chain_dashboard.html',
                results={
                    'metrics': {
                        'total_cost': 0,
                        'avg_delay': 0,
                        'on_time_delivery': 0
                    },
                    'distributions': {
                        'category_costs': {}
                    },
                    'trends': {
                        'monthly_costs': {},
                        'monthly_delays': {}
                    },
                    'performance': {
                        'suppliers': {}
                    },
                    'ml_results': {
                        'classification_report': {
                            'accuracy': 0,
                            'macro avg': {'f1-score': 0}
                        },
                        'regression_metrics': {
                            'r2_score': 0,
                            'mse': 0
                        }
                    },
                    'recent_orders': [],
                    'analysis': {
                        'insights': [],
                        'recommendations': []
                    }
                }
            )
    except Exception as e:
        flash(f'Error loading Supply Chain dashboard: {str(e)}', 'error')
    return render_template('supply_chain_dashboard.html')

@app.route('/hr-dashboard')
@login_required
@department_access_required('HR')
def hr_dashboard():
    try:
        json_path = os.path.join('Outputs', 'hr_analytics_summary.json')
        if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                from hr_analytics import run_hr_analytics
                data = run_hr_analytics()
        else:
            from hr_analytics import run_hr_analytics
            data = run_hr_analytics()
        metrics = data.get('metrics', {})
        distributions = data.get('distributions', {})
        insights = data.get('insights', {})
        trends = data.get('trends', {})
        analysis = data.get('analysis', {})
        ml_results = data.get('ml_results', {})
        # Ensure classification_report has keys '0' and '1' with 'precision'
        classification_report = ml_results.get('classification_report', {})
        for label in ['0', '1']:
            if label not in classification_report:
                classification_report[label] = {'precision': 0}
            elif 'precision' not in classification_report[label]:
                classification_report[label]['precision'] = 0
        # Ensure macro avg exists
        if 'macro avg' not in classification_report:
            classification_report['macro avg'] = {'f1-score': 0}
        # Ensure clustering has cluster_distribution
        clustering = ml_results.get('clustering', {})
        if 'cluster_distribution' not in clustering:
            clustering['cluster_distribution'] = {}
        template_data = {
            'metrics': {
                'total_employees': metrics.get('total_employees', 0),
                'avg_tenure': metrics.get('avg_tenure', 0),
                'training_completion': metrics.get('training_completion', 0),
                'satisfaction_score': metrics.get('satisfaction_score', 0)
            },
            'department_distribution': distributions.get('departments', {}),
            'department_insights': insights.get('departments', []),
            'performance_distribution': distributions.get('performance', []),
            'top_performers': insights.get('top_performers', []),
            'training_effectiveness': trends.get('training', {}),
            'recommended_training': data.get('training_recommendations', []),
            'analysis': {
                'insights': analysis.get('insights', []),
                'recommendations': analysis.get('recommendations', [])
            },
            'ml_results': {
                'classification_report': classification_report,
                'regression_metrics': ml_results.get('regression_metrics', {
                    'mse': 0,
                    'r2_score': 0
                }),
                'clustering': clustering
            }
        }
        return render_template('hr_dashboard.html', **template_data)
    except Exception as e:
        flash(f'Error loading HR dashboard: {str(e)}', 'error')
        # Provide safe defaults for classification_report and clustering
        classification_report = {
            '0': {'precision': 0},
            '1': {'precision': 0},
            'macro avg': {'f1-score': 0}
        }
        clustering = {'cluster_distribution': {}}
        return render_template('hr_dashboard.html',
            metrics={
                'total_employees': 0,
                'avg_tenure': 0,
                'training_completion': 0,
                'satisfaction_score': 0
            },
            department_distribution={},
            department_insights=[],
            performance_distribution=[],
            top_performers=[],
            training_effectiveness={},
            recommended_training=[],
            analysis={
                'insights': [],
                'recommendations': []
            },
            ml_results={
                'classification_report': classification_report,
                'regression_metrics': {
                    'mse': 0,
                    'r2_score': 0
                },
                'clustering': clustering
            }
        )

@app.route('/transport-dashboard')
@login_required
@department_access_required('Transport')
def transport_dashboard():
    try:
        json_path = os.path.join('Outputs', 'transport_analytics_summary.json')
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            # Generate analytics if JSON doesn't exist
            from transport_analytics import run_transport_analytics
            data = run_transport_analytics()
            
        metrics = {
            'total_deliveries': data.get('metrics', {}).get('total_deliveries', 0),
            'avg_delivery_time': data.get('metrics', {}).get('avg_delivery_time', 0),
            'avg_on_time_rate': data.get('metrics', {}).get('avg_on_time_rate', 0),
            'total_incidents': data.get('metrics', {}).get('total_incidents', 0)
        }
        distributions = {
            'vehicle_types': data.get('distributions', {}).get('vehicle_types', {}),
            'roles': data.get('distributions', {}).get('roles', {})
        }
        performance = {
            'by_vehicle_type': data.get('performance', {}).get('by_vehicle_type', {})
        }
        trends = data.get('performance', {}).get('monthly_trends', {})
        model_performance = {
            'delivery_time': data.get('model_performance', {}).get('delivery_time', {'mse': 0, 'r2_score': 0}),
            'fuel_consumption': data.get('model_performance', {}).get('fuel_consumption', {'mse': 0, 'r2_score': 0}),
            'incident_prediction': data.get('model_performance', {}).get('maintenance_cost', {'mse': 0, 'r2_score': 0})
        }
        recent_entries = data.get('recent_entries', [])
        
        # Add analysis data
        analysis = {
            'insights': [
                {
                    'title': 'Vehicle Performance',
                    'description': f"Average delivery time is {metrics['avg_delivery_time']:.1f} hours with {metrics['avg_on_time_rate']:.1f}% on-time rate.",
                    'impact': 'High',
                    'trend': 'Improving'
                },
                {
                    'title': 'Incident Analysis',
                    'description': f"Total incidents reported: {metrics['total_incidents']}. Focus on preventive maintenance.",
                    'impact': 'Medium',
                    'trend': 'Stable'
                },
                {
                    'title': 'Fleet Utilization',
                    'description': 'Analyze vehicle type distribution for optimal resource allocation.',
                    'impact': 'High',
                    'trend': 'Increasing'
                }
            ],
            'recommendations': [
                {
                    'title': 'Optimize Route Planning',
                    'description': 'Implement AI-based route optimization to reduce delivery times.',
                    'priority': 'High',
                    'expected_impact': 'Reduce delivery time by 15%',
                    'implementation_time': '1 month'
                },
                {
                    'title': 'Enhance Maintenance Schedule',
                    'description': 'Implement predictive maintenance based on vehicle performance data.',
                    'priority': 'High',
                    'expected_impact': 'Reduce incidents by 30%',
                    'implementation_time': '2-3 weeks'
                },
                {
                    'title': 'Driver Training Program',
                    'description': 'Develop comprehensive driver training program focusing on safety and efficiency.',
                    'priority': 'Medium',
                    'expected_impact': 'Improve on-time rate by 20%',
                    'implementation_time': '2 months'
                }
            ]
        }
        
        return render_template('transport_dashboard.html',
            metrics=metrics,
            distributions=distributions,
            performance=performance,
            trends=trends,
            model_performance=model_performance,
            recent_entries=recent_entries,
            analysis=analysis
        )
    except Exception as e:
        flash(f'Error loading Transport dashboard: {str(e)}', 'error')
    return render_template('transport_dashboard.html')

@app.route('/finance-dashboard')
@login_required
@department_access_required('Finance')
def finance_dashboard():
    try:
        json_path = os.path.join('Outputs', 'finance_analytics_summary.json')
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return render_template('finance_dashboard.html',
                    metrics={
                        'total_revenue': data.get('metrics', {}).get('total_revenue', 0),
                        'total_expenses': data.get('metrics', {}).get('total_expenses', 0),
                        'net_profit': data.get('metrics', {}).get('net_profit', 0),
                        'profit_margin': data.get('metrics', {}).get('profit_margin', 0)
                    },
                    revenue={
                        'by_category': data.get('revenue', {}).get('by_category', {})
                    },
                    expenses={
                        'by_category': data.get('expenses', {}).get('by_category', {})
                    },
                    trends={
                        'revenue': data.get('trends', {}).get('revenue', {}),
                        'performance': data.get('trends', {}).get('performance', {})
                    },
                    performance={
                        'by_department': data.get('performance', {}).get('by_department', {})
                    },
                    analysis={
                        'insights': data.get('analysis', {}).get('insights', []),
                        'recommendations': data.get('analysis', {}).get('recommendations', [])
                    },
                    recent_transactions=data.get('recent_transactions', [])
                )
        else:
            return render_template('finance_dashboard.html',
                metrics={
                    'total_revenue': 0,
                    'total_expenses': 0,
                    'net_profit': 0,
                    'profit_margin': 0
                },
                revenue={
                    'by_category': {}
                },
                expenses={
                    'by_category': {}
                },
                trends={
                    'revenue': {},
                    'performance': {}
                },
                performance={
                    'by_department': {}
                },
                analysis={
                    'insights': [],
                    'recommendations': []
                },
                recent_transactions=[]
            )
    except Exception as e:
        flash(f'Error loading Finance dashboard: {str(e)}', 'error')
    return render_template('finance_dashboard.html')

@app.route('/marketing-dashboard')
def marketing_dashboard():
    try:
        json_path = os.path.join('Outputs', 'marketing_analytics_summary.json')
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return render_template('marketing_dashboard.html',
                    metrics=data.get('metrics', {
                        'total_campaigns': 0,
                        'conversion_rate': 0,
                        'customer_acquisition_cost': 0,
                        'roi': 0
                    }),
                    campaign_performance=data.get('campaign_performance', {}),
                    customer_insights=data.get('customer_insights', {}),
                    channel_effectiveness=data.get('channel_effectiveness', {}),
                    analysis={
                        'insights': [
                            {
                                'title': 'Campaign Performance',
                                'description': f"Current conversion rate is {data.get('metrics', {}).get('conversion_rate', 0):.1f}% with ROI of {data.get('metrics', {}).get('roi', 0):.1f}%.",
                                'impact': 'High',
                                'trend': 'Improving'
                            },
                            {
                                'title': 'Customer Acquisition',
                                'description': f"Customer acquisition cost is ${data.get('metrics', {}).get('customer_acquisition_cost', 0):.2f} per customer.",
                                'impact': 'Medium',
                                'trend': 'Stable'
                            },
                            {
                                'title': 'Channel Effectiveness',
                                'description': 'Analyze channel performance for optimal resource allocation.',
                                'impact': 'High',
                                'trend': 'Increasing'
                            }
                        ],
                        'recommendations': [
                            {
                                'title': 'Optimize Campaign Strategy',
                                'description': 'Implement data-driven campaign optimization based on performance metrics.',
                                'priority': 'High',
                                'expected_impact': 'Increase conversion rate by 20%',
                                'implementation_time': '1 month'
                            },
                            {
                                'title': 'Enhance Channel Mix',
                                'description': 'Reallocate budget to high-performing channels and optimize underperforming ones.',
                                'priority': 'High',
                                'expected_impact': 'Reduce acquisition cost by 15%',
                                'implementation_time': '2-3 weeks'
                            },
                            {
                                'title': 'Customer Segmentation',
                                'description': 'Develop targeted campaigns based on customer segments and behavior.',
                                'priority': 'Medium',
                                'expected_impact': 'Improve ROI by 25%',
                                'implementation_time': '2 months'
                            }
                        ]
                    }
                )
        else:
            return render_template('marketing_dashboard.html',
                metrics={
                    'total_campaigns': 0,
                    'conversion_rate': 0,
                    'customer_acquisition_cost': 0,
                    'roi': 0
                },
                campaign_performance={},
                customer_insights={},
                channel_effectiveness={},
                analysis={
                    'insights': [],
                    'recommendations': []
                }
            )
    except Exception as e:
        print("Marketing Dashboard Error:", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/it-dashboard')
@login_required
@department_access_required('IT')
def it_dashboard():
    try:
        json_path = os.path.join('Outputs', 'it_analytics_summary.json')
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return render_template('it_dashboard.html',
                    metrics={
                        'total_tickets': data.get('metrics', {}).get('total_tickets', 0),
                        'avg_resolution_time': data.get('metrics', {}).get('avg_resolution_time', 0),
                        'first_call_resolution': data.get('metrics', {}).get('first_call_resolution', 0),
                        'system_uptime': data.get('metrics', {}).get('system_uptime', 0)
                    },
                    distributions={
                        'ticket_types': data.get('distributions', {}).get('ticket_types', {})
                    },
                    performance={
                        'by_system': data.get('performance', {}).get('by_system', {})
                    },
                    trends={
                        'monthly': data.get('trends', {}).get('monthly', {}),
                        'performance': data.get('trends', {}).get('performance', {})
                    },
                    analysis={
                        'insights': data.get('analysis', {}).get('insights', []),
                        'recommendations': data.get('analysis', {}).get('recommendations', [])
                    },
                    recent_tickets=data.get('recent_tickets', [])
                )
        else:
            return render_template('it_dashboard.html',
                metrics={
                    'total_tickets': 0,
                    'avg_resolution_time': 0,
                    'first_call_resolution': 0,
                    'system_uptime': 0
                },
                distributions={
                    'ticket_types': {}
                },
                performance={
                    'by_system': {}
                },
                trends={
                    'monthly': {},
                    'performance': {}
                },
                analysis={
                    'insights': [],
                    'recommendations': []
                },
                recent_tickets=[]
            )
    except Exception as e:
        flash(f'Error loading IT dashboard: {str(e)}', 'error')
    return render_template('it_dashboard.html')

@app.route('/production-dashboard')
@login_required
@department_access_required('Production')
def production_dashboard():
    try:
        json_path = os.path.join('Outputs', 'production_analytics_summary.json')
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return render_template('production_dashboard.html',
                    metrics={
                        'total_operations': data.get('metrics', {}).get('total_operations', 0),
                        'avg_efficiency': data.get('metrics', {}).get('avg_efficiency', 0),
                        'avg_quality_score': data.get('metrics', {}).get('avg_quality_score', 0),
                        'avg_machine_utilization': data.get('metrics', {}).get('avg_machine_utilization', 0)
                    },
                    distributions={
                        'product_lines': data.get('distributions', {}).get('product_lines', {})
                    },
                    performance={
                        'by_line': data.get('performance', {}).get('by_line', {})
                    },
                    trends=data.get('trends', {}),
                    analysis=data.get('analysis', {'insights': [], 'recommendations': []}),
                    recent_operations=data.get('recent_operations', [])
                )
        else:
            return render_template('production_dashboard.html',
                metrics={
                    'total_operations': 0,
                    'avg_efficiency': 0,
                    'avg_quality_score': 0,
                    'avg_machine_utilization': 0
                },
                distributions={
                    'product_lines': {}
                },
                performance={
                    'by_line': {}
                },
                trends={},
                analysis={'insights': [], 'recommendations': []},
                recent_operations=[]
            )
    except Exception as e:
        flash(f'Error loading Production dashboard: {str(e)}', 'error')
    return render_template('production_dashboard.html',
        metrics={
            'total_operations': 0,
            'avg_efficiency': 0,
            'avg_quality_score': 0,
            'avg_machine_utilization': 0
        },
        distributions={
            'product_lines': {}
        },
        performance={
            'by_line': {}
        },
        trends={},
        analysis={'insights': [], 'recommendations': []},
        recent_operations=[]
    )

@app.route('/sales-dashboard')
@login_required
@department_access_required('Sales')
def sales_dashboard():
    try:
        json_path = os.path.join('Outputs', 'sales_analytics_summary.json')
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                metrics = data.get('metrics', {})
                # Always provide avg_unit_price
                metrics.setdefault('avg_unit_price', 0)
                return render_template('sales_dashboard.html',
                    metrics={
                        'total_sales': metrics.get('total_sales', 0),
                        'total_orders': metrics.get('total_orders', 0),
                        'avg_order_value': metrics.get('avg_order_value', 0),
                        'conversion_rate': metrics.get('conversion_rate', 0),
                        'avg_unit_price': metrics.get('avg_unit_price', 0)
                    },
                    distributions=data.get('distributions', {}),
                    performance=data.get('performance', {}),
                    trends=data.get('trends', {}),
                    analysis=data.get('analysis', {'insights': [], 'recommendations': []}),
                    top_products=data.get('top_products', [])
                )
        else:
            return render_template('sales_dashboard.html',
                metrics={
                    'total_sales': 0,
                    'total_orders': 0,
                    'avg_order_value': 0,
                    'conversion_rate': 0,
                    'avg_unit_price': 0
                },
                distributions={},
                performance={},
                trends={},
                analysis={'insights': [], 'recommendations': []},
                top_products=[]
            )
    except Exception as e:
        flash(f'Error loading Sales dashboard: {str(e)}', 'error')
    return render_template('sales_dashboard.html',
        metrics={
            'total_sales': 0,
            'total_orders': 0,
            'avg_order_value': 0,
            'conversion_rate': 0,
            'avg_unit_price': 0
        },
        distributions={},
        performance={},
        trends={},
        analysis={'insights': [], 'recommendations': []},
        top_products=[]
    )

# API Routes for Department Data
@app.route('/api/supply-chain/metrics')
def get_supply_chain_metrics():
    try:
        # Check if JSON file exists
        json_path = os.path.join('Outputs', 'supply_chain_analytics_summary.json')
        if os.path.exists(json_path):
            # Load existing JSON data
            with open(json_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
        else:
            # Generate new analytics if JSON doesn't exist
            results = run_supply_chain_analytics()
        
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/hr/metrics')
def get_hr_metrics():
    try:
        json_path = os.path.join('Outputs', 'hr_analytics_summary.json')
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                return jsonify(json.load(f))
        return jsonify({'error': 'HR analytics not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/transport/metrics')
def get_transport_metrics():
    try:
        json_path = os.path.join('Outputs', 'transport_analytics_summary.json')
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                return jsonify(json.load(f))
        return jsonify({'error': 'Transport analytics not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/finance/metrics')
def get_finance_metrics():
    try:
        json_path = os.path.join('Outputs', 'finance_analytics_summary.json')
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                return jsonify(json.load(f))
        return jsonify({'error': 'Finance analytics not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/marketing/metrics')
def get_marketing_metrics():
    try:
        json_path = os.path.join('Outputs', 'marketing_analytics_summary.json')
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                return jsonify(json.load(f))
        return jsonify({'error': 'Marketing analytics not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/it/metrics')
def get_it_metrics():
    try:
        json_path = os.path.join('Outputs', 'it_analytics_summary.json')
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                return jsonify(json.load(f))
        return jsonify({'error': 'IT analytics not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/production/metrics')
def get_production_metrics():
    try:
        json_path = os.path.join('Outputs', 'production_analytics_summary.json')
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                return jsonify(json.load(f))
        return jsonify({'error': 'Production analytics not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sales/metrics')
def get_sales_metrics():
    try:
        json_path = os.path.join('Outputs', 'sales_analytics_summary.json')
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                return jsonify(json.load(f))
        return jsonify({'error': 'Sales analytics not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('logout.html'))

@app.route('/what-if-analysis')
@login_required
def what_if_analysis():
    return render_template('what_if_analysis.html')

@app.route('/download_report/hr')
@login_required
@department_access_required('HR')
def download_hr_report():
    json_path = os.path.join('Outputs', 'hr_analytics_summary.json')
    if not os.path.exists(json_path):
        return 'No analytics summary found.', 404
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Prepare DataFrames for each section
    metrics_df = pd.DataFrame([data.get('metrics', {})])
    analysis_df = pd.DataFrame({'Insights': data.get('analysis', {}).get('insights', []),
                                'Recommendations': data.get('analysis', {}).get('recommendations', [])})
    # Optionally add more sections (trends, distributions, etc.)
    # Write to Excel in memory
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        metrics_df.to_excel(writer, index=False, sheet_name='Key Metrics')
        analysis_df.to_excel(writer, index=False, sheet_name='Analysis')
    output.seek(0)
    return send_file(output, as_attachment=True, download_name='hr_analytics_report.xlsx', mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

def get_summary_path(department):
    return os.path.join('Outputs', f'{department}_analytics_summary.json')

@app.route('/download_report/<department>')
@login_required
@department_access_required('<department>')
def download_report(department):
    fmt = request.args.get('format', 'csv')
    json_path = get_summary_path(department)
    if not os.path.exists(json_path):
        return 'No analytics summary found.', 404
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if fmt == 'csv':
        # Example: export metrics and predictions
        metrics_df = pd.DataFrame([data.get('metrics', {})])
        output = BytesIO()
        metrics_df.to_csv(output, index=False)
        output.seek(0)
        return send_file(output, as_attachment=True, download_name=f'{department}_analytics_metrics.csv', mimetype='text/csv')
    elif fmt == 'pdf':
        return 'PDF export not implemented yet.', 501
    else:
        return 'Invalid format.', 400

if __name__ == '__main__':
    app.run(debug=True)

