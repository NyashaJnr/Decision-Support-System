from flask import Blueprint, request, jsonify, session, redirect, url_for, flash, render_template
from flask_login import login_user, logout_user, login_required, current_user
from models.database import db
from models.user import User
from functools import wraps
import os
import time
import random
import string

auth = Blueprint('auth', __name__)

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            flash('Please log in to access this page.', 'danger')
            return redirect(url_for('auth.login'))
        
        if not current_user.is_admin():
            flash('You do not have permission to access this page.', 'danger')
            return redirect(url_for('auth.dashboard'))
        return f(*args, **kwargs)
    return decorated_function

def department_access_required(department):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated:
                flash('Please log in to access this page.', 'danger')
                return redirect(url_for('auth.login'))
            
            if not current_user.is_admin() and current_user.department != department:
                flash('You do not have permission to access this department\'s data.', 'danger')
                return redirect(url_for('auth.dashboard'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def manager_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            flash('Please log in to access this page.', 'danger')
            return redirect(url_for('auth.login'))
        
        if not current_user.is_manager():
            flash('You do not have permission to access this page.', 'danger')
            return redirect(url_for('auth.dashboard'))
        return f(*args, **kwargs)
    return decorated_function

@auth.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('auth.dashboard'))
        
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        role = request.form.get('role')
        remember = request.form.get('remember') == 'on'

        user = User.get_by_email(email)
        
        if user and user.check_password(password) and user.role == role:
            if not user.is_active:
                flash('Your account has been deactivated. Please contact support.', 'danger')
                return redirect(url_for('auth.login'))

            login_user(user, remember=remember)
            user.update_last_login()
            
            flash('Login successful!', 'success')
            return redirect(url_for('auth.dashboard'))
        else:
            flash('Invalid email, password, or role.', 'danger')
            return redirect(url_for('auth.login'))

    return render_template('login.html')

@auth.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out successfully.', 'success')
    return redirect(url_for('auth.login'))

@auth.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@auth.route('/profile')
@login_required
def profile():
    return render_template('profile.html', user=current_user)

@auth.route('/change-password', methods=['POST'])
@login_required
def change_password():
    current_password = request.form.get('current_password')
    new_password = request.form.get('new_password')
    confirm_password = request.form.get('confirm_password')

    if not current_user.check_password(current_password):
        flash('Current password is incorrect.', 'danger')
        return redirect(url_for('auth.profile'))

    if new_password != confirm_password:
        flash('New passwords do not match.', 'danger')
        return redirect(url_for('auth.profile'))

    if len(new_password) < 6:
        flash('New password must be at least 6 characters long.', 'danger')
        return redirect(url_for('auth.profile'))

    current_user.set_password(new_password)
    db.session.commit()

    flash('Password changed successfully.', 'success')
    return redirect(url_for('auth.profile'))

@auth.route('/check-cooldown')
def check_cooldown():
    if 'login_attempts' in session:
        last_attempt = session['login_attempts'].get('timestamp', 0)
        cooldown_time = 30  # 30 seconds cooldown
        current_time = time.time()
        remaining_time = int(cooldown_time - (current_time - last_attempt))
        
        if remaining_time > 0:
            return jsonify({
                'cooldown': True,
                'remaining_time': remaining_time
            })
    
    return jsonify({
        'cooldown': False,
        'remaining_time': 0
    })

@auth.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        role = request.form.get('role')
        department = request.form.get('department')
        password = request.form.get('password')  # Get password for general manager

        # Validate required fields
        if not all([email, first_name, last_name, role]):
            flash('Please fill in all required fields', 'error')
            return redirect(url_for('auth.register'))

        # Check if email already exists
        if User.get_by_email(email):
            flash('Email already registered', 'error')
            return redirect(url_for('auth.register'))

        try:
            # Create user with password if role is admin (general manager)
            if role == 'admin':
                if not password:
                    flash('Password is required for General Manager accounts', 'error')
                    return redirect(url_for('auth.register'))
                user = User.create_user(
                    email=email,
                    password=password,
                    role=role,
                    department=department,
                    first_name=first_name,
                    last_name=last_name
                )
            else:
                # For departmental managers, generate a temporary password
                temp_password = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
                user = User.create_user(
                    email=email,
                    password=temp_password,
                    role=role,
                    department=department,
                    first_name=first_name,
                    last_name=last_name
                )
                # TODO: Send email with temporary password to user

            if user:
                db.session.commit()  # Ensure changes are committed to the database
                flash('Registration successful!', 'success')
                return redirect(url_for('auth.login'))
            else:
                flash('Registration failed', 'error')
                return redirect(url_for('auth.register'))
        except Exception as e:
            db.session.rollback()  # Rollback in case of error
            flash(f'Registration failed: {str(e)}', 'error')
            return redirect(url_for('auth.register'))

    return render_template('register.html')

@auth.route('/department-dashboard')
@login_required
def department_dashboard():
    if current_user.is_admin():
        return redirect(url_for('auth.dashboard'))
    return render_template('department_dashboard.html')

@auth.route('/api/department/<department>/metrics')
@login_required
def get_department_metrics(department):
    if not current_user.is_admin() and current_user.department != department:
        return jsonify({'error': 'Unauthorized'}), 403
    
    # TODO: Replace with actual database queries
    metrics = {
        'efficiency': 85,
        'taskCompletion': 92,
        'resourceUtilization': 78,
        'qualityScore': 88
    }
    return jsonify(metrics)

@auth.route('/api/department/<department>/activities')
@login_required
def get_department_activities(department):
    if not current_user.is_admin() and current_user.department != department:
        return jsonify({'error': 'Unauthorized'}), 403
    
    # TODO: Replace with actual database queries
    activities = [
        {
            'date': '2024-03-15',
            'description': 'Completed Q1 Performance Review',
            'status': 'Completed',
            'impact': 'High'
        },
        {
            'date': '2024-03-14',
            'description': 'Resource Allocation Meeting',
            'status': 'In Progress',
            'impact': 'Medium'
        },
        {
            'date': '2024-03-13',
            'description': 'Team Training Session',
            'status': 'Completed',
            'impact': 'High'
        }
    ]
    return jsonify({'activities': activities})

@auth.route('/about')
@login_required
def about():
    return render_template('about.html') 