from flask import Blueprint, render_template, request, flash, redirect, url_for, Response
from flask_login import login_required, current_user
from models.user import User
from models.database import db
from models.log import Log
from routes.auth import admin_required
import csv
import io

user_management_bp = Blueprint('user_management_bp', __name__)

@user_management_bp.route('/user-management')
@login_required
@admin_required
def user_management():
    users = User.query.all()
    return render_template('user_management.html', users=users)

@user_management_bp.route('/user-management/add', methods=['POST'])
@login_required
@admin_required
def add_user():
    email = request.form.get('email')
    password = request.form.get('password')
    department = request.form.get('department')
    role = request.form.get('role')

    if role == 'admin':
        department = 'General Manager'

    if User.get_by_email(email):
        flash('Email address already exists.', 'error')
    else:
        new_user = User.create_user(email=email, password=password, department=department, role=role)
        if new_user:
            db.session.add(new_user)
            db.session.commit()
            Log.add_log(current_user.id, 'add_user', f'Added new user: {email}')
            flash('New user created successfully!', 'success')
        else:
            flash('Error creating user.', 'error')
            
    return redirect(url_for('user_management_bp.user_management'))

@user_management_bp.route('/user-management/edit/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def edit_user(user_id):
    user = User.get_by_id(user_id)
    if user:
        email = request.form.get('email')
        department = request.form.get('department')
        role = request.form.get('role')
        if role == 'admin':
            department = 'General Manager'
        # Check if email is being changed to one that already exists
        if email != user.email and User.get_by_email(email):
            flash('That email is already registered to another user.', 'error')
            return redirect(url_for('user_management_bp.user_management'))

        user.email = email
        user.department = department
        user.role = role
        if request.form.get('password'):
            user.set_password(request.form.get('password'))
        db.session.commit()
        Log.add_log(current_user.id, 'edit_user', f'Edited user ID: {user_id}')
        flash('User updated successfully!', 'success')
    else:
        flash('User not found.', 'error')
        
    return redirect(url_for('user_management_bp.user_management'))

@user_management_bp.route('/user-management/delete/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def delete_user(user_id):
    user = User.get_by_id(user_id)
    if user:
        email = user.email
        db.session.delete(user)
        db.session.commit()
        Log.add_log(current_user.id, 'delete_user', f'Deleted user: {email} (ID: {user_id})')
        flash('User deleted successfully!', 'success')
    else:
        flash('User not found.', 'error')
        
    return redirect(url_for('user_management_bp.user_management'))

@user_management_bp.route('/user-management/download')
@login_required
@admin_required
def download_users():
    users = User.query.all()
    output = io.StringIO()
    writer = csv.writer(output)
    
    writer.writerow(['ID', 'Email', 'Department', 'Role'])
    for user in users:
        writer.writerow([user.id, user.email, user.department, user.role])
    
    output.seek(0)
    
    return Response(
        output,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=users.csv"}
    ) 