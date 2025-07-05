from flask import Blueprint, render_template, request, flash, redirect, url_for, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
import os
from routes.auth import admin_required
from models.user import User
from models.log import Log
from models.uploaded_dataset import UploadedDataset
from models.database import db

uploads_bp = Blueprint('uploads_bp', __name__)

ALLOWED_EXTENSIONS = {'csv'}
DATASET_FILENAMES = {
    'hr': 'hr_data.csv',
    'sales': 'sales_data.csv',
    'production': 'production_data.csv',
    'supply_chain': 'supply_chain_data.csv',
    'transport': 'transport_data.csv',
    'finance': 'finance_data.csv',
    'it': 'it_data.csv'
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@uploads_bp.route('/uploads', methods=['GET', 'POST'])
@login_required
@admin_required
def uploads():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        dataset_type = request.form.get('dataset_type')

        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)

        if not dataset_type or dataset_type not in DATASET_FILENAMES:
            flash('Invalid dataset type', 'error')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            department = request.form.get('department')
            
            if not department:
                flash('Department is required.', 'danger')
                return redirect(request.url)

            # Create department-specific folder if it doesn't exist
            department_folder = os.path.join(app.config['UPLOAD_FOLDER'], department)
            if not os.path.exists(department_folder):
                os.makedirs(department_folder)

            file_path = os.path.join(department_folder, filename)
            file.save(file_path)

            # Add to database
            new_dataset = UploadedDataset(
                filename=filename,
                department=department,
                user_id=current_user.id
            )
            db.session.add(new_dataset)
            db.session.commit()

            # Add log
            Log.add_log(
                user_id=current_user.id,
                action='upload_dataset',
                details=f'Uploaded {filename} to {department}'
            )

            flash(f'File "{filename}" uploaded successfully to {department}!', 'success')
            return redirect(url_for('uploads_bp.uploads'))
        else:
            flash('Invalid file type. Allowed types are: txt, csv, xlsx, json', 'danger')
            return redirect(request.url)

    return render_template('uploads.html') 