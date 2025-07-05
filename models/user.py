from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from models.database import db

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(150), nullable=False)
    first_name = db.Column(db.String(50))
    last_name = db.Column(db.String(50))
    role = db.Column(db.String(50), nullable=False, default='user')
    department = db.Column(db.String(100), nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    theme = db.Column(db.String(10), default='light')  # 'light' or 'dark'
    font_size = db.Column(db.String(10), default='medium')  # 'small', 'medium', or 'large'
    
    # Notification settings
    email_notifications = db.Column(db.Boolean, default=True)
    dashboard_alerts = db.Column(db.Boolean, default=True)
    report_updates = db.Column(db.Boolean, default=True)
    system_updates = db.Column(db.Boolean, default=True)
    
    # Appearance settings
    compact_view = db.Column(db.Boolean, default=False)
    
    # Security settings
    two_factor = db.Column(db.Boolean, default=False)
    
    # Data management settings
    data_retention = db.Column(db.Integer, default=90)
    auto_backup = db.Column(db.Boolean, default=True)
    backup_frequency = db.Column(db.String(20), default='weekly')
    
    # Integration settings
    mongo_integration = db.Column(db.Boolean, default=True)
    api_integration = db.Column(db.Boolean, default=False)
    api_key = db.Column(db.String(64))

    logs = db.relationship('Log', back_populates='user', cascade="all, delete-orphan")

    def __init__(self, email, password, role, department, first_name=None, last_name=None, is_active=True):
        self.email = email
        self.set_password(password)
        self.role = role
        self.department = department
        self.first_name = first_name
        self.last_name = last_name
        self.is_active = is_active
        self.last_login = None
        self.created_at = datetime.utcnow()

    @staticmethod
    def get_by_email(email):
        return User.query.filter_by(email=email).first()

    @staticmethod
    def get_by_id(user_id):
        return User.query.get(int(user_id))

    def get_id(self):
        return str(self.id)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def save(self):
        db.session.add(self)
        db.session.commit()

    def update_last_login(self):
        self.last_login = datetime.utcnow()
        self.save()

    @staticmethod
    def create_user(email, password, role, department, first_name=None, last_name=None):
        if User.get_by_email(email):
            return None
        
        user = User(
            email=email,
            password=password,
            role=role,
            department=department,
            first_name=first_name,
            last_name=last_name
        )
        user.save()
        return user

    def is_admin(self):
        return self.role == 'admin'

    def is_manager(self):
        return self.role == 'manager'

    def to_dict(self):
        return {
            'id': self.id,
            'email': self.email,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'role': self.role,
            'department': self.department,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'is_active': self.is_active,
            'email_notifications': self.email_notifications,
            'dashboard_alerts': self.dashboard_alerts,
            'report_updates': self.report_updates,
            'system_updates': self.system_updates,
            'theme': self.theme,
            'font_size': self.font_size,
            'compact_view': self.compact_view,
            'two_factor': self.two_factor,
            'data_retention': self.data_retention,
            'auto_backup': self.auto_backup,
            'backup_frequency': self.backup_frequency,
            'mongo_integration': self.mongo_integration,
            'api_integration': self.api_integration,
            'api_key': self.api_key
        }

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def update_profile(self, data):
        """Update user profile information"""
        if 'first_name' in data:
            self.first_name = data['first_name']
        if 'last_name' in data:
            self.last_name = data['last_name']
        if 'email' in data:
            self.email = data['email']
        if 'role' in data:
            self.role = data['role']
        if 'department' in data:
            self.department = data['department']
        if 'theme' in data:
            self.theme = data['theme']
        if 'font_size' in data:
            self.font_size = data['font_size']
        db.session.commit() 