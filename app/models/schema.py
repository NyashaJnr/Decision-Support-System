from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from app.__init__ import db

class User(db.Model):
    __tablename__ = 'users'
    UserID = db.Column(db.Integer, primary_key=True)
    Name = db.Column(db.String(100), nullable=False)
    Email = db.Column(db.String(120), unique=True, nullable=False)
    Password = db.Column(db.String(200), nullable=False)
    Role = db.Column(db.String(50), nullable=False)
    Department = db.Column(db.String(100), nullable=True)
    # Relationships
    datasets = db.relationship('Dataset', backref='uploader', lazy=True)
    predictions = db.relationship('Prediction', backref='user', lazy=True)

class Prediction(db.Model):
    __tablename__ = 'predictions'
    PredictionID = db.Column(db.Integer, primary_key=True)
    DepartmentID = db.Column(db.Integer, db.ForeignKey('users.UserID'), nullable=False)
    Model = db.Column(db.String(100), nullable=False)
    InputData = db.Column(db.Text, nullable=False)
    PredictionResult = db.Column(db.Text, nullable=False)

class Dataset(db.Model):
    __tablename__ = 'datasets'
    DatasetID = db.Column(db.Integer, primary_key=True)
    Department = db.Column(db.String(100), nullable=False)
    UploadedBy = db.Column(db.Integer, db.ForeignKey('users.UserID'), nullable=False)
    DateUploaded = db.Column(db.DateTime, default=datetime.utcnow)

class Report(db.Model):
    __tablename__ = 'reports'
    ReportID = db.Column(db.Integer, primary_key=True)
    Title = db.Column(db.String(200), nullable=False)
    Department = db.Column(db.String(100), nullable=False)
    FilePath = db.Column(db.String(300), nullable=False) 