from models.database import db
from datetime import datetime

class UploadedDataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200), nullable=False)
    department = db.Column(db.String(100), nullable=False)
    upload_timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    user = db.relationship('User')

    def __repr__(self):
        return f'<UploadedDataset {self.filename}>' 