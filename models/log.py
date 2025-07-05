from models.database import db
from datetime import datetime

class Log(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    action = db.Column(db.String(200), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    details = db.Column(db.String(500))

    user = db.relationship('User', back_populates='logs')

    def __repr__(self):
        return f'<Log {self.id} - {self.action}>'

    @staticmethod
    def add_log(user_id, action, details=''):
        """Helper to add a new log entry."""
        new_log = Log(user_id=user_id, action=action, details=details)
        db.session.add(new_log)
        db.session.commit() 