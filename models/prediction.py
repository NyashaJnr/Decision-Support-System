from models.database import db
from datetime import datetime

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    prediction_type = db.Column(db.String(100), nullable=False)
    input_data = db.Column(db.JSON)
    prediction_result = db.Column(db.JSON)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True) # Can be nullable if predictions are automated

    user = db.relationship('User')

    def __repr__(self):
        return f'<Prediction {self.id} - {self.prediction_type}>' 