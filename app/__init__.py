from flask import Flask
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ddss.db'  # Change as needed
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    CORS(app)
    db.init_app(app)

    # Register blueprints
    from .routes.routes import routes
    app.register_blueprint(routes)

    return app 