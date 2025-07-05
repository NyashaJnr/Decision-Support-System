from app import create_app
from app.__init__ import db
from app.models import schema  # Ensures models are imported

app = create_app()
with app.app_context():
    db.create_all()
    print("✅ Database tables created successfully.") 