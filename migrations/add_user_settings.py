import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.database import db
from run import app  # Import the Flask app from run.py
from sqlalchemy import text

def upgrade():
    with app.app_context():
        with db.engine.connect() as conn:
            # Add only the missing columns
            columns = [
                "ADD COLUMN first_name VARCHAR(50)",
                "ADD COLUMN last_name VARCHAR(50)"
            ]
            
            for column in columns:
                try:
                    conn.execute(text(f"ALTER TABLE users {column}"))
                except Exception as e:
                    print(f"Error adding column {column}: {str(e)}")
                    continue
            
            conn.commit()

def downgrade():
    with app.app_context():
        with db.engine.connect() as conn:
            # Remove only the columns we added
            columns = [
                "first_name",
                "last_name"
            ]
            
            for column in columns:
                try:
                    conn.execute(text(f"ALTER TABLE users DROP COLUMN {column}"))
                except Exception as e:
                    print(f"Error removing column {column}: {str(e)}")
                    continue
            
            conn.commit()

if __name__ == '__main__':
    upgrade() 