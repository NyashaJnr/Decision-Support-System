from models.user import User

def init_db():
    # Create admin user
    admin = User.create_user(
        email='admin@example.com',
        password='admin123',
        role='admin',
        department='Management'
    )
    
    # Create manager user
    manager = User.create_user(
        email='manager@example.com',
        password='manager123',
        role='manager',
        department='Operations'
    )
    
    if admin and manager:
        print("Database initialized successfully!")
        print("Admin user created:", admin.email)
        print("Manager user created:", manager.email)
    else:
        print("Error initializing database!")

if __name__ == '__main__':
    init_db() 