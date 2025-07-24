"""
Database Setup Script for NL to SQL Agent
Creates PostgreSQL database with sample payment data for testing
"""

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import pandas as pd
from datetime import datetime, timedelta
import random
from faker import Faker
import os

# Database configuration
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "nlsql_db"
DB_USER = "nlsql_user"
DB_PASSWORD = "nlsql_password"

# Sample data configuration
fake = Faker()
Faker.seed(42)  # For reproducible data
random.seed(42)

class DatabaseSetup:
    """Setup and populate PostgreSQL database"""
    
    def __init__(self):
        self.connection = None
        
    def create_database_and_user(self):
        """Create database and user if they don't exist"""
        try:
            # Connect as postgres superuser (you may need to adjust these credentials)
            conn = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                database="postgres",
                user="postgres",
                password="postgres"  # Change this to your postgres password
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            # Create user if not exists
            try:
                cursor.execute(f"""
                CREATE USER {DB_USER} WITH PASSWORD '{DB_PASSWORD}';
                """)
                print(f"✅ Created user: {DB_USER}")
            except psycopg2.Error as e:
                if "already exists" in str(e):
                    print(f"ℹ️  User {DB_USER} already exists")
                else:
                    raise
            
            # Create database if not exists
            try:
                cursor.execute(f"""
                CREATE DATABASE {DB_NAME} OWNER {DB_USER};
                """)
                print(f"✅ Created database: {DB_NAME}")
            except psycopg2.Error as e:
                if "already exists" in str(e):
                    print(f"ℹ️  Database {DB_NAME} already exists")
                else:
                    raise
            
            # Grant privileges
            cursor.execute(f"""
            GRANT ALL PRIVILEGES ON DATABASE {DB_NAME} TO {DB_USER};
            """)
            
            cursor.close()
            conn.close()
            
        except psycopg2.Error as e:
            print(f"❌ Error creating database/user: {e}")
            print("Make sure PostgreSQL is running and you have the correct postgres user password")
            raise
    
    def connect_to_app_database(self):
        """Connect to the application database"""
        try:
            self.connection = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD
            )
            print(f"✅ Connected to {DB_NAME}")
        except psycopg2.Error as e:
            print(f"❌ Error connecting to database: {e}")
            raise
    
    def create_tables(self):
        """Create database tables"""
        cursor = self.connection.cursor()
        
        # Drop tables if they exist (for clean setup)
        drop_tables_sql = """
        DROP TABLE IF EXISTS payment_items CASCADE;
        DROP TABLE IF EXISTS payments CASCADE;
        DROP TABLE IF EXISTS merchants CASCADE;
        DROP TABLE IF EXISTS categories CASCADE;
        DROP TABLE IF EXISTS users CASCADE;
        """
        
        # Create tables
        create_tables_sql = """
        -- Users table
        CREATE TABLE users (
            user_id VARCHAR(100) PRIMARY KEY,
            username VARCHAR(100) UNIQUE NOT NULL,
            email VARCHAR(255) UNIQUE NOT NULL,
            first_name VARCHAR(100) NOT NULL,
            last_name VARCHAR(100) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Categories table
        CREATE TABLE categories (
            category_id SERIAL PRIMARY KEY,
            category_name VARCHAR(100) UNIQUE NOT NULL,
            category_type VARCHAR(50) -- 'entertainment', 'food', 'transport', etc.
        );
        
        -- Merchants table
        CREATE TABLE merchants (
            merchant_id SERIAL PRIMARY KEY,
            merchant_name VARCHAR(255) NOT NULL,
            merchant_type VARCHAR(100),
            category_id INTEGER REFERENCES categories(category_id)
        );
        
        -- Payments table
        CREATE TABLE payments (
            payment_id SERIAL PRIMARY KEY,
            user_id VARCHAR(100) REFERENCES users(user_id),
            merchant_id INTEGER REFERENCES merchants(merchant_id),
            amount DECIMAL(10,2) NOT NULL,
            currency VARCHAR(3) DEFAULT 'USD',
            payment_date TIMESTAMP NOT NULL,
            payment_method VARCHAR(50), -- 'card', 'cash', 'transfer', etc.
            description TEXT,
            status VARCHAR(20) DEFAULT 'completed',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Payment items table (for detailed breakdown)
        CREATE TABLE payment_items (
            item_id SERIAL PRIMARY KEY,
            payment_id INTEGER REFERENCES payments(payment_id) ON DELETE CASCADE,
            item_name VARCHAR(255) NOT NULL,
            item_category VARCHAR(100),
            quantity INTEGER DEFAULT 1,
            unit_price DECIMAL(10,2) NOT NULL,
            total_price DECIMAL(10,2) NOT NULL
        );
        
        -- Create indexes for better performance
        CREATE INDEX idx_payments_user_date ON payments(user_id, payment_date);
        CREATE INDEX idx_payments_merchant ON payments(merchant_id);
        CREATE INDEX idx_payments_date ON payments(payment_date);
        CREATE INDEX idx_payments_amount ON payments(amount);
        """
        
        try:
            cursor.execute(drop_tables_sql)
            cursor.execute(create_tables_sql)
            self.connection.commit()
            print("✅ Tables created successfully")
        except psycopg2.Error as e:
            print(f"❌ Error creating tables: {e}")
            self.connection.rollback()
            raise
        finally:
            cursor.close()
    
    def generate_sample_data(self):
        """Generate realistic sample payment data"""
        
        # Categories data
        categories_data = [
            ('Food & Dining', 'food'),
            ('Entertainment', 'entertainment'),
            ('Transportation', 'transport'),
            ('Shopping', 'shopping'),
            ('Utilities', 'utilities'),
            ('Healthcare', 'healthcare'),
            ('Education', 'education'),
            ('Travel', 'travel'),
            ('Groceries', 'food'),
            ('Gas & Fuel', 'transport'),
            ('Subscriptions', 'entertainment'),
            ('Coffee & Cafes', 'food')
        ]
        
        # Merchants data with realistic names
        merchants_data = [
            ('Netflix', 'Streaming Service', 'Entertainment'),
            ('Spotify', 'Music Streaming', 'Entertainment'),
            ('Amazon', 'Online Shopping', 'Shopping'),
            ('Starbucks', 'Coffee Shop', 'Coffee & Cafes'),
            ('McDonald\'s', 'Fast Food', 'Food & Dining'),
            ('Uber', 'Rideshare', 'Transportation'),
            ('Shell', 'Gas Station', 'Gas & Fuel'),
            ('Whole Foods', 'Grocery Store', 'Groceries'),
            ('AMC Theatres', 'Movie Theater', 'Entertainment'),
            ('Target', 'Retail Store', 'Shopping'),
            ('Walgreens', 'Pharmacy', 'Healthcare'),
            ('Apple Store', 'Electronics', 'Shopping'),
            ('Chipotle', 'Fast Casual', 'Food & Dining'),
            ('Steam', 'Gaming Platform', 'Entertainment'),
            ('Disney+', 'Streaming Service', 'Entertainment'),
            ('Costco', 'Warehouse Store', 'Groceries'),
            ('CVS Pharmacy', 'Pharmacy', 'Healthcare'),
            ('Panera Bread', 'Bakery Cafe', 'Food & Dining'),
            ('Best Buy', 'Electronics Store', 'Shopping'),
            ('Dunkin\'', 'Coffee Shop', 'Coffee & Cafes')
        ]
        
        # Users data with string user_ids
        users_data = [
            ('user_001', 'john_doe', 'john.doe@email.com', 'John', 'Doe'),
            ('user_002', 'jane_smith', 'jane.smith@email.com', 'Jane', 'Smith'),
            ('user_003', 'test_user', 'test@email.com', 'Test', 'User')
        ]
        
        cursor = self.connection.cursor()
        
        try:
            # Insert categories
            for category_name, category_type in categories_data:
                cursor.execute("""
                    INSERT INTO categories (category_name, category_type) 
                    VALUES (%s, %s)
                """, (category_name, category_type))
            
            # Insert users
            for user_id, username, email, first_name, last_name in users_data:
                cursor.execute("""
                    INSERT INTO users (user_id, username, email, first_name, last_name) 
                    VALUES (%s, %s, %s, %s, %s)
                """, (user_id, username, email, first_name, last_name))
            
            # Insert merchants
            for merchant_name, merchant_type, category_name in merchants_data:
                cursor.execute("""
                    INSERT INTO merchants (merchant_name, merchant_type, category_id)
                    SELECT %s, %s, category_id FROM categories WHERE category_name = %s
                """, (merchant_name, merchant_type, category_name))
            
            self.connection.commit()
            print("✅ Master data inserted successfully")
            
            # Generate payments for the last 12 months
            self.generate_payment_data(cursor)
            
        except psycopg2.Error as e:
            print(f"❌ Error inserting sample data: {e}")
            self.connection.rollback()
            raise
        finally:
            cursor.close()
    
    def generate_payment_data(self, cursor):
        """Generate realistic payment transactions"""
        
        # Get user IDs (now strings)
        cursor.execute("SELECT user_id FROM users")
        user_ids = [row[0] for row in cursor.fetchall()]
        
        cursor.execute("SELECT merchant_id, merchant_name FROM merchants")
        merchants = cursor.fetchall()
        
        # Payment methods and their weights
        payment_methods = ['card', 'card', 'card', 'card', 'cash', 'transfer']  # More cards
        
        # Generate payments for the last 12 months
        start_date = datetime.now() - timedelta(days=365)
        end_date = datetime.now()
        
        payments_to_insert = []
        payment_items_to_insert = []
        payment_id_counter = 1
        
        # Generate 500-1000 transactions per user
        for user_id in user_ids:
            num_payments = random.randint(500, 1000)
            
            for _ in range(num_payments):
                # Random date within the last year
                random_date = start_date + timedelta(
                    seconds=random.randint(0, int((end_date - start_date).total_seconds()))
                )
                
                # Select random merchant
                merchant_id, merchant_name = random.choice(merchants)
                
                # Generate realistic amount based on merchant type
                amount = self.generate_realistic_amount(merchant_name)
                
                # Random payment method
                payment_method = random.choice(payment_methods)
                
                # Generate description
                description = self.generate_payment_description(merchant_name, amount)
                
                payment_data = (
                    payment_id_counter,  # payment_id
                    user_id,  # now a string
                    merchant_id,
                    amount,
                    'USD',
                    random_date,
                    payment_method,
                    description,
                    'completed'
                )
                payments_to_insert.append(payment_data)
                
                # Generate payment items (some payments have itemized breakdown)
                if random.random() < 0.3:  # 30% chance of itemized payment
                    num_items = random.randint(1, 4)
                    remaining_amount = float(amount)
                    
                    for i in range(num_items):
                        if i == num_items - 1:  # Last item gets remaining amount
                            item_price = round(remaining_amount, 2)
                        else:
                            item_price = round(random.uniform(5, remaining_amount * 0.4), 2)
                            remaining_amount -= item_price
                        
                        item_name = self.generate_item_name(merchant_name)
                        item_category = self.get_item_category(merchant_name)
                        
                        payment_items_to_insert.append((
                            payment_id_counter,  # payment_id
                            item_name,
                            item_category,
                            1,  # quantity
                            item_price,
                            item_price
                        ))
                
                payment_id_counter += 1
        
        # Batch insert payments
        cursor.executemany("""
            INSERT INTO payments (payment_id, user_id, merchant_id, amount, currency, 
                                payment_date, payment_method, description, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, payments_to_insert)
        
        # Batch insert payment items
        if payment_items_to_insert:
            cursor.executemany("""
                INSERT INTO payment_items (payment_id, item_name, item_category, 
                                         quantity, unit_price, total_price)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, payment_items_to_insert)
        
        self.connection.commit()
        print(f"✅ Generated {len(payments_to_insert)} payments with {len(payment_items_to_insert)} items")
    
    def generate_realistic_amount(self, merchant_name):
        """Generate realistic payment amounts based on merchant type"""
        amounts_by_merchant = {
            'Netflix': (9.99, 19.99),
            'Spotify': (9.99, 14.99),
            'Disney+': (7.99, 13.99),
            'Steam': (5.99, 59.99),
            'Starbucks': (3.50, 15.00),
            'Dunkin\'': (2.50, 12.00),
            'McDonald\'s': (4.99, 25.00),
            'Chipotle': (8.50, 20.00),
            'Panera Bread': (6.99, 30.00),
            'AMC Theatres': (12.50, 45.00),
            'Uber': (8.00, 50.00),
            'Shell': (25.00, 80.00),
            'Amazon': (15.00, 200.00),
            'Target': (20.00, 150.00),
            'Whole Foods': (30.00, 120.00),
            'Costco': (50.00, 300.00),
            'Apple Store': (50.00, 1200.00),
            'Best Buy': (25.00, 800.00),
            'Walgreens': (5.99, 50.00),
            'CVS Pharmacy': (8.99, 75.00)
        }
        
        min_amount, max_amount = amounts_by_merchant.get(merchant_name, (10.00, 100.00))
        return round(random.uniform(min_amount, max_amount), 2)
    
    def generate_payment_description(self, merchant_name, amount):
        """Generate realistic payment descriptions"""
        descriptions = {
            'Netflix': f'Netflix Monthly Subscription',
            'Spotify': f'Spotify Premium Subscription',
            'Disney+': f'Disney+ Monthly Subscription',
            'Steam': f'Steam Game Purchase',
            'Starbucks': f'Coffee and pastry',
            'Dunkin\'': f'Coffee and donut',
            'McDonald\'s': f'Fast food order',
            'Chipotle': f'Burrito bowl meal',
            'Panera Bread': f'Sandwich and soup',
            'AMC Theatres': f'Movie tickets',
            'Uber': f'Rideshare trip',
            'Shell': f'Gasoline fill-up',
            'Amazon': f'Online shopping order',
            'Target': f'Retail purchase',
            'Whole Foods': f'Grocery shopping',
            'Costco': f'Bulk shopping',
            'Apple Store': f'Apple product purchase',
            'Best Buy': f'Electronics purchase',
            'Walgreens': f'Pharmacy and essentials',
            'CVS Pharmacy': f'Medication and health items'
        }
        
        return descriptions.get(merchant_name, f'Purchase at {merchant_name}')
    
    def generate_item_name(self, merchant_name):
        """Generate realistic item names based on merchant"""
        items_by_merchant = {
            'Starbucks': ['Latte', 'Cappuccino', 'Frappuccino', 'Croissant', 'Muffin'],
            'McDonald\'s': ['Big Mac', 'Quarter Pounder', 'Chicken McNuggets', 'Fries', 'McFlurry'],
            'Chipotle': ['Burrito', 'Bowl', 'Tacos', 'Chips', 'Guacamole'],
            'Amazon': ['Book', 'Electronics', 'Home & Garden', 'Clothing', 'Toys'],
            'Target': ['Clothing', 'Home Decor', 'Electronics', 'Groceries', 'Toys'],
            'Whole Foods': ['Organic Vegetables', 'Fresh Fruit', 'Meat & Seafood', 'Dairy', 'Bakery Items'],
        }
        
        items = items_by_merchant.get(merchant_name, ['Item', 'Product', 'Service'])
        return random.choice(items)
    
    def get_item_category(self, merchant_name):
        """Get item category based on merchant"""
        categories = {
            'Starbucks': 'Coffee & Beverages',
            'McDonald\'s': 'Fast Food',
            'Chipotle': 'Fast Casual',
            'Amazon': 'Online Shopping',
            'Target': 'Retail',
            'Whole Foods': 'Groceries',
        }
        return categories.get(merchant_name, 'General')
    
    def create_indexes_and_views(self):
        """Create additional indexes and useful views"""
        cursor = self.connection.cursor()
        
        views_sql = """
        -- View for payment summaries with merchant and category info
        CREATE OR REPLACE VIEW payment_summary AS
        SELECT 
            p.payment_id,
            p.user_id,
            u.username,
            u.first_name,
            u.last_name,
            p.amount,
            p.currency,
            p.payment_date,
            p.payment_method,
            p.description,
            m.merchant_name,
            m.merchant_type,
            c.category_name,
            c.category_type,
            EXTRACT(YEAR FROM p.payment_date) as payment_year,
            EXTRACT(MONTH FROM p.payment_date) as payment_month,
            EXTRACT(DOW FROM p.payment_date) as day_of_week,
            CASE 
                WHEN EXTRACT(DOW FROM p.payment_date) IN (0, 6) THEN 'Weekend'
                ELSE 'Weekday'
            END as weekend_indicator
        FROM payments p
        JOIN users u ON p.user_id = u.user_id
        JOIN merchants m ON p.merchant_id = m.merchant_id
        JOIN categories c ON m.category_id = c.category_id;
        
        -- View for monthly spending summaries
        CREATE OR REPLACE VIEW monthly_spending AS
        SELECT 
            user_id,
            username,
            DATE_TRUNC('month', payment_date) as month,
            COUNT(*) as transaction_count,
            SUM(amount) as total_amount,
            AVG(amount) as avg_amount,
            MIN(amount) as min_amount,
            MAX(amount) as max_amount
        FROM payment_summary
        GROUP BY user_id, username, DATE_TRUNC('month', payment_date)
        ORDER BY month DESC;
        
        -- View for category spending
        CREATE OR REPLACE VIEW category_spending AS
        SELECT 
            user_id,
            username,
            category_name,
            category_type,
            COUNT(*) as transaction_count,
            SUM(amount) as total_amount,
            AVG(amount) as avg_amount
        FROM payment_summary
        GROUP BY user_id, username, category_name, category_type
        ORDER BY total_amount DESC;
        """
        
        try:
            cursor.execute(views_sql)
            self.connection.commit()
            print("✅ Views and indexes created successfully")
        except psycopg2.Error as e:
            print(f"❌ Error creating views: {e}")
            self.connection.rollback()
            raise
        finally:
            cursor.close()
    
    def verify_setup(self):
        """Verify the database setup"""
        cursor = self.connection.cursor()
        
        try:
            # Check table counts
            tables_to_check = ['users', 'categories', 'merchants', 'payments', 'payment_items']
            
            print("\n📊 Database Setup Summary:")
            print("=" * 40)
            
            for table in tables_to_check:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"{table.title()}: {count:,} records")
            
            # Check date range of payments
            cursor.execute("""
                SELECT 
                    MIN(payment_date) as earliest_payment,
                    MAX(payment_date) as latest_payment,
                    COUNT(DISTINCT user_id) as unique_users
                FROM payments
            """)
            
            result = cursor.fetchone()
            print(f"\nPayment Date Range:")
            print(f"Earliest: {result[0].strftime('%Y-%m-%d')}")
            print(f"Latest: {result[1].strftime('%Y-%m-%d')}")
            print(f"Unique Users: {result[2]}")
            
            # Sample some data
            cursor.execute("""
                SELECT 
                    username,
                    merchant_name,
                    amount,
                    payment_date,
                    description
                FROM payment_summary 
                ORDER BY payment_date DESC 
                LIMIT 5
            """)
            
            print(f"\nSample Recent Transactions:")
            print("-" * 80)
            for row in cursor.fetchall():
                print(f"{row[0]} | {row[1]} | ${row[2]} | {row[3].strftime('%Y-%m-%d')} | {row[4]}")
            
            # Show user IDs to confirm they're strings
            cursor.execute("SELECT user_id, username FROM users ORDER BY user_id")
            print(f"\nUser IDs (now string type):")
            print("-" * 30)
            for row in cursor.fetchall():
                print(f"{row[0]} | {row[1]}")
            
            print("\n✅ Database setup completed successfully!")
            print(f"🔗 Connection string: postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
            
        except psycopg2.Error as e:
            print(f"❌ Error verifying setup: {e}")
            raise
        finally:
            cursor.close()
    
    def close_connection(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            print("🔌 Database connection closed")

def main():
    """Main setup function"""
    print("🚀 Setting up NL to SQL Agent Database")
    print("=" * 50)
    
    setup = DatabaseSetup()
    
    try:
        # Step 1: Create database and user
        print("\n1. Creating database and user...")
        setup.create_database_and_user()
        
        # Step 2: Connect to application database
        print("\n2. Connecting to application database...")
        setup.connect_to_app_database()
        
        # Step 3: Create tables
        print("\n3. Creating database tables...")
        setup.create_tables()
        
        # Step 4: Generate sample data
        print("\n4. Generating sample data...")
        setup.generate_sample_data()
        
        # Step 5: Create views and indexes
        print("\n5. Creating views and indexes...")
        setup.create_indexes_and_views()
        
        # Step 6: Verify setup
        print("\n6. Verifying database setup...")
        setup.verify_setup()
        
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        return False
    
    finally:
        setup.close_connection()
    
    print("\n🎉 Database setup completed successfully!")
    print("\nNext steps:")
    print("1. Set your OPENAI_API_KEY environment variable")
    print("2. Install required Python packages")
    print("3. Run the main application: python nl_to_sql_agent.py")
    
    return True

if __name__ == "__main__":
    main()