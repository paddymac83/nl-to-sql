-- SQL script to set up the test database with sample data

-- Create database (run as superuser)
-- CREATE DATABASE nlsql_db;
-- CREATE USER nlsql_user WITH ENCRYPTED PASSWORD 'nlsql_password';
-- GRANT ALL PRIVILEGES ON DATABASE nlsql_db TO nlsql_user;

-- Connect to nlsql_db and run the following:

-- Users table
CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(50) PRIMARY KEY,
    username VARCHAR(100) NOT NULL,
    email VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Categories table
CREATE TABLE IF NOT EXISTS categories (
    category_id SERIAL PRIMARY KEY,
    category_name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Merchants table
CREATE TABLE IF NOT EXISTS merchants (
    merchant_id SERIAL PRIMARY KEY,
    merchant_name VARCHAR(200) NOT NULL,
    merchant_type VARCHAR(100),
    location VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Payments table
CREATE TABLE IF NOT EXISTS payments (
    payment_id SERIAL PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    amount DECIMAL(10,2) NOT NULL,
    category_id INTEGER,
    merchant_id INTEGER,
    payment_date DATE NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (category_id) REFERENCES categories(category_id),
    FOREIGN KEY (merchant_id) REFERENCES merchants(merchant_id)
);

-- Budgets table
CREATE TABLE IF NOT EXISTS budgets (
    budget_id SERIAL PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    category_id INTEGER,
    budget_amount DECIMAL(10,2) NOT NULL,
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
);

-- Insert sample data

-- Sample users
INSERT INTO users (user_id, username, email) VALUES 
('user_001', 'john_doe', 'john@example.com'),
('user_002', 'jane_smith', 'jane@example.com'),
('user_003', 'bob_wilson', 'bob@example.com'),
('user_004', 'alice_brown', 'alice@example.com'),
('user_005', 'charlie_davis', 'charlie@example.com')
ON CONFLICT (user_id) DO NOTHING;

-- Sample categories
INSERT INTO categories (category_name, description) VALUES 
('Food & Dining', 'Restaurants, groceries, and food delivery'),
('Transportation', 'Gas, public transit, rideshare, and parking'),
('Entertainment', 'Movies, games, streaming services, and events'),
('Shopping', 'Clothing, electronics, and general retail'),
('Utilities', 'Electric, gas, water, internet, and phone bills'),
('Healthcare', 'Medical expenses, prescriptions, and insurance'),
('Travel', 'Hotels, flights, and vacation expenses'),
('Education', 'Books, courses, and educational materials'),
('Home & Garden', 'Home improvement, furniture, and gardening'),
('Personal Care', 'Haircuts, cosmetics, and personal services')
ON CONFLICT (category_name) DO NOTHING;

-- Sample merchants
INSERT INTO merchants (merchant_name, merchant_type, location) VALUES 
('Whole Foods Market', 'Grocery', 'New York, NY'),
('Shell Gas Station', 'Gas Station', 'Los Angeles, CA'),
('Netflix', 'Streaming Service', 'Online'),
('Amazon', 'E-commerce', 'Online'),
('Starbucks', 'Coffee Shop', 'Seattle, WA'),
('Target', 'Retail', 'Minneapolis, MN'),
('Uber', 'Transportation', 'Online'),
('Apple Store', 'Electronics', 'Cupertino, CA'),
('CVS Pharmacy', 'Pharmacy', 'Woonsocket, RI'),
('Home Depot', 'Home Improvement', 'Atlanta, GA'),
('Spotify', 'Streaming Service', 'Online'),
('McDonald''s', 'Fast Food', 'Chicago, IL'),
('Best Buy', 'Electronics', 'Richfield, MN'),
('Costco', 'Wholesale', 'Issaquah, WA'),
('Airbnb', 'Travel', 'Online')
ON CONFLICT DO NOTHING;

-- Sample payments (generate for the last 6 months)
INSERT INTO payments (user_id, amount, category_id, merchant_id, payment_date, description) VALUES 
-- User 001 payments
('user_001', 45.67, 1, 1, CURRENT_DATE - INTERVAL '1 day', 'Weekly grocery shopping'),
('user_001', 25.00, 2, 2, CURRENT_DATE - INTERVAL '2 days', 'Gas fill-up'),
('user_001', 15.99, 3, 3, CURRENT_DATE - INTERVAL '5 days', 'Netflix monthly subscription'),
('user_001', 89.34, 4, 4, CURRENT_DATE - INTERVAL '7 days', 'Amazon purchase'),
('user_001', 5.50, 1, 5, CURRENT_DATE - INTERVAL '10 days', 'Morning coffee'),
('user_001', 120.00, 4, 6, CURRENT_DATE - INTERVAL '15 days', 'Target shopping'),
('user_001', 18.75, 2, 7, CURRENT_DATE - INTERVAL '20 days', 'Uber ride'),
('user_001', 299.99, 4, 8, CURRENT_DATE - INTERVAL '25 days', 'iPhone case'),
('user_001', 12.99, 6, 9, CURRENT_DATE - INTERVAL '30 days', 'Pharmacy'),

-- User 002 payments
('user_002', 78.90, 1, 1, CURRENT_DATE - INTERVAL '1 day', 'Organic groceries'),
('user_002', 32.50, 2, 2, CURRENT_DATE - INTERVAL '3 days', 'Premium gas'),
('user_002', 9.99, 3, 11, CURRENT_DATE - INTERVAL '6 days', 'Spotify Premium'),
('user_002', 156.78, 4, 4, CURRENT_DATE - INTERVAL '8 days', 'Amazon electronics'),
('user_002', 8.25, 1, 12, CURRENT_DATE - INTERVAL '12 days', 'Fast food lunch'),
('user_002', 245.00, 9, 10, CURRENT_DATE - INTERVAL '18 days', 'Home improvement'),
('user_002', 22.40, 2, 7, CURRENT_DATE - INTERVAL '22 days', 'Airport ride'),
('user_002', 67.89, 4, 13, CURRENT_DATE - INTERVAL '28 days', 'Best Buy purchase'),

-- User 003 payments
('user_003', 95.50, 1, 14, CURRENT_DATE - INTERVAL '2 days', 'Costco bulk shopping'),
('user_003', 40.00, 2, 2, CURRENT_DATE - INTERVAL '4 days', 'Gas station'),
('user_003', 15.99, 3, 3, CURRENT_DATE - INTERVAL '7 days', 'Netflix'),
('user_003', 200.00, 7, 15, CURRENT_DATE - INTERVAL '14 days', 'Airbnb booking'),
('user_003', 45.67, 1, 5, CURRENT_DATE - INTERVAL '21 days', 'Coffee and pastries'),
('user_003', 89.99, 4, 6, CURRENT_DATE - INTERVAL '26 days', 'Target essentials'),

-- More sample data for other users...
('user_004', 67.34, 1, 1, CURRENT_DATE - INTERVAL '3 days', 'Fresh produce'),
('user_004', 28.75, 2, 2, CURRENT_DATE - INTERVAL '5 days', 'Fuel'),
('user_004', 12.99, 3, 11, CURRENT_DATE - INTERVAL '9 days', 'Music streaming'),
('user_004', 134.56, 4, 4, CURRENT_DATE - INTERVAL '11 days', 'Online shopping'),

('user_005', 54.32, 1, 12, CURRENT_DATE - INTERVAL '1 day', 'Quick meal'),
('user_005', 35.60, 2, 7, CURRENT_DATE - INTERVAL '6 days', 'Rideshare'),
('user_005', 15.99, 3, 3, CURRENT_DATE - INTERVAL '13 days', 'Streaming service'),
('user_005', 78.90, 4, 8, CURRENT_DATE - INTERVAL '19 days', 'Tech accessory')
ON CONFLICT DO NOTHING;

-- Sample budgets
INSERT INTO budgets (user_id, category_id, budget_amount, period_start, period_end) VALUES 
('user_001', 1, 300.00, DATE_TRUNC('month', CURRENT_DATE), DATE_TRUNC('month', CURRENT_DATE) + INTERVAL '1 month' - INTERVAL '1 day'),
('user_001', 2, 150.00, DATE_TRUNC('month', CURRENT_DATE), DATE_TRUNC('month', CURRENT_DATE) + INTERVAL '1 month' - INTERVAL '1 day'),
('user_001', 3, 50.00, DATE_TRUNC('month', CURRENT_DATE), DATE_TRUNC('month', CURRENT_DATE) + INTERVAL '1 month' - INTERVAL '1 day'),
('user_001', 4, 200.00, DATE_TRUNC('month', CURRENT_DATE), DATE_TRUNC('month', CURRENT_DATE) + INTERVAL '1 month' - INTERVAL '1 day'),

('user_002', 1, 400.00, DATE_TRUNC('month', CURRENT_DATE), DATE_TRUNC('month', CURRENT_DATE) + INTERVAL '1 month' - INTERVAL '1 day'),
('user_002', 2, 100.00, DATE_TRUNC('month', CURRENT_DATE), DATE_TRUNC('month', CURRENT_DATE) + INTERVAL '1 month' - INTERVAL '1 day'),
('user_002', 3, 75.00, DATE_TRUNC('month', CURRENT_DATE), DATE_TRUNC('month', CURRENT_DATE) + INTERVAL '1 month' - INTERVAL '1 day'),
('user_002', 4, 250.00, DATE_TRUNC('month', CURRENT_DATE), DATE_TRUNC('month', CURRENT_DATE) + INTERVAL '1 month' - INTERVAL '1 day')
ON CONFLICT DO NOTHING;

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_payments_user_id ON payments(user_id);
CREATE INDEX IF NOT EXISTS idx_payments_date ON payments(payment_date);
CREATE INDEX IF NOT EXISTS idx_payments_category ON payments(category_id);
CREATE INDEX IF NOT EXISTS idx_payments_merchant ON payments(merchant_id);
CREATE INDEX IF NOT EXISTS idx_budgets_user_id ON budgets(user_id);
CREATE INDEX IF NOT EXISTS idx_budgets_category ON budgets(category_id);

-- Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO nlsql_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO nlsql_user;