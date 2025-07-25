CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(50) PRIMARY KEY,
    username VARCHAR(100) NOT NULL,
    email VARCHAR(255),
    customer_id INTEGER UNIQUE,
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

-- Accounts table (new for payload support)
CREATE TABLE IF NOT EXISTS accounts (
    account_id SERIAL PRIMARY KEY,
    account_number BIGINT NOT NULL UNIQUE,
    sort_code INTEGER NOT NULL,
    user_id VARCHAR(50) NOT NULL,
    customer_id INTEGER,
    account_type VARCHAR(50) DEFAULT 'current',
    balance DECIMAL(15,2) DEFAULT 0.00,
    currency VARCHAR(3) DEFAULT 'GBP',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (customer_id) REFERENCES users(customer_id)
);

-- Enhanced Payments table with payload support columns
CREATE TABLE IF NOT EXISTS payments (
    payment_id SERIAL PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    customer_id INTEGER,
    account_number BIGINT,
    sort_code INTEGER,
    amount DECIMAL(10,2) NOT NULL,
    category_id INTEGER,
    merchant_id INTEGER,
    payment_date DATE NOT NULL,
    description TEXT,
    reference VARCHAR(100),
    transaction_type VARCHAR(20) DEFAULT 'debit',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (customer_id) REFERENCES users(customer_id),
    FOREIGN KEY (account_number) REFERENCES accounts(account_number),
    FOREIGN KEY (category_id) REFERENCES categories(category_id),
    FOREIGN KEY (merchant_id) REFERENCES merchants(merchant_id)
);

-- Budgets table
CREATE TABLE IF NOT EXISTS budgets (
    budget_id SERIAL PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    customer_id INTEGER,
    category_id INTEGER,
    budget_amount DECIMAL(10,2) NOT NULL,
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (customer_id) REFERENCES users(customer_id),
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
);

-- Insert enhanced sample data

-- Sample users with customer IDs
INSERT INTO users (user_id, username, email, customer_id) VALUES 
('user_001', 'john_doe', 'john@example.com', 22),
('user_002', 'jane_smith', 'jane@example.com', 23),
('user_003', 'bob_wilson', 'bob@example.com', 24),
('user_004', 'alice_brown', 'alice@example.com', 25),
('user_005', 'charlie_davis', 'charlie@example.com', 26)
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

-- Sample accounts with payload-compatible data
INSERT INTO accounts (account_number, sort_code, user_id, customer_id, account_type, balance) VALUES 
(900914, 123456, 'user_001', 22, 'current', 2500.00),
(900915, 123456, 'user_002', 23, 'current', 1800.50),
(900916, 123457, 'user_003', 24, 'savings', 5000.00),
(900917, 123457, 'user_004', 25, 'current', 750.25),
(900918, 123458, 'user_005', 26, 'current', 3200.75)
ON CONFLICT (account_number) DO NOTHING;

-- Enhanced sample payments with payload support columns
INSERT INTO payments (user_id, customer_id, account_number, sort_code, amount, category_id, merchant_id, payment_date, description, reference) VALUES 
-- User 001 (CIN: 22, Account: 900914, Sort Code: 123456) payments
('user_001', 22, 900914, 123456, 45.67, 1, 1, CURRENT_DATE - INTERVAL '1 day', 'Weekly grocery shopping', 'REF001'),
('user_001', 22, 900914, 123456, 25.00, 2, 2, CURRENT_DATE - INTERVAL '2 days', 'Gas fill-up', 'REF002'),
('user_001', 22, 900914, 123456, 15.99, 3, 3, CURRENT_DATE - INTERVAL '5 days', 'Netflix monthly subscription', 'REF003'),
('user_001', 22, 900914, 123456, 89.34, 4, 4, CURRENT_DATE - INTERVAL '7 days', 'Amazon purchase', 'REF004'),
('user_001', 22, 900914, 123456, 5.50, 1, 5, CURRENT_DATE - INTERVAL '10 days', 'Morning coffee', 'REF005'),

-- User 002 (CIN: 23, Account: 900915, Sort Code: 123456) payments
('user_002', 23, 900915, 123456, 78.90, 1, 1, CURRENT_DATE - INTERVAL '1 day', 'Organic groceries', 'REF006'),
('user_002', 23, 900915, 123456, 32.50, 2, 2, CURRENT_DATE - INTERVAL '3 days', 'Premium gas', 'REF007'),
('user_002', 23, 900915, 123456, 9.99, 3, 11, CURRENT_DATE - INTERVAL '6 days', 'Spotify Premium', 'REF008'),
('user_002', 23, 900915, 123456, 156.78, 4, 4, CURRENT_DATE - INTERVAL '8 days', 'Amazon electronics', 'REF009'),

-- User 003 (CIN: 24, Account: 900916, Sort Code: 123457) payments
('user_003', 24, 900916, 123457, 95.50, 1, 14, CURRENT_DATE - INTERVAL '2 days', 'Costco bulk shopping', 'REF010'),
('user_003', 24, 900916, 123457, 40.00, 2, 2, CURRENT_DATE - INTERVAL '4 days', 'Gas station', 'REF011'),
('user_003', 24, 900916, 123457, 15.99, 3, 3, CURRENT_DATE - INTERVAL '7 days', 'Netflix', 'REF012'),

-- User 004 (CIN: 25, Account: 900917, Sort Code: 123457) payments
('user_004', 25, 900917, 123457, 67.34, 1, 1, CURRENT_DATE - INTERVAL '3 days', 'Fresh produce', 'REF013'),
('user_004', 25, 900917, 123457, 28.75, 2, 2, CURRENT_DATE - INTERVAL '5 days', 'Fuel', 'REF014'),
('user_004', 25, 900917, 123457, 12.99, 3, 11, CURRENT_DATE - INTERVAL '9 days', 'Music streaming', 'REF015'),

-- User 005 (CIN: 26, Account: 900918, Sort Code: 123458) payments
('user_005', 26, 900918, 123458, 54.32, 1, 12, CURRENT_DATE - INTERVAL '1 day', 'Quick meal', 'REF016'),
('user_005', 26, 900918, 123458, 35.60, 2, 7, CURRENT_DATE - INTERVAL '6 days', 'Rideshare', 'REF017'),
('user_005', 26, 900918, 123458, 15.99, 3, 3, CURRENT_DATE - INTERVAL '13 days', 'Streaming service', 'REF018')
ON CONFLICT DO NOTHING;

-- Sample budgets with customer ID support
INSERT INTO budgets (user_id, customer_id, category_id, budget_amount, period_start, period_end) VALUES 
('user_001', 22, 1, 300.00, DATE_TRUNC('month', CURRENT_DATE), DATE_TRUNC('month', CURRENT_DATE) + INTERVAL '1 month' - INTERVAL '1 day'),
('user_001', 22, 2, 150.00, DATE_TRUNC('month', CURRENT_DATE), DATE_TRUNC('month', CURRENT_DATE) + INTERVAL '1 month' - INTERVAL '1 day'),
('user_001', 22, 3, 50.00, DATE_TRUNC('month', CURRENT_DATE), DATE_TRUNC('month', CURRENT_DATE) + INTERVAL '1 month' - INTERVAL '1 day'),
('user_001', 22, 4, 200.00, DATE_TRUNC('month', CURRENT_DATE), DATE_TRUNC('month', CURRENT_DATE) + INTERVAL '1 month' - INTERVAL '1 day'),

('user_002', 23, 1, 400.00, DATE_TRUNC('month', CURRENT_DATE), DATE_TRUNC('month', CURRENT_DATE) + INTERVAL '1 month' - INTERVAL '1 day'),
('user_002', 23, 2, 100.00, DATE_TRUNC('month', CURRENT_DATE), DATE_TRUNC('month', CURRENT_DATE) + INTERVAL '1 month' - INTERVAL '1 day'),
('user_002', 23, 3, 75.00, DATE_TRUNC('month', CURRENT_DATE), DATE_TRUNC('month', CURRENT_DATE) + INTERVAL '1 month' - INTERVAL '1 day')
ON CONFLICT DO NOTHING;

-- Create enhanced indexes for better performance
CREATE INDEX IF NOT EXISTS idx_payments_user_id ON payments(user_id);
CREATE INDEX IF NOT EXISTS idx_payments_customer_id ON payments(customer_id);
CREATE INDEX IF NOT EXISTS idx_payments_account_number ON payments(account_number);
CREATE INDEX IF NOT EXISTS idx_payments_sort_code ON payments(sort_code);
CREATE INDEX IF NOT EXISTS idx_payments_date ON payments(payment_date);
CREATE INDEX IF NOT EXISTS idx_payments_category ON payments(category_id);
CREATE INDEX IF NOT EXISTS idx_payments_merchant ON payments(merchant_id);
CREATE INDEX IF NOT EXISTS idx_accounts_customer_id ON accounts(customer_id);
CREATE INDEX IF NOT EXISTS idx_accounts_sort_code ON accounts(sort_code);
CREATE INDEX IF NOT EXISTS idx_budgets_user_id ON budgets(user_id);
CREATE INDEX IF NOT EXISTS idx_budgets_customer_id ON budgets(customer_id);
CREATE INDEX IF NOT EXISTS idx_budgets_category ON budgets(category_id);

-- Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO nlsql_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO nlsql_user;