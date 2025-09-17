-- Initialize Seven Steps to Poem database
-- This script sets up the initial database schema and users

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create application user (if needed for production)
-- For development, we use the default postgres user

-- Set up database settings for better performance
ALTER DATABASE sevensteps SET timezone TO 'UTC';
ALTER DATABASE sevensteps SET log_statement TO 'none';
ALTER DATABASE sevensteps SET log_min_duration_statement TO 1000;