-- Initialize PostgreSQL database for Single-Cell Graph Hub
-- This script runs on first container startup

-- Create additional databases if needed
CREATE DATABASE scgraph_hub_test;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS datasets;
CREATE SCHEMA IF NOT EXISTS models; 
CREATE SCHEMA IF NOT EXISTS experiments;
CREATE SCHEMA IF NOT EXISTS users;

-- Set up basic tables structure
CREATE TABLE IF NOT EXISTS datasets.catalog (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    organism VARCHAR(100),
    tissue VARCHAR(100),
    n_cells INTEGER,
    n_genes INTEGER,
    modality VARCHAR(100),
    has_spatial BOOLEAN DEFAULT FALSE,
    data_format VARCHAR(50),
    file_path TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS models.registry (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    model_type VARCHAR(100),
    architecture JSONB,
    parameters JSONB,
    performance_metrics JSONB,
    file_path TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS experiments.runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_id UUID REFERENCES datasets.catalog(id),
    model_id UUID REFERENCES models.registry(id),
    config JSONB,
    results JSONB,
    status VARCHAR(50) DEFAULT 'pending',
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_by VARCHAR(255)
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_datasets_organism ON datasets.catalog(organism);
CREATE INDEX IF NOT EXISTS idx_datasets_modality ON datasets.catalog(modality);
CREATE INDEX IF NOT EXISTS idx_datasets_metadata ON datasets.catalog USING GIN(metadata);
CREATE INDEX IF NOT EXISTS idx_models_type ON models.registry(model_type);
CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments.runs(status);
CREATE INDEX IF NOT EXISTS idx_experiments_dataset ON experiments.runs(dataset_id);

-- Create update timestamp trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply triggers to tables
CREATE TRIGGER update_datasets_updated_at BEFORE UPDATE ON datasets.catalog FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_models_updated_at BEFORE UPDATE ON models.registry FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE scgraph_hub TO scgraph;
GRANT ALL PRIVILEGES ON DATABASE scgraph_hub_test TO scgraph;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA datasets TO scgraph;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA models TO scgraph;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA experiments TO scgraph;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA datasets TO scgraph;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA models TO scgraph;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA experiments TO scgraph;