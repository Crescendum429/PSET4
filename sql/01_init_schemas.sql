CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS analytics;

COMMENT ON SCHEMA raw IS 'Raw ingestion layer - Bronze';
COMMENT ON SCHEMA analytics IS 'Analytics layer - Gold - OBT';
