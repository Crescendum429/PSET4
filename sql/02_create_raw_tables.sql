CREATE TABLE IF NOT EXISTS raw.yellow_taxi_trip (
    vendorid INTEGER,
    tpep_pickup_datetime TIMESTAMP,
    tpep_dropoff_datetime TIMESTAMP,
    passenger_count INTEGER,
    trip_distance NUMERIC(10,2),
    ratecodeid INTEGER,
    store_and_fwd_flag VARCHAR(1),
    pulocationid INTEGER,
    dolocationid INTEGER,
    payment_type INTEGER,
    fare_amount NUMERIC(10,2),
    extra NUMERIC(10,2),
    mta_tax NUMERIC(10,2),
    tip_amount NUMERIC(10,2),
    tolls_amount NUMERIC(10,2),
    improvement_surcharge NUMERIC(10,2),
    total_amount NUMERIC(10,2),
    congestion_surcharge NUMERIC(10,2),
    airport_fee NUMERIC(10,2),
    run_id VARCHAR(100),
    service_type VARCHAR(10) DEFAULT 'yellow',
    source_year INTEGER,
    source_month INTEGER,
    ingested_at_utc TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source_path TEXT
);

CREATE INDEX IF NOT EXISTS idx_yellow_pickup_dt ON raw.yellow_taxi_trip(tpep_pickup_datetime);
CREATE INDEX IF NOT EXISTS idx_yellow_source ON raw.yellow_taxi_trip(source_year, source_month);
CREATE INDEX IF NOT EXISTS idx_yellow_run_id ON raw.yellow_taxi_trip(run_id);

CREATE TABLE IF NOT EXISTS raw.green_taxi_trip (
    vendorid INTEGER,
    lpep_pickup_datetime TIMESTAMP,
    lpep_dropoff_datetime TIMESTAMP,
    passenger_count INTEGER,
    trip_distance NUMERIC(10,2),
    ratecodeid INTEGER,
    store_and_fwd_flag VARCHAR(1),
    pulocationid INTEGER,
    dolocationid INTEGER,
    payment_type INTEGER,
    fare_amount NUMERIC(10,2),
    extra NUMERIC(10,2),
    mta_tax NUMERIC(10,2),
    tip_amount NUMERIC(10,2),
    tolls_amount NUMERIC(10,2),
    improvement_surcharge NUMERIC(10,2),
    total_amount NUMERIC(10,2),
    congestion_surcharge NUMERIC(10,2),
    airport_fee NUMERIC(10,2),
    trip_type INTEGER,
    run_id VARCHAR(100),
    service_type VARCHAR(10) DEFAULT 'green',
    source_year INTEGER,
    source_month INTEGER,
    ingested_at_utc TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source_path TEXT
);

CREATE INDEX IF NOT EXISTS idx_green_pickup_dt ON raw.green_taxi_trip(lpep_pickup_datetime);
CREATE INDEX IF NOT EXISTS idx_green_source ON raw.green_taxi_trip(source_year, source_month);
CREATE INDEX IF NOT EXISTS idx_green_run_id ON raw.green_taxi_trip(run_id);

CREATE TABLE IF NOT EXISTS raw.taxi_zone_lookup (
    locationid INTEGER PRIMARY KEY,
    borough VARCHAR(50),
    zone VARCHAR(100),
    service_zone VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS raw.ingestion_audit (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(100) NOT NULL,
    service_type VARCHAR(10) NOT NULL,
    source_year INTEGER NOT NULL,
    source_month INTEGER NOT NULL,
    source_path TEXT,
    record_count INTEGER,
    status VARCHAR(20),
    error_message TEXT,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    duration_seconds NUMERIC(10,2)
);

CREATE INDEX IF NOT EXISTS idx_audit_run_id ON raw.ingestion_audit(run_id);
CREATE INDEX IF NOT EXISTS idx_audit_service_year_month ON raw.ingestion_audit(service_type, source_year, source_month);
