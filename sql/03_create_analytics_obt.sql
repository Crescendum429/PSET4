CREATE TABLE IF NOT EXISTS analytics.obt_trips (
    pickup_datetime TIMESTAMP NOT NULL,
    dropoff_datetime TIMESTAMP NOT NULL,
    pickup_hour INTEGER,
    pickup_dow INTEGER,
    month INTEGER,
    year INTEGER,

    pu_location_id INTEGER,
    pu_zone VARCHAR(100),
    pu_borough VARCHAR(50),
    do_location_id INTEGER,
    do_zone VARCHAR(100),
    do_borough VARCHAR(50),

    service_type VARCHAR(10) NOT NULL,
    vendor_id INTEGER,
    vendor_name VARCHAR(50),
    rate_code_id INTEGER,
    rate_code_desc VARCHAR(50),
    payment_type INTEGER,
    payment_type_desc VARCHAR(50),
    trip_type INTEGER,

    passenger_count INTEGER,
    trip_distance NUMERIC(10,2),
    fare_amount NUMERIC(10,2),
    extra NUMERIC(10,2),
    mta_tax NUMERIC(10,2),
    tip_amount NUMERIC(10,2),
    tolls_amount NUMERIC(10,2),
    improvement_surcharge NUMERIC(10,2),
    congestion_surcharge NUMERIC(10,2),
    airport_fee NUMERIC(10,2),
    total_amount NUMERIC(10,2),
    store_and_fwd_flag VARCHAR(1),

    trip_duration_min NUMERIC(10,2),
    avg_speed_mph NUMERIC(10,2),
    tip_pct NUMERIC(10,2),

    run_id VARCHAR(100),
    source_year INTEGER,
    source_month INTEGER,
    ingested_at_utc TIMESTAMP,
    obt_created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_obt_pickup_dt ON analytics.obt_trips(pickup_datetime);
CREATE INDEX IF NOT EXISTS idx_obt_service ON analytics.obt_trips(service_type);
CREATE INDEX IF NOT EXISTS idx_obt_year_month ON analytics.obt_trips(year, month);
CREATE INDEX IF NOT EXISTS idx_obt_pu_borough ON analytics.obt_trips(pu_borough);
CREATE INDEX IF NOT EXISTS idx_obt_do_borough ON analytics.obt_trips(do_borough);
CREATE INDEX IF NOT EXISTS idx_obt_source ON analytics.obt_trips(source_year, source_month, service_type);

COMMENT ON TABLE analytics.obt_trips IS 'One Big Table - Unified trip data 2015-2025';
