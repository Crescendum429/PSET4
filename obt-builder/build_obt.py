#!/usr/bin/env python3
import os
import sys
import argparse
import psycopg2
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

PG_HOST = os.getenv('PG_HOST', 'postgres')
PG_PORT = os.getenv('PG_PORT', '5432')
PG_DB = os.getenv('PG_DB', 'nyc_taxi')
PG_USER = os.getenv('PG_USER')
PG_PASSWORD = os.getenv('PG_PASSWORD')
PG_SCHEMA_RAW = os.getenv('PG_SCHEMA_RAW', 'raw')
PG_SCHEMA_ANALYTICS = os.getenv('PG_SCHEMA_ANALYTICS', 'analytics')

def get_connection():
    return psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        database=PG_DB,
        user=PG_USER,
        password=PG_PASSWORD
    )

def log(msg):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {msg}")
    sys.stdout.flush()

def get_vendor_name(vendor_id):
    mapping = {1: 'Creative Mobile', 2: 'VeriFone Inc'}
    return mapping.get(vendor_id, 'Unknown')

def get_rate_code_desc(rate_code_id):
    mapping = {
        1: 'Standard rate',
        2: 'JFK',
        3: 'Newark',
        4: 'Nassau or Westchester',
        5: 'Negotiated fare',
        6: 'Group ride'
    }
    return mapping.get(rate_code_id, 'Unknown')

def get_payment_type_desc(payment_type):
    mapping = {
        1: 'Credit card',
        2: 'Cash',
        3: 'No charge',
        4: 'Dispute',
        5: 'Unknown',
        6: 'Voided trip'
    }
    return mapping.get(payment_type, 'Unknown')

def build_obt_full(conn, year_start, year_end, services, run_id, overwrite):
    log("=== MODE: FULL REBUILD ===")
    log(f"Years: {year_start}-{year_end}")
    log(f"Services: {services}")
    log(f"Run ID: {run_id}")
    log(f"Overwrite: {overwrite}")
    log("")

    cursor = conn.cursor()

    if overwrite:
        log("Truncating analytics.obt_trips...")
        cursor.execute(f"TRUNCATE TABLE {PG_SCHEMA_ANALYTICS}.obt_trips")
        conn.commit()
        log("Table truncated")

    total_start = datetime.now()
    total_records = 0

    for service in services:
        log(f"\n--- Processing service: {service} ---")

        for year in range(year_start, year_end + 1):
            for month in range(1, 13):
                if year == 2025 and month > datetime.now().month:
                    break

                partition_start = datetime.now()
                log(f"Processing {service} {year}-{month:02d}...")

                try:
                    records = insert_obt_partition(cursor, service, year, month, run_id)
                    conn.commit()

                    duration = (datetime.now() - partition_start).total_seconds()
                    total_records += records
                    log(f"  SUCCESS: {records:,} records in {duration:.2f}s")

                except Exception as e:
                    conn.rollback()
                    log(f"  FAILED: {str(e)}")

    total_duration = (datetime.now() - total_start).total_seconds()

    log("\n=== SUMMARY ===")
    log(f"Total records inserted: {total_records:,}")
    log(f"Total duration: {total_duration:.2f}s")
    log(f"OBT build completed successfully")

    cursor.close()

def build_obt_by_partition(conn, year_start, year_end, services, months, run_id, overwrite):
    log("=== MODE: BY-PARTITION ===")
    log(f"Years: {year_start}-{year_end}")
    log(f"Services: {services}")
    log(f"Months: {months if months else 'ALL'}")
    log(f"Run ID: {run_id}")
    log("")

    cursor = conn.cursor()
    total_start = datetime.now()
    total_records = 0

    months_to_process = months if months else list(range(1, 13))

    for service in services:
        log(f"\n--- Processing service: {service} ---")

        for year in range(year_start, year_end + 1):
            for month in months_to_process:
                if year == 2025 and month > datetime.now().month:
                    break

                if not overwrite:
                    cursor.execute(f"""
                        SELECT COUNT(*) FROM {PG_SCHEMA_ANALYTICS}.obt_trips
                        WHERE service_type = %s AND source_year = %s AND source_month = %s
                    """, (service, year, month))
                    count = cursor.fetchone()[0]
                    if count > 0:
                        log(f"Skipping {service} {year}-{month:02d} (already exists: {count:,} records)")
                        continue

                partition_start = datetime.now()
                log(f"Processing {service} {year}-{month:02d}...")

                try:
                    if overwrite:
                        cursor.execute(f"""
                            DELETE FROM {PG_SCHEMA_ANALYTICS}.obt_trips
                            WHERE service_type = %s AND source_year = %s AND source_month = %s
                        """, (service, year, month))

                    records = insert_obt_partition(cursor, service, year, month, run_id)
                    conn.commit()

                    duration = (datetime.now() - partition_start).total_seconds()
                    total_records += records
                    log(f"  SUCCESS: {records:,} records in {duration:.2f}s")

                except Exception as e:
                    conn.rollback()
                    log(f"  FAILED: {str(e)}")

    total_duration = (datetime.now() - total_start).total_seconds()

    log("\n=== SUMMARY ===")
    log(f"Total records inserted: {total_records:,}")
    log(f"Total duration: {total_duration:.2f}s")
    log(f"OBT build completed successfully")

    cursor.close()

def insert_obt_partition(cursor, service, year, month, run_id):
    raw_table = f"{PG_SCHEMA_RAW}.{service}_taxi_trip"

    if service == 'yellow':
        pickup_col = 'tpep_pickup_datetime'
        dropoff_col = 'tpep_dropoff_datetime'
    else:
        pickup_col = 'lpep_pickup_datetime'
        dropoff_col = 'lpep_dropoff_datetime'

    trip_type_col = 'trip_type' if service == 'green' else 'NULL'

    sql = f"""
    INSERT INTO {PG_SCHEMA_ANALYTICS}.obt_trips (
        pickup_datetime, dropoff_datetime, pickup_hour, pickup_dow, month, year,
        pu_location_id, pu_zone, pu_borough,
        do_location_id, do_zone, do_borough,
        service_type, vendor_id, vendor_name,
        rate_code_id, rate_code_desc,
        payment_type, payment_type_desc,
        trip_type,
        passenger_count, trip_distance,
        fare_amount, extra, mta_tax, tip_amount, tolls_amount,
        improvement_surcharge, congestion_surcharge, airport_fee, total_amount,
        store_and_fwd_flag,
        trip_duration_min, avg_speed_mph, tip_pct,
        run_id, source_year, source_month, ingested_at_utc
    )
    SELECT
        t.{pickup_col} as pickup_datetime,
        t.{dropoff_col} as dropoff_datetime,
        EXTRACT(HOUR FROM t.{pickup_col})::INTEGER as pickup_hour,
        EXTRACT(DOW FROM t.{pickup_col})::INTEGER as pickup_dow,
        EXTRACT(MONTH FROM t.{pickup_col})::INTEGER as month,
        EXTRACT(YEAR FROM t.{pickup_col})::INTEGER as year,

        t.pulocationid as pu_location_id,
        z1.zone as pu_zone,
        z1.borough as pu_borough,
        t.dolocationid as do_location_id,
        z2.zone as do_zone,
        z2.borough as do_borough,

        t.service_type,
        t.vendorid as vendor_id,
        CASE
            WHEN t.vendorid = 1 THEN 'Creative Mobile'
            WHEN t.vendorid = 2 THEN 'VeriFone Inc'
            ELSE 'Unknown'
        END as vendor_name,

        t.ratecodeid as rate_code_id,
        CASE
            WHEN t.ratecodeid = 1 THEN 'Standard rate'
            WHEN t.ratecodeid = 2 THEN 'JFK'
            WHEN t.ratecodeid = 3 THEN 'Newark'
            WHEN t.ratecodeid = 4 THEN 'Nassau or Westchester'
            WHEN t.ratecodeid = 5 THEN 'Negotiated fare'
            WHEN t.ratecodeid = 6 THEN 'Group ride'
            ELSE 'Unknown'
        END as rate_code_desc,

        t.payment_type,
        CASE
            WHEN t.payment_type = 1 THEN 'Credit card'
            WHEN t.payment_type = 2 THEN 'Cash'
            WHEN t.payment_type = 3 THEN 'No charge'
            WHEN t.payment_type = 4 THEN 'Dispute'
            WHEN t.payment_type = 5 THEN 'Unknown'
            WHEN t.payment_type = 6 THEN 'Voided trip'
            ELSE 'Unknown'
        END as payment_type_desc,

        {trip_type_col} as trip_type,

        t.passenger_count,
        t.trip_distance,
        t.fare_amount,
        t.extra,
        t.mta_tax,
        t.tip_amount,
        t.tolls_amount,
        t.improvement_surcharge,
        t.congestion_surcharge,
        t.airport_fee,
        t.total_amount,
        t.store_and_fwd_flag,

        EXTRACT(EPOCH FROM (t.{dropoff_col} - t.{pickup_col})) / 60.0 as trip_duration_min,
        CASE
            WHEN EXTRACT(EPOCH FROM (t.{dropoff_col} - t.{pickup_col})) > 0
            THEN t.trip_distance / (EXTRACT(EPOCH FROM (t.{dropoff_col} - t.{pickup_col})) / 3600.0)
            ELSE NULL
        END as avg_speed_mph,
        CASE
            WHEN t.fare_amount > 0 THEN (t.tip_amount / t.fare_amount) * 100
            ELSE 0
        END as tip_pct,

        t.run_id,
        t.source_year,
        t.source_month,
        t.ingested_at_utc
    FROM {raw_table} t
    LEFT JOIN {PG_SCHEMA_RAW}.taxi_zone_lookup z1 ON t.pulocationid = z1.locationid
    LEFT JOIN {PG_SCHEMA_RAW}.taxi_zone_lookup z2 ON t.dolocationid = z2.locationid
    WHERE t.source_year = %s AND t.source_month = %s
        AND t.{pickup_col} IS NOT NULL
        AND t.{dropoff_col} IS NOT NULL
        AND t.{dropoff_col} > t.{pickup_col}
        AND t.trip_distance >= 0
        AND EXTRACT(EPOCH FROM (t.{dropoff_col} - t.{pickup_col})) / 60.0 BETWEEN 0 AND 1440
    """

    cursor.execute(sql, (year, month))
    return cursor.rowcount

def main():
    parser = argparse.ArgumentParser(description='Build OBT from raw tables')
    parser.add_argument('--mode', choices=['full', 'by-partition'], default='full',
                        help='Build mode: full rebuild or by-partition')
    parser.add_argument('--year-start', type=int, default=2015,
                        help='Start year (default: 2015)')
    parser.add_argument('--year-end', type=int, default=2025,
                        help='End year (default: 2025)')
    parser.add_argument('--months', type=str, default=None,
                        help='Months to process (comma-separated, e.g., 1,2,3)')
    parser.add_argument('--services', type=str, default='yellow,green',
                        help='Services to process (comma-separated)')
    parser.add_argument('--run-id', type=str, default=f"obt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        help='Run ID for tracking')
    parser.add_argument('--overwrite', type=str, choices=['true', 'false'], default='false',
                        help='Overwrite existing data')
    parser.add_argument('--full-rebuild', action='store_true',
                        help='Shortcut for --mode full --overwrite true')

    args = parser.parse_args()

    if args.full_rebuild:
        args.mode = 'full'
        args.overwrite = 'true'

    services = [s.strip() for s in args.services.split(',')]
    months = [int(m.strip()) for m in args.months.split(',')] if args.months else None
    overwrite = args.overwrite == 'true'

    log("=== OBT BUILDER STARTED ===")
    log(f"Postgres: {PG_HOST}:{PG_PORT}/{PG_DB}")
    log("")

    try:
        conn = get_connection()
        log("Connected to Postgres")

        if args.mode == 'full':
            build_obt_full(conn, args.year_start, args.year_end, services, args.run_id, overwrite)
        else:
            build_obt_by_partition(conn, args.year_start, args.year_end, services, months, args.run_id, overwrite)

        conn.close()
        log("\n=== OBT BUILDER FINISHED ===")
        return 0

    except Exception as e:
        log(f"\n!!! ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
