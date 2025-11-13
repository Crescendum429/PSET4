#!/usr/bin/env python3
import os
import csv
import psycopg2
from dotenv import load_dotenv

load_dotenv()

PG_HOST = os.getenv('PG_HOST', 'postgres')
PG_PORT = os.getenv('PG_PORT', '5432')
PG_DB = os.getenv('PG_DB', 'nyc_taxi')
PG_USER = os.getenv('PG_USER')
PG_PASSWORD = os.getenv('PG_PASSWORD')
PG_SCHEMA_RAW = os.getenv('PG_SCHEMA_RAW', 'raw')

def load_taxi_zones(csv_path):
    conn = psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        database=PG_DB,
        user=PG_USER,
        password=PG_PASSWORD
    )
    cursor = conn.cursor()

    print(f"Loading taxi zones from {csv_path}...")

    cursor.execute(f"TRUNCATE TABLE {PG_SCHEMA_RAW}.taxi_zone_lookup")

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            cursor.execute(f"""
                INSERT INTO {PG_SCHEMA_RAW}.taxi_zone_lookup
                (locationid, borough, zone, service_zone)
                VALUES (%s, %s, %s, %s)
            """, (
                int(row['LocationID']),
                row['Borough'],
                row['Zone'],
                row['service_zone']
            ))
            count += 1

    conn.commit()
    cursor.close()
    conn.close()

    print(f"Loaded {count} taxi zones")

if __name__ == '__main__':
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else '/home/jovyan/data/taxi_zone_lookup.csv'
    load_taxi_zones(csv_path)
