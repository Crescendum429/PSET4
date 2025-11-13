# PSET4: NYC Taxi ML Pipeline

Pipeline de datos e ML sobre el dataset NYC TLC (año 2020). Incluye ingesta con Spark, construcción de OBT en Postgres, y predicción de total_amount con modelos de regresión.

## Arquitectura

```
NYC TLC Parquet → Spark → Postgres RAW → OBT Builder → Analytics OBT → ML
```

## Requisitos

- Docker y Docker Compose
- Mínimo 12GB RAM
- Conexión a internet

## Configuración

Crear archivo `.env` a partir de `.env.example`:

```bash
cp .env.example .env
```

Editar `.env` y configurar:
```bash
PG_PASSWORD=tu_password
```

Las demás variables tienen valores por defecto funcionales.

## Ejecución

### 1. Levantar servicios

```bash
docker compose up -d
```

Verificar:
```bash
docker compose ps
```

Obtener token de Jupyter:
```bash
docker logs spark-notebook 2>&1 | grep token
```

Acceder a: http://localhost:8888

### 2. Ingesta RAW

Abrir `notebooks/01_ingesta_parquet_raw.ipynb` y ejecutar todas las celdas.

Tiempo: 15-20 minutos (24 archivos del año 2020)

Resultado: ~32.8M registros en `raw.yellow_taxi_trip` y `raw.green_taxi_trip`

### 3. Cargar taxi zones

```bash
docker exec -it spark-notebook python /home/jovyan/scripts/load_taxi_zones.py /home/jovyan/data/taxi_zone_lookup.csv
```

### 4. Construir OBT

```bash
docker compose run --rm obt-builder --full-rebuild
```

Este comando lee las tablas RAW, realiza JOIN con taxi_zone_lookup, calcula métricas derivadas (trip_duration_min, avg_speed_mph, tip_pct), y escribe en `analytics.obt_trips`.

Tiempo: 15 minutos

Resultado: 32,738,347 registros en `analytics.obt_trips`

Verificar:
```bash
docker exec -it postgres-nyc-taxi psql -U taxi_user -d nyc_taxi -c "SELECT COUNT(*) FROM analytics.obt_trips;"
```

### 5. Machine Learning

Abrir `notebooks/ml_total_amount_regression.ipynb` y ejecutar todas las celdas.

Tiempo: 10-15 minutos

Outputs:
- `model_comparison.csv`: comparación de 4 modelos sklearn
- `diagnostic_plots.png`: residuales y Q-Q plot
- `error_by_bucket.csv`: errores por percentiles

Nota: Los modelos from-scratch (NumPy) están comentados en el notebook por velocidad. El código está presente pero no se ejecuta por defecto.

## Estructura

```
PSET4/
├── docker-compose.yml
├── .env.example
├── notebooks/
│   ├── 01_ingesta_parquet_raw.ipynb
│   └── ml_total_amount_regression.ipynb
├── scripts/
│   ├── models_from_scratch.py
│   └── load_taxi_zones.py
├── sql/
│   ├── 01_init_schemas.sql
│   ├── 02_create_raw_tables.sql
│   └── 03_create_analytics_obt.sql
├── obt-builder/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── build_obt.py
└── data/
    └── taxi_zone_lookup.csv
```

## Comandos adicionales

Conectar a Postgres:
```bash
docker exec -it postgres-nyc-taxi psql -U taxi_user -d nyc_taxi
```

Ver logs OBT:
```bash
docker compose logs obt-builder
```

Construir OBT por particiones:
```bash
docker compose run --rm obt-builder --mode by-partition --year-start 2020 --year-end 2020 --months 1,2,3
```

Detener servicios:
```bash
docker compose down
```

Limpiar volúmenes:
```bash
docker compose down -v
```

## Alcance

El proyecto está configurado para procesar únicamente el año 2020 (variable `YEARS=2020` en `.env`). El pipeline está diseñado para soportar 2015-2025, pero por tiempos de ejecución se usa solo 2020 para demostración.

Para procesar años adicionales, editar las variables en `.env`:
```bash
YEARS=2020,2021,2022
OBT_YEAR_START=2020
OBT_YEAR_END=2022
```

## Entregables

Según PDF Sección 11:

1. Docker Compose con 3 servicios (spark-notebook, postgres, obt-builder)
2. Script CLI `obt-builder/build_obt.py` con modos full y by-partition
3. Notebooks de ingesta y ML
4. Scripts auxiliares (load_taxi_zones.py, models_from_scratch.py)
5. Archivos SQL de schemas y tablas
6. Evidencias: model_comparison.csv, diagnostic_plots.png, error_by_bucket.csv

Universidad San Francisco de Quito - Data Mining 2025
