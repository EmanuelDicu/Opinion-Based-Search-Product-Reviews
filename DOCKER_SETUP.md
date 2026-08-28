# Docker Setup Guide

This project is now fully containerized using Docker and Docker Compose.

## Architecture

The application consists of 4 main services:

1. **Elasticsearch** - Search engine for indexing and querying reviews
2. **Preprocessing** - Data processing pipeline (runs once on startup)
3. **Backend** - Flask API server
4. **Frontend** - React web application served via Nginx

## Quick Start

```bash
# Build and start all services
docker-compose up --build

# Or use Makefile
make build
make up
```

## Services

### Elasticsearch
- **Image**: `docker.elastic.co/elasticsearch/elasticsearch:8.11.0`
- **Port**: 9200
- **Volume**: `elasticsearch_data` (persistent storage)
- **Health Check**: Enabled

### Preprocessing
- **Purpose**: Extracts aspects and opinions from reviews
- **Runs**: Once on startup (or manually)
- **Output**: Indexes data in Elasticsearch and saves to `/app/data/extractions.json`

### Backend
- **Framework**: Flask
- **Port**: 5000
- **Health Check**: `/api/health`
- **Dependencies**: Elasticsearch, Preprocessing

### Frontend
- **Framework**: React
- **Port**: 3000 (mapped to Nginx port 80)
- **Proxy**: API requests proxied to backend via Nginx

## Environment Variables

All services can be configured via environment variables in `docker-compose.yml`:

- `DATASET_LIMIT`: Number of reviews to process
- `ELASTICSEARCH_HOST`: Elasticsearch hostname
- `ELASTICSEARCH_PORT`: Elasticsearch port
- `FLASK_PORT`: Backend API port
- `CORS_ORIGINS`: Allowed CORS origins

## Data Persistence

- Elasticsearch data: Stored in Docker volume `elasticsearch_data`
- Extracted reviews: Stored in `./data/extractions.json` (mounted volume)

## Development

### Rebuild after code changes

```bash
# Rebuild specific service
docker-compose build backend
docker-compose up backend

# Rebuild all
docker-compose build
```

### View logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
```

### Run preprocessing manually

```bash
docker-compose run --rm preprocessing
```

### Access services

- Frontend: http://localhost:3000
- Backend API: http://localhost:5000
- Elasticsearch: http://localhost:9200

## Troubleshooting

### Elasticsearch not starting
- Check available memory (needs at least 512MB)
- Check logs: `docker-compose logs elasticsearch`

### Preprocessing fails
- Ensure Elasticsearch is healthy: `docker-compose ps`
- Check preprocessing logs: `docker-compose logs preprocessing`

### Backend can't connect to Elasticsearch
- Verify Elasticsearch is running: `curl http://localhost:9200`
- Check network: `docker-compose ps`
- Ensure service dependencies are correct in docker-compose.yml

## Clean Up

```bash
# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Or use Makefile
make clean
```

