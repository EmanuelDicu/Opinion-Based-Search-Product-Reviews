.PHONY: build up down restart logs clean

# Build all containers
build:
	docker-compose build

# Start all services
up:
	docker-compose up -d

# Start all services with logs
up-logs:
	docker-compose up

# Stop all services
down:
	docker-compose down

# Restart all services
restart:
	docker-compose restart

# View logs
logs:
	docker-compose logs -f

# Clean up containers and volumes
clean:
	docker-compose down -v
	docker system prune -f

# Rebuild and restart
rebuild: build up

# Run preprocessing only
preprocess:
	docker-compose run --rm preprocessing

# Check service status
status:
	docker-compose ps

