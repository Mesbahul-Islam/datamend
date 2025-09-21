# Docker Deployment Guide

This guide covers how to deploy DataMend using Docker containers for both development and production environments.

## Prerequisites

- Docker Engine 20.10+ installed
- Docker Compose V2 installed
- At least 4GB RAM available for containers
- Required environment variables configured

## Quick Start

### Development Environment

1. Clone the repository and navigate to the project directory
2. Copy the environment template:
   ```bash
   cp .env.example .env
   ```
3. Edit `.env` with your actual credentials
4. Start the development environment:
   ```bash
   docker-compose -f docker-compose.dev.yml up --build
   ```
5. Access the application at: http://localhost:8501

### Production Environment

1. Set up production directories:
   ```bash
   sudo mkdir -p /opt/datamend/{data,logs,wallet}
   sudo chown $(whoami):$(whoami) /opt/datamend -R
   ```

2. Copy your Oracle wallet files to `/opt/datamend/wallet/` (if using Oracle)

3. Create production environment file:
   ```bash
   cp .env.example .env.prod
   # Edit .env.prod with production values
   ```

4. Deploy with production compose:
   ```bash
   docker-compose -f docker-compose.prod.yml --env-file .env.prod up -d
   ```

5. Access the application at: http://localhost (port 80)

## Environment Configuration

### Required Environment Variables

Create a `.env` file with the following variables:

```bash
# LLM Configuration
LLM_PROVIDER=openai
LLM_API_KEY=your_openai_api_key
LLM_MODEL=gpt-4

# Snowflake Configuration
SNOWFLAKE_ACCOUNT=your_account.region
SNOWFLAKE_USERNAME=your_username
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_DATABASE=your_database
SNOWFLAKE_SCHEMA=your_schema

# Oracle Cloud Configuration
ORACLE_CONNECTION_STRING=tcps://host:port/service_name
ORACLE_USERNAME=your_username
ORACLE_PASSWORD=your_password

# HDFS Configuration
HDFS_NAMENODE_URL=hdfs://namenode:9000
HDFS_USERNAME=hdfs_user
HDFS_TIMEOUT=10
```

### Optional Environment Variables

```bash
# Streamlit Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

## Docker Images

### Base Image
- **Base**: Python 3.11-slim
- **Security**: Non-root user execution
- **Optimization**: Multi-stage build for smaller image size
- **Health Checks**: Built-in health monitoring

### Image Size Optimization
The Docker image uses multi-stage builds to minimize the final image size:
- Builder stage: Installs build dependencies
- Production stage: Only runtime dependencies
- Expected final image size: ~500MB

## Networking

### Development
- Application: http://localhost:8501
- PostgreSQL (optional): localhost:5432

### Production
- Application: http://localhost:80 (via nginx)
- HTTPS: https://localhost:443 (with SSL certificates)
- Internal container communication on bridge network

## Volume Mounts

### Development
- Source code: `.:/app` (for live reload)
- Data directory: `./data:/app/data`
- Oracle wallet: `./wallet:/app/wallet:ro`

### Production
- Data directory: `/opt/datamend/data:/app/data:ro`
- Logs: `/opt/datamend/logs:/app/logs`
- Oracle wallet: `/opt/datamend/wallet:/app/wallet:ro`

## Health Monitoring

The containers include health checks that monitor:
- Application responsiveness
- Streamlit server status
- Container resource usage

Check health status:
```bash
docker-compose ps
docker inspect datamend-prod | grep Health -A 10
```

## Scaling and Performance

### Resource Limits
- Development: 1 CPU, 2GB RAM
- Production: 2 CPU, 4GB RAM (configurable)

### Horizontal Scaling
To run multiple instances:
```bash
docker-compose -f docker-compose.prod.yml up -d --scale datamend=3
```

### Load Balancing
Use the included nginx service for load balancing multiple instances.

## Troubleshooting

### Common Issues

1. **Port conflicts**:
   ```bash
   # Check what's using the port
   sudo lsof -i :8501
   # Change port in docker-compose.yml
   ```

2. **Permission errors**:
   ```bash
   # Fix data directory permissions
   sudo chown -R 1000:1000 /opt/datamend/
   ```

3. **Memory issues**:
   ```bash
   # Check container memory usage
   docker stats
   # Increase memory limits in docker-compose.yml
   ```

4. **Oracle wallet issues**:
   ```bash
   # Verify wallet files
   ls -la /opt/datamend/wallet/
   # Check file permissions
   ```

### Debugging

1. **View logs**:
   ```bash
   # Application logs
   docker-compose logs datamend
   
   # Follow logs in real-time
   docker-compose logs -f datamend
   ```

2. **Access container shell**:
   ```bash
   docker exec -it datamend-prod /bin/bash
   ```

3. **Check container health**:
   ```bash
   docker inspect datamend-prod --format='{{.State.Health.Status}}'
   ```

## Security Considerations

### Production Security
- Containers run as non-root user (uid 1000)
- Sensitive data mounted as read-only volumes
- Network isolation between services
- Regular security updates for base images

### SSL/TLS Configuration
For production HTTPS, configure nginx with SSL certificates:
1. Place certificates in `/etc/ssl/`
2. Update nginx.conf with SSL configuration
3. Restart nginx service

## Backup and Recovery

### Data Backup
```bash
# Backup data directory
tar -czf datamend-backup-$(date +%Y%m%d).tar.gz /opt/datamend/data/

# Backup container volumes
docker run --rm -v datamend_data:/data -v $(pwd):/backup ubuntu tar czf /backup/data-backup.tar.gz /data
```

### Container Updates
```bash
# Update to latest version
docker-compose pull
docker-compose up -d

# Rollback if needed
docker-compose down
docker tag datamend:previous datamend:latest
docker-compose up -d
```

## Monitoring and Maintenance

### Log Rotation
Production setup includes automatic log rotation:
- Max log size: 100MB
- Max log files: 5
- Automatic cleanup of old logs

### Health Monitoring
Set up monitoring for:
- Container health status
- Resource usage (CPU, memory)
- Application response times
- Error rates in logs

### Regular Maintenance
- Update base images monthly
- Review and rotate logs
- Monitor disk space usage
- Update environment variables as needed

## Support

For issues specific to Docker deployment:
1. Check the troubleshooting section above
2. Review container logs for error messages
3. Verify environment variable configuration
4. Ensure all required volumes are properly mounted
