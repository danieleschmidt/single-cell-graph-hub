# Maintenance Runbook

This runbook covers routine maintenance procedures for Single-Cell Graph Hub.

## Scheduled Maintenance Windows

### Weekly Maintenance
- **When**: Sunday 02:00-04:00 UTC
- **Duration**: 2 hours maximum
- **Activities**: Updates, patches, routine maintenance

### Monthly Maintenance  
- **When**: First Sunday of month 01:00-05:00 UTC
- **Duration**: 4 hours maximum
- **Activities**: Major updates, infrastructure changes

### Emergency Maintenance
- **When**: As needed
- **Duration**: Variable
- **Activities**: Critical security patches, major incident recovery

## Pre-Maintenance Checklist

### 24 Hours Before

- [ ] Announce maintenance window to users
- [ ] Update status page with maintenance schedule
- [ ] Verify backup procedures are working
- [ ] Create rollback plan
- [ ] Prepare maintenance scripts
- [ ] Schedule post-maintenance testing

### 2 Hours Before

- [ ] Perform full system backup
- [ ] Verify all team members are available
- [ ] Confirm maintenance window with stakeholders
- [ ] Set up monitoring for maintenance activities
- [ ] Prepare communication channels

### 30 Minutes Before

- [ ] Final system health check
- [ ] Enable maintenance mode if available
- [ ] Notify users of impending maintenance
- [ ] Begin log collection for maintenance period

## System Updates

### Application Updates

1. **Backup Current Version**
   ```bash
   # Tag current state
   git tag "pre-maintenance-$(date +%Y%m%d)"
   
   # Backup current container
   docker commit scgraph-hub-app scgraph-hub:backup-$(date +%Y%m%d)
   
   # Export configuration
   docker-compose config > config-backup-$(date +%Y%m%d).yml
   ```

2. **Deploy New Version**
   ```bash
   # Pull latest changes
   git pull origin main
   
   # Build new image
   docker-compose build scgraph-hub
   
   # Update containers with zero-downtime deployment
   docker-compose up -d --force-recreate --no-deps scgraph-hub
   
   # Verify deployment
   ./scripts/health-check.sh
   ```

3. **Database Migrations**
   ```bash
   # Check for pending migrations
   docker exec scgraph-hub-app python -m scgraph_hub.db check_migrations
   
   # Run migrations
   docker exec scgraph-hub-app python -m scgraph_hub.db migrate
   
   # Verify migration success
   docker exec scgraph-postgres psql -U scgraph -d scgraph_hub -c "\dt"
   ```

### System Package Updates

1. **Update Base Images**
   ```bash
   # Update Docker base images
   docker pull python:3.11-slim
   docker pull postgres:15-alpine
   docker pull redis:7-alpine
   
   # Rebuild with updated base images
   docker-compose build --pull
   ```

2. **Security Updates**
   ```bash
   # Update system packages in containers
   docker exec scgraph-hub-app apt-get update && apt-get upgrade -y
   
   # Scan for vulnerabilities
   docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
     aquasec/trivy image single-cell-graph-hub:latest
   ```

### Dependency Updates

1. **Update Python Dependencies**
   ```bash
   # Check for outdated packages
   docker exec scgraph-hub-app pip list --outdated
   
   # Update specific critical packages
   docker exec scgraph-hub-app pip install --upgrade torch torch-geometric
   
   # Update all packages (carefully)
   docker exec scgraph-hub-app pip install --upgrade -r requirements.txt
   
   # Verify no breaking changes
   docker exec scgraph-hub-app python -m pytest tests/unit/
   ```

2. **Update System Dependencies**
   ```bash
   # Update Docker Compose version
   curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" \
     -o /usr/local/bin/docker-compose
   chmod +x /usr/local/bin/docker-compose
   
   # Update monitoring stack
   docker-compose -f docker-compose.monitoring.yml pull
   docker-compose -f docker-compose.monitoring.yml up -d
   ```

## Database Maintenance

### PostgreSQL Maintenance

1. **Routine Optimization**
   ```sql
   -- Update table statistics
   ANALYZE;
   
   -- Vacuum tables to reclaim space
   VACUUM ANALYZE;
   
   -- Check for bloated tables
   SELECT schemaname, tablename, 
          pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
   FROM pg_tables 
   WHERE schemaname NOT IN ('information_schema', 'pg_catalog')
   ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
   
   -- Reindex if needed
   REINDEX DATABASE scgraph_hub;
   ```

2. **Connection Pool Maintenance**
   ```bash
   # Check connection pool status
   docker exec scgraph-postgres psql -U scgraph -d scgraph_hub \
     -c "SELECT count(*), state FROM pg_stat_activity GROUP BY state;"
   
   # Kill idle connections
   docker exec scgraph-postgres psql -U scgraph -d scgraph_hub \
     -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity 
         WHERE state = 'idle' AND state_change < now() - interval '1 hour';"
   ```

3. **Backup Verification**
   ```bash
   # Test backup restoration
   docker exec scgraph-postgres pg_dump -U scgraph scgraph_hub > test_backup.sql
   
   # Create test database and restore
   docker exec scgraph-postgres psql -U scgraph -c "CREATE DATABASE test_restore;"
   docker exec -i scgraph-postgres psql -U scgraph test_restore < test_backup.sql
   
   # Verify backup integrity
   docker exec scgraph-postgres psql -U scgraph test_restore -c "SELECT count(*) FROM datasets.catalog;"
   
   # Clean up test database
   docker exec scgraph-postgres psql -U scgraph -c "DROP DATABASE test_restore;"
   rm test_backup.sql
   ```

### Redis Maintenance

1. **Memory Optimization**
   ```bash
   # Check memory usage
   docker exec scgraph-redis redis-cli info memory
   
   # Clear expired keys
   docker exec scgraph-redis redis-cli --eval "
     for i=0,redis.call('CONFIG','GET','databases')[2]-1 do
       redis.call('SELECT',i)
       redis.call('SCAN',0,'MATCH','*','COUNT',1000)
     end" 0
   
   # Save current dataset
   docker exec scgraph-redis redis-cli BGSAVE
   
   # Check save status
   docker exec scgraph-redis redis-cli LASTSAVE
   ```

2. **Performance Tuning**
   ```bash
   # Check slow queries
   docker exec scgraph-redis redis-cli SLOWLOG GET 10
   
   # Monitor real-time commands
   docker exec scgraph-redis redis-cli MONITOR | head -100
   
   # Check configuration
   docker exec scgraph-redis redis-cli CONFIG GET "*"
   ```

## Log Maintenance

### Log Rotation

1. **Application Logs**
   ```bash
   # Check log sizes
   docker exec scgraph-hub-app du -sh /app/logs/
   
   # Rotate logs manually if needed
   docker exec scgraph-hub-app logrotate -f /etc/logrotate.conf
   
   # Compress old logs
   docker exec scgraph-hub-app find /app/logs/ -name "*.log.*" -mtime +7 -exec gzip {} \;
   
   # Clean up old compressed logs
   docker exec scgraph-hub-app find /app/logs/ -name "*.gz" -mtime +30 -delete
   ```

2. **Docker Logs**
   ```bash
   # Check Docker log sizes
   docker system df
   
   # Clean up Docker logs
   docker system prune -f
   
   # Configure log rotation for containers
   cat > /etc/docker/daemon.json << EOF
   {
     "log-driver": "json-file",
     "log-opts": {
       "max-size": "10m",
       "max-file": "3"
     }
   }
   EOF
   
   systemctl restart docker
   ```

### Log Analysis

1. **Error Pattern Analysis**
   ```bash
   # Check for error patterns
   docker logs scgraph-hub-app --since 24h 2>&1 | grep ERROR | sort | uniq -c | sort -rn
   
   # Check for memory issues
   docker logs scgraph-hub-app --since 24h 2>&1 | grep -i "memory\|oom"
   
   # Check for database issues
   docker logs scgraph-postgres --since 24h 2>&1 | grep ERROR
   ```

2. **Performance Analysis**
   ```bash
   # Extract slow requests from logs
   docker logs scgraph-hub-app --since 24h 2>&1 | grep "response_time" | \
     awk '{if($NF > 1000) print}' | head -20
   
   # Check request patterns
   docker logs scgraph-hub-app --since 24h 2>&1 | grep "api_request" | \
     awk '{print $5}' | sort | uniq -c | sort -rn
   ```

## Storage Maintenance

### Disk Space Management

1. **Check Disk Usage**
   ```bash
   # Overall disk usage
   df -h
   
   # Container-specific usage
   docker system df
   
   # Application data usage
   docker exec scgraph-hub-app du -sh /app/data/*
   
   # Database usage
   docker exec scgraph-postgres du -sh /var/lib/postgresql/data
   ```

2. **Clean Up Unnecessary Files**
   ```bash
   # Clean temporary files
   docker exec scgraph-hub-app find /tmp -type f -mtime +7 -delete
   
   # Clean cache directories
   docker exec scgraph-hub-app find /app/cache -type f -mtime +30 -delete
   
   # Clean old model checkpoints
   docker exec scgraph-hub-app find /app/models -name "checkpoint_*" -mtime +90 -delete
   
   # Docker cleanup
   docker system prune -af --volumes
   ```

### Backup Maintenance

1. **Verify Backups**
   ```bash
   # Check backup schedule
   crontab -l | grep backup
   
   # Verify latest backup
   ls -la /backup/ | head -10
   
   # Test backup restoration (non-destructive)
   ./scripts/test-backup-restore.sh
   ```

2. **Clean Old Backups**
   ```bash
   # Remove backups older than 30 days
   find /backup -name "*.sql" -mtime +30 -delete
   find /backup -name "*.tar.gz" -mtime +30 -delete
   
   # Keep only weekly backups older than 90 days
   find /backup -name "*weekly*" -mtime +90 -delete
   ```

## Security Maintenance

### Certificate Management

1. **SSL Certificate Renewal**
   ```bash
   # Check certificate expiration
   openssl x509 -in /etc/ssl/certs/scgraphhub.crt -text -noout | grep "Not After"
   
   # Renew Let's Encrypt certificates
   certbot renew --dry-run
   certbot renew
   
   # Restart services to load new certificates
   docker-compose restart nginx
   ```

2. **Update Security Configurations**
   ```bash
   # Update security headers
   docker exec scgraph-nginx nginx -t
   docker exec scgraph-nginx nginx -s reload
   
   # Check security configuration
   curl -I https://scgraphhub.org | grep -i security
   ```

### Access Control Review

1. **Review User Access**
   ```bash
   # Check active sessions
   docker exec scgraph-postgres psql -U scgraph -d scgraph_hub \
     -c "SELECT username, last_login FROM auth_users WHERE last_login > now() - interval '30 days';"
   
   # Review administrative access
   docker exec scgraph-postgres psql -U scgraph -d scgraph_hub \
     -c "SELECT username, is_superuser FROM auth_users WHERE is_superuser = true;"
   ```

2. **Update API Keys**
   ```bash
   # Rotate API keys (if applicable)
   docker exec scgraph-hub-app python -m scgraph_hub.auth rotate_keys
   
   # Update external service credentials
   docker-compose up -d --force-recreate scgraph-hub
   ```

## Monitoring System Maintenance

### Prometheus Maintenance

1. **Data Retention Management**
   ```bash
   # Check data size
   docker exec scgraph-prometheus du -sh /prometheus
   
   # Compact old data
   docker exec scgraph-prometheus promtool tsdb create-blocks-from-snapshot \
     /prometheus/snapshots/snapshot-name /prometheus/data
   
   # Clean up old snapshots
   docker exec scgraph-prometheus find /prometheus/snapshots -mtime +7 -exec rm -rf {} \;
   ```

2. **Configuration Updates**
   ```bash
   # Validate configuration
   docker exec scgraph-prometheus promtool check config /etc/prometheus/prometheus.yml
   
   # Reload configuration
   docker exec scgraph-prometheus kill -HUP 1
   
   # Verify reload
   docker logs scgraph-prometheus | tail -10
   ```

### Grafana Maintenance

1. **Dashboard Cleanup**
   ```bash
   # Export important dashboards
   curl -X GET "http://admin:admin@localhost:3000/api/dashboards/uid/dashboard-uid" > dashboard-backup.json
   
   # Clean up old snapshots
   docker exec scgraph-grafana find /var/lib/grafana -name "*.snapshot" -mtime +30 -delete
   ```

2. **Plugin Updates**
   ```bash
   # List installed plugins
   docker exec scgraph-grafana grafana-cli plugins ls
   
   # Update all plugins
   docker exec scgraph-grafana grafana-cli plugins update-all
   
   # Restart Grafana
   docker-compose restart grafana
   ```

## Performance Optimization

### Database Performance

1. **Query Optimization**
   ```sql
   -- Identify slow queries
   SELECT query, calls, total_time, mean_time 
   FROM pg_stat_statements 
   ORDER BY total_time DESC 
   LIMIT 10;
   
   -- Check index usage
   SELECT schemaname, tablename, attname, n_distinct, correlation 
   FROM pg_stats 
   WHERE schemaname = 'datasets' 
   ORDER BY n_distinct DESC;
   
   -- Add missing indexes if needed
   CREATE INDEX CONCURRENTLY idx_datasets_organism 
   ON datasets.catalog(organism) 
   WHERE organism IS NOT NULL;
   ```

2. **Connection Pool Optimization**
   ```bash
   # Monitor connection pool
   docker logs scgraph-hub-app | grep "connection_pool" | tail -20
   
   # Adjust pool settings if needed
   # Edit docker-compose.yml to update DATABASE_POOL_SIZE
   docker-compose up -d --force-recreate scgraph-hub
   ```

### Application Performance

1. **Memory Optimization**
   ```bash
   # Check memory usage patterns
   docker stats scgraph-hub-app --no-stream --format "table {{.MemUsage}}\t{{.MemPerc}}"
   
   # Profile memory usage
   docker exec scgraph-hub-app python -m memory_profiler /app/src/scgraph_hub/main.py
   
   # Garbage collection tuning
   docker exec scgraph-hub-app python -c "import gc; gc.set_threshold(700, 10, 10)"
   ```

2. **Cache Optimization**
   ```bash
   # Check cache hit rates
   docker exec scgraph-redis redis-cli info stats | grep hit_rate
   
   # Adjust cache policies
   docker exec scgraph-redis redis-cli config set maxmemory-policy allkeys-lru
   
   # Warm up critical caches
   docker exec scgraph-hub-app python -m scgraph_hub.cache warm_up
   ```

## Post-Maintenance Procedures

### System Verification

1. **Health Checks**
   ```bash
   # Comprehensive health check
   ./scripts/health-check-detailed.sh
   
   # Performance benchmarks
   ./scripts/benchmark.sh
   
   # Security scan
   ./scripts/security-scan.sh
   ```

2. **User Acceptance Testing**
   ```bash
   # Run smoke tests
   docker exec scgraph-hub-app python -m pytest tests/smoke/
   
   # Load test critical endpoints
   ab -n 100 -c 10 http://localhost:8000/api/v1/datasets
   
   # Verify data integrity
   docker exec scgraph-hub-app python -m scgraph_hub.validate data_integrity
   ```

### Documentation Updates

1. **Update Maintenance Log**
   ```markdown
   # Maintenance Log Entry
   
   Date: 2025-01-01 02:00 UTC
   Duration: 1.5 hours
   Type: Weekly routine maintenance
   
   Activities Completed:
   - Updated application to v0.2.0
   - Database optimization and vacuuming
   - Log rotation and cleanup
   - Security certificate renewal
   
   Issues Encountered:
   - None
   
   Next Maintenance: 2025-01-08 02:00 UTC
   ```

2. **Update System Documentation**
   - Update version numbers in documentation
   - Document any configuration changes
   - Update monitoring thresholds if changed
   - Record any lessons learned

### Communication

1. **Completion Notification**
   ```
   MAINTENANCE COMPLETE
   
   Maintenance Window: 2025-01-01 02:00-03:30 UTC
   Services Affected: All services
   Downtime: 15 minutes (during application restart)
   
   Changes Made:
   - Application updated to v0.2.0
   - Performance optimizations applied
   - Security patches installed
   
   All services are now fully operational.
   
   Next scheduled maintenance: 2025-01-08 02:00 UTC
   ```

2. **Status Page Update**
   - Mark maintenance as completed
   - Update system status to operational
   - Post summary of maintenance activities

## Emergency Procedures

### Rollback Procedures

1. **Application Rollback**
   ```bash
   # Stop current version
   docker-compose stop scgraph-hub
   
   # Restore previous version
   docker tag scgraph-hub:backup-$(date +%Y%m%d) scgraph-hub:latest
   
   # Start with previous configuration
   docker-compose -f config-backup-$(date +%Y%m%d).yml up -d scgraph-hub
   
   # Verify rollback
   ./scripts/health-check.sh
   ```

2. **Database Rollback**
   ```bash
   # Stop application
   docker-compose stop scgraph-hub
   
   # Restore database
   docker exec -i scgraph-postgres psql -U scgraph scgraph_hub < pre_maintenance_backup.sql
   
   # Restart application
   docker-compose start scgraph-hub
   ```

### Contact Information

- **Primary On-Call**: [Phone number]
- **Secondary On-Call**: [Phone number]
- **Engineering Manager**: [Phone number]
- **Infrastructure Team**: [Slack channel]

### Emergency Resources

- **Break Glass Procedures**: [Link to emergency access procedures]
- **Vendor Support Contacts**: [List of critical vendor contacts]
- **Emergency Communication Plan**: [Link to communication procedures]