# Incident Response Runbook

This runbook provides step-by-step procedures for responding to incidents in Single-Cell Graph Hub.

## Incident Classification

### Severity Levels

#### SEV-1 (Critical)
- **Definition**: Complete service outage or critical functionality unavailable
- **Examples**: Application down, database corruption, security breach
- **Response Time**: Immediate (< 5 minutes)
- **Escalation**: On-call engineer + manager

#### SEV-2 (High)
- **Definition**: Significant degradation affecting multiple users
- **Examples**: High error rates, slow response times, partial functionality loss
- **Response Time**: < 15 minutes
- **Escalation**: On-call engineer

#### SEV-3 (Medium)
- **Definition**: Limited impact on subset of users
- **Examples**: Single dataset unavailable, minor feature issues
- **Response Time**: < 1 hour
- **Escalation**: Normal business hours

#### SEV-4 (Low)
- **Definition**: Minor issues with workarounds available
- **Examples**: UI glitches, documentation errors
- **Response Time**: Next business day
- **Escalation**: None required

## Common Incident Scenarios

### 1. Service Unavailable (SEV-1)

#### Symptoms
- Health check endpoint returns 5xx errors
- Users cannot access the application
- Monitoring alerts firing

#### Investigation Steps

1. **Check Service Status**
   ```bash
   # Check if service is running
   docker ps | grep scgraph-hub
   
   # Check service logs
   docker logs scgraph-hub-app --tail 100
   
   # Check health endpoint
   curl -I http://localhost:8000/health
   ```

2. **Check Dependencies**
   ```bash
   # Database connectivity
   docker exec scgraph-postgres pg_isready -U scgraph -d scgraph_hub
   
   # Redis connectivity  
   docker exec scgraph-redis redis-cli ping
   
   # Storage accessibility
   docker exec scgraph-hub python -c "import boto3; s3=boto3.client('s3'); print(s3.list_buckets())"
   ```

3. **Check Resources**
   ```bash
   # Memory usage
   docker stats scgraph-hub-app --no-stream
   
   # Disk space
   df -h
   
   # CPU usage
   top -p $(docker inspect -f '{{.State.Pid}}' scgraph-hub-app)
   ```

#### Resolution Steps

1. **Restart Services**
   ```bash
   # Restart application
   docker-compose restart scgraph-hub
   
   # If that fails, full restart
   docker-compose down && docker-compose up -d
   ```

2. **Scale Resources (if resource constrained)**
   ```bash
   # Increase memory limits
   docker-compose -f docker-compose.yml -f docker-compose.scale.yml up -d
   
   # Scale to multiple instances
   docker-compose up -d --scale scgraph-hub=3
   ```

3. **Database Recovery**
   ```bash
   # If database is corrupted
   docker exec scgraph-postgres pg_dump -U scgraph scgraph_hub > backup.sql
   docker-compose restart postgres
   docker exec -i scgraph-postgres psql -U scgraph scgraph_hub < backup.sql
   ```

### 2. High Error Rate (SEV-2)

#### Symptoms
- Error rate > 5% for > 5 minutes
- 5xx HTTP responses increasing
- User complaints about failures

#### Investigation Steps

1. **Check Error Patterns**
   ```bash
   # Recent error logs
   docker logs scgraph-hub-app | grep ERROR | tail -50
   
   # Error distribution by endpoint
   curl 'http://localhost:9090/api/v1/query?query=rate(scgraph_api_requests_total{status=~"5.."}[5m]) by (endpoint)'
   ```

2. **Check Database Performance**
   ```bash
   # Active connections
   docker exec scgraph-postgres psql -U scgraph -d scgraph_hub -c "SELECT count(*) FROM pg_stat_activity;"
   
   # Long running queries
   docker exec scgraph-postgres psql -U scgraph -d scgraph_hub -c "SELECT query, state, query_start FROM pg_stat_activity WHERE state = 'active' AND query_start < now() - interval '1 minute';"
   ```

3. **Check External Dependencies**
   ```bash
   # S3/MinIO connectivity
   curl -I http://localhost:9000/minio/health/live
   
   # Redis performance
   docker exec scgraph-redis redis-cli info stats
   ```

#### Resolution Steps

1. **Identify Root Cause**
   - Database slow queries → optimize queries or add indexes
   - External service timeouts → implement circuit breakers
   - Memory leaks → restart service and investigate

2. **Immediate Mitigation**
   ```bash
   # Restart application to clear memory issues
   docker-compose restart scgraph-hub
   
   # Scale up if capacity issue
   docker-compose up -d --scale scgraph-hub=2
   
   # Enable rate limiting
   # Update configuration to limit requests
   ```

3. **Monitor Recovery**
   ```bash
   # Watch error rate
   watch 'curl -s "http://localhost:9090/api/v1/query?query=rate(scgraph_api_requests_total{status=~\"5..\"}[5m])" | jq .data.result[].value[1]'
   ```

### 3. High Response Time (SEV-2)

#### Symptoms
- 95th percentile response time > 1 second
- Users reporting slow performance
- Timeout errors increasing

#### Investigation Steps

1. **Check Application Performance**
   ```bash
   # Current response times
   curl 'http://localhost:9090/api/v1/query?query=histogram_quantile(0.95, rate(scgraph_api_request_duration_seconds_bucket[5m]))'
   
   # CPU usage
   docker stats scgraph-hub-app --no-stream
   ```

2. **Check Database Performance**
   ```bash
   # Query performance
   docker exec scgraph-postgres psql -U scgraph -d scgraph_hub -c "SELECT query, calls, total_time, mean_time FROM pg_stat_statements ORDER BY total_time DESC LIMIT 10;"
   
   # Connection pool status
   docker logs scgraph-hub-app | grep "pool"
   ```

3. **Check I/O Performance**
   ```bash
   # Disk I/O
   iostat -x 1 5
   
   # Network I/O
   iftop -t -s 10
   ```

#### Resolution Steps

1. **Optimize Database**
   ```sql
   -- Kill long-running queries
   SELECT pg_terminate_backend(pid) FROM pg_stat_activity 
   WHERE state = 'active' AND query_start < now() - interval '5 minutes';
   
   -- Update statistics
   ANALYZE;
   
   -- Reindex if needed
   REINDEX DATABASE scgraph_hub;
   ```

2. **Scale Resources**
   ```bash
   # Add more application instances
   docker-compose up -d --scale scgraph-hub=3
   
   # Increase resource limits
   # Edit docker-compose.yml to increase memory/CPU limits
   ```

3. **Enable Caching**
   ```bash
   # Restart with increased cache settings
   docker-compose restart redis
   
   # Check cache hit rate
   docker exec scgraph-redis redis-cli info stats | grep hit
   ```

### 4. Database Connection Issues (SEV-1)

#### Symptoms
- "Connection refused" errors
- Database health check failing
- Application unable to start

#### Investigation Steps

1. **Check Database Status**
   ```bash
   # Container status
   docker ps | grep postgres
   
   # Database logs
   docker logs scgraph-postgres --tail 50
   
   # Connection attempt
   docker exec scgraph-postgres pg_isready -U scgraph -d scgraph_hub
   ```

2. **Check Resources**
   ```bash
   # Disk space (PostgreSQL needs disk space)
   docker exec scgraph-postgres df -h
   
   # Memory usage
   docker stats scgraph-postgres --no-stream
   ```

#### Resolution Steps

1. **Restart Database**
   ```bash
   # Graceful restart
   docker-compose restart postgres
   
   # If that fails, force restart
   docker-compose stop postgres
   docker-compose up -d postgres
   
   # Wait for startup
   until docker exec scgraph-postgres pg_isready -U scgraph -d scgraph_hub; do sleep 1; done
   ```

2. **Check and Fix Corruption**
   ```bash
   # Check database integrity
   docker exec scgraph-postgres pg_dump -U scgraph scgraph_hub > /dev/null
   
   # If corruption detected, restore from backup
   docker exec -i scgraph-postgres psql -U scgraph -d scgraph_hub < latest_backup.sql
   ```

3. **Clear Connection Pool**
   ```bash
   # Restart application to clear connection pool
   docker-compose restart scgraph-hub
   ```

### 5. Out of Memory (SEV-2)

#### Symptoms
- Container killed by OOM killer
- Memory usage > 90%
- Slow performance and timeouts

#### Investigation Steps

1. **Check Memory Usage**
   ```bash
   # Current memory usage
   docker stats --no-stream
   
   # System memory
   free -h
   
   # Check for memory leaks in logs
   docker logs scgraph-hub-app | grep -i memory
   ```

2. **Identify Memory Consumer**
   ```bash
   # Top processes by memory
   docker exec scgraph-hub-app top -o %MEM
   
   # Python memory profiling (if available)
   docker exec scgraph-hub-app python -c "import psutil; print(psutil.virtual_memory())"
   ```

#### Resolution Steps

1. **Immediate Relief**
   ```bash
   # Restart service to free memory
   docker-compose restart scgraph-hub
   
   # Clear caches
   docker exec scgraph-redis redis-cli FLUSHALL
   
   # Garbage collection (if Python)
   docker exec scgraph-hub-app python -c "import gc; gc.collect()"
   ```

2. **Scale Resources**
   ```bash
   # Increase memory limits
   # Edit docker-compose.yml to increase memory limit
   docker-compose up -d
   
   # Scale horizontally
   docker-compose up -d --scale scgraph-hub=2
   ```

3. **Investigate Memory Leak**
   ```bash
   # Enable memory profiling
   docker exec scgraph-hub-app python -m memory_profiler app.py
   
   # Monitor memory over time
   while true; do docker stats --no-stream | grep scgraph-hub; sleep 60; done
   ```

## Recovery Procedures

### Database Recovery

1. **From Backup**
   ```bash
   # Stop application
   docker-compose stop scgraph-hub
   
   # Drop and recreate database
   docker exec scgraph-postgres psql -U scgraph -c "DROP DATABASE scgraph_hub;"
   docker exec scgraph-postgres psql -U scgraph -c "CREATE DATABASE scgraph_hub;"
   
   # Restore from backup
   docker exec -i scgraph-postgres psql -U scgraph scgraph_hub < backup.sql
   
   # Start application
   docker-compose start scgraph-hub
   ```

2. **Point-in-Time Recovery**
   ```bash
   # If WAL archiving is enabled
   docker exec scgraph-postgres pg_basebackup -U scgraph -D /backup -Ft -z -P
   
   # Restore to specific time
   # (Requires PostgreSQL configuration for PITR)
   ```

### Data Recovery

1. **File System Recovery**
   ```bash
   # Check file system integrity
   docker exec scgraph-hub-app fsck /app/data
   
   # Restore from backup
   rsync -av /backup/data/ /app/data/
   ```

2. **S3/Object Storage Recovery**
   ```bash
   # List available backups
   aws s3 ls s3://scgraph-hub-backups/
   
   # Restore specific backup
   aws s3 sync s3://scgraph-hub-backups/2025-01-01/ /app/data/
   ```

## Communication Procedures

### Internal Communication

1. **Incident Declaration**
   ```
   INCIDENT DECLARED: [SEV-X] Brief Description
   
   Impact: [Description of user impact]
   Services Affected: [List of affected services]
   Incident Commander: [Name]
   ETA for Update: [Time]
   
   War Room: [Link to video call]
   Status Page: [Link to status updates]
   ```

2. **Status Updates (Every 15-30 minutes)**
   ```
   INCIDENT UPDATE: [Timestamp]
   
   Current Status: [What's happening now]
   Actions Taken: [What we've done]
   Next Steps: [What we're doing next]
   ETA for Resolution: [Updated estimate]
   ```

3. **Resolution Notice**
   ```
   INCIDENT RESOLVED: [Timestamp]
   
   Resolution: [What fixed the issue]
   Root Cause: [Brief initial assessment]
   Post-Mortem: [Date/time for detailed review]
   ```

### External Communication

1. **Status Page Updates**
   - Update status.scgraphhub.org
   - Include impact and ETA
   - Regular updates every 30 minutes

2. **User Notifications**
   - Email notifications for SEV-1/SEV-2
   - In-app notifications if possible
   - Social media for major outages

## Post-Incident Activities

### Immediate (< 24 hours)

1. **System Stability Check**
   ```bash
   # Monitor key metrics
   ./scripts/health-check-detailed.sh
   
   # Check for recurring issues
   docker logs scgraph-hub-app | grep ERROR | tail -100
   ```

2. **Data Integrity Verification**
   ```bash
   # Database consistency check
   docker exec scgraph-postgres pg_dump -U scgraph scgraph_hub > /dev/null
   
   # File system check
   docker exec scgraph-hub-app find /app/data -type f -exec md5sum {} \; > integrity.check
   ```

### Short Term (< 1 week)

1. **Post-Mortem Meeting**
   - Schedule within 48 hours
   - Include all involved parties
   - Focus on timeline, root cause, and improvements

2. **Documentation Updates**
   - Update runbooks with lessons learned
   - Add new monitoring if needed
   - Improve alerting thresholds

### Long Term (< 1 month)

1. **Process Improvements**
   - Implement preventive measures
   - Automate manual steps
   - Improve monitoring coverage

2. **Infrastructure Hardening**
   - Add redundancy where needed
   - Improve backup/recovery procedures
   - Enhance security measures

## Contact Information

### Escalation Chain

1. **On-Call Engineer**: [Phone/Slack]
2. **Engineering Manager**: [Phone/Slack]
3. **VP Engineering**: [Phone/Slack]
4. **CTO**: [Phone/Slack]

### External Contacts

1. **Cloud Provider Support**: [Account details]
2. **Database Support**: [Contract details]
3. **Security Team**: [Contact information]

## Tools and Resources

### Monitoring Links
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000
- AlertManager: http://localhost:9093
- Jaeger: http://localhost:16686

### Documentation Links
- Architecture Diagram: [Link]
- Network Diagram: [Link]
- Database Schema: [Link]
- API Documentation: [Link]

### Emergency Procedures
- Break Glass Access: [Procedure]
- Emergency Contacts: [List]
- Vendor Support: [Contact info]