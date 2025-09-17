# Deployment & Infrastructure Guide

## Overview

This guide provides comprehensive instructions for deploying the Seven Steps to Poem AI Agent system across development, staging, and production environments. The system is designed as cloud-native microservices with Kubernetes orchestration.

## Infrastructure Requirements

### Compute Resources

#### Development Environment
```yaml
minimum_requirements:
  nodes: 1
  cpu_cores: 8
  memory: 32GB
  storage: 200GB SSD
  
services:
  docker: ">=20.10"
  kubernetes: ">=1.24" (kind/minikube)
  helm: ">=3.8"
```

#### Staging Environment
```yaml
recommended_specs:
  nodes: 3
  cpu_cores_per_node: 8
  memory_per_node: 32GB
  storage: 500GB SSD per node
  network: 10Gbps
  
load_balancer: managed (AWS ALB/GCP GLB/Azure ALB)
dns: managed DNS service
monitoring: dedicated monitoring stack
```

#### Production Environment
```yaml
high_availability:
  nodes: 6+ (across 3 AZs)
  cpu_cores_per_node: 16
  memory_per_node: 64GB
  storage: 1TB NVMe SSD per node
  network: 25Gbps with redundancy
  
databases:
  postgresql: multi-AZ with read replicas
  redis: clustered with auto-failover
  neo4j: clustered deployment
  
external_services:
  vector_db: Pinecone Pro or Weaviate Cloud
  object_storage: AWS S3/GCP GCS/Azure Blob (hot+cold tiers)
  cdn: CloudFlare or AWS CloudFront
```

## Cloud Provider Configurations

### AWS Deployment

#### EKS Cluster Setup
```yaml
# cluster-config.yaml
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: seven-steps-prod
  region: us-west-2
  version: "1.24"

nodeGroups:
  - name: worker-nodes
    instanceType: m5.2xlarge
    desiredCapacity: 6
    minSize: 3
    maxSize: 12
    volumeSize: 100
    ssh:
      allow: true
    iam:
      withAddonPolicies:
        autoScaler: true
        awsLoadBalancerController: true
        ebs: true
        efs: true
        certManager: true

addons:
  - name: vpc-cni
  - name: coredns
  - name: kube-proxy
  - name: aws-ebs-csi-driver
```

#### RDS Configuration
```yaml
# terraform/rds.tf
resource "aws_db_instance" "postgresql" {
  identifier = "seven-steps-postgres"
  
  engine         = "postgres"
  engine_version = "14.6"
  instance_class = "db.r6g.2xlarge"
  
  allocated_storage     = 200
  max_allocated_storage = 1000
  storage_type         = "gp3"
  storage_encrypted    = true
  
  db_name  = "sevensteps"
  username = var.db_username
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  multi_az               = true
  publicly_accessible    = false
  
  performance_insights_enabled = true
  monitoring_interval         = 60
  
  tags = local.common_tags
}
```

### GCP Deployment

#### GKE Cluster
```yaml
# gke-cluster.yaml
resource "google_container_cluster" "seven_steps" {
  name     = "seven-steps-prod"
  location = "us-central1"
  
  remove_default_node_pool = true
  initial_node_count       = 1
  
  network    = google_compute_network.vpc.name
  subnetwork = google_compute_subnetwork.subnet.name
  
  master_auth {
    client_certificate_config {
      issue_client_certificate = false
    }
  }
  
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }
  
  addons_config {
    http_load_balancing {
      disabled = false
    }
    horizontal_pod_autoscaling {
      disabled = false
    }
  }
}

resource "google_container_node_pool" "workers" {
  name       = "worker-pool"
  location   = "us-central1"
  cluster    = google_container_cluster.seven_steps.name
  node_count = 2
  
  node_config {
    preemptible  = false
    machine_type = "e2-standard-8"
    
    service_account = google_service_account.gke.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
  }
  
  autoscaling {
    min_node_count = 3
    max_node_count = 12
  }
}
```

## Container Images & Registry

### Docker Registry Setup
```bash
# AWS ECR
aws ecr create-repository --repository-name seven-steps/problem-framer
aws ecr create-repository --repository-name seven-steps/issue-tree
aws ecr create-repository --repository-name seven-steps/prioritization
# ... create repositories for all services

# Build and push images
./scripts/build-and-push.sh --registry 123456789.dkr.ecr.us-west-2.amazonaws.com
```

### Build Pipeline (GitHub Actions)
```yaml
# .github/workflows/build-deploy.yml
name: Build and Deploy
on:
  push:
    branches: [main, staging]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        service: 
          - problem-framer
          - issue-tree
          - prioritization
          - planner
          - analysis
          - synthesizer
          - presentation
          - orchestrator
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-west-2
    
    - name: Login to ECR
      id: login-ecr
      uses: aws-actions/amazon-ecr-login@v1
    
    - name: Build and push Docker image
      env:
        ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
        IMAGE_TAG: ${{ github.sha }}
      run: |
        cd services/${{ matrix.service }}
        docker build -t $ECR_REGISTRY/seven-steps/${{ matrix.service }}:$IMAGE_TAG .
        docker push $ECR_REGISTRY/seven-steps/${{ matrix.service }}:$IMAGE_TAG
        
        # Also tag as latest for main branch
        if [[ "${{ github.ref }}" == "refs/heads/main" ]]; then
          docker tag $ECR_REGISTRY/seven-steps/${{ matrix.service }}:$IMAGE_TAG \
                     $ECR_REGISTRY/seven-steps/${{ matrix.service }}:latest
          docker push $ECR_REGISTRY/seven-steps/${{ matrix.service }}:latest
        fi
```

## Kubernetes Deployment

### Namespace Setup
```yaml
# k8s/namespaces.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: seven-steps-prod
  labels:
    name: seven-steps-prod
    environment: production
---
apiVersion: v1
kind: Namespace
metadata:
  name: seven-steps-staging
  labels:
    name: seven-steps-staging
    environment: staging
```

### ConfigMaps and Secrets
```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: seven-steps-config
  namespace: seven-steps-prod
data:
  DATABASE_HOST: "postgresql.seven-steps-prod.svc.cluster.local"
  REDIS_HOST: "redis-cluster.seven-steps-prod.svc.cluster.local"
  KAFKA_BROKERS: "kafka-broker-0:9092,kafka-broker-1:9092,kafka-broker-2:9092"
  LLM_PROVIDER: "openai"
  LLM_MODEL: "gpt-4-turbo"
  VECTOR_DB_URL: "https://seven-steps-index.pinecone.io"
  OBJECT_STORAGE_ENDPOINT: "https://s3.amazonaws.com"
  
---
apiVersion: v1
kind: Secret
metadata:
  name: seven-steps-secrets
  namespace: seven-steps-prod
type: Opaque
data:
  DATABASE_PASSWORD: <base64-encoded-password>
  REDIS_PASSWORD: <base64-encoded-password>
  OPENAI_API_KEY: <base64-encoded-key>
  JWT_SECRET: <base64-encoded-secret>
  PINECONE_API_KEY: <base64-encoded-key>
  AWS_ACCESS_KEY_ID: <base64-encoded-key>
  AWS_SECRET_ACCESS_KEY: <base64-encoded-secret>
```

### Service Deployments
```yaml
# k8s/deployments/problem-framer.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: problem-framer
  namespace: seven-steps-prod
spec:
  replicas: 3
  selector:
    matchLabels:
      app: problem-framer
  template:
    metadata:
      labels:
        app: problem-framer
    spec:
      containers:
      - name: problem-framer
        image: seven-steps/problem-framer:latest
        ports:
        - containerPort: 8001
        env:
        - name: DATABASE_URL
          value: "postgresql://$(DATABASE_USER):$(DATABASE_PASSWORD)@$(DATABASE_HOST):5432/sevensteps"
        - name: DATABASE_USER
          value: "sevensteps_user"
        - name: DATABASE_PASSWORD
          valueFrom:
            secretKeyRef:
              name: seven-steps-secrets
              key: DATABASE_PASSWORD
        - name: DATABASE_HOST
          valueFrom:
            configMapKeyRef:
              name: seven-steps-config
              key: DATABASE_HOST
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: seven-steps-secrets
              key: OPENAI_API_KEY
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 1000m
            memory: 2Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8001
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
        readinessProbe:
          httpGet:
            path: /ready
            port: 8001
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3

---
apiVersion: v1
kind: Service
metadata:
  name: problem-framer
  namespace: seven-steps-prod
spec:
  selector:
    app: problem-framer
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8001
  type: ClusterIP
```

## Database Setup

### PostgreSQL Initialization
```sql
-- init/001_create_database.sql
CREATE DATABASE sevensteps;
CREATE USER sevensteps_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE sevensteps TO sevensteps_user;

-- init/002_extensions.sql
\c sevensteps
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- init/003_schema.sql
-- Apply schema from DATA_MODELS.md
```

### Redis Cluster
```yaml
# k8s/redis-cluster.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: redis-cluster-config
data:
  redis.conf: |
    cluster-enabled yes
    cluster-config-file nodes.conf
    cluster-node-timeout 5000
    appendonly yes
    protected-mode no
    bind 0.0.0.0
    port 6379

---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis-cluster
spec:
  serviceName: redis-cluster
  replicas: 6
  selector:
    matchLabels:
      app: redis-cluster
  template:
    metadata:
      labels:
        app: redis-cluster
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        - containerPort: 16379
        volumeMounts:
        - name: conf
          mountPath: /usr/local/etc/redis
        - name: data
          mountPath: /data
        command:
        - redis-server
        - /usr/local/etc/redis/redis.conf
      volumes:
      - name: conf
        configMap:
          name: redis-cluster-config
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 50Gi
```

### Neo4j Setup
```yaml
# k8s/neo4j.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: neo4j
spec:
  serviceName: neo4j
  replicas: 3
  selector:
    matchLabels:
      app: neo4j
  template:
    metadata:
      labels:
        app: neo4j
    spec:
      containers:
      - name: neo4j
        image: neo4j:5.0-enterprise
        ports:
        - containerPort: 7474  # HTTP
        - containerPort: 7687  # Bolt
        - containerPort: 6362  # Backup
        - containerPort: 5000  # Discovery
        - containerPort: 6000  # Transaction
        - containerPort: 7000  # Raft
        env:
        - name: NEO4J_AUTH
          value: "neo4j/your-secure-password"
        - name: NEO4J_ACCEPT_LICENSE_AGREEMENT
          value: "yes"
        - name: NEO4J_dbms_mode
          value: "CORE"
        - name: NEO4J_causal__clustering_initial__discovery__members
          value: "neo4j-0.neo4j:5000,neo4j-1.neo4j:5000,neo4j-2.neo4j:5000"
        volumeMounts:
        - name: neo4j-data
          mountPath: /data
        - name: neo4j-logs
          mountPath: /logs
  volumeClaimTemplates:
  - metadata:
      name: neo4j-data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi
  - metadata:
      name: neo4j-logs
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 10Gi
```

## Load Balancing & Ingress

### Ingress Controller (NGINX)
```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: seven-steps-ingress
  namespace: seven-steps-prod
  annotations:
    kubernetes.io/ingress.class: "nginx"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
  - hosts:
    - api.sevenstepstopoem.com
    secretName: seven-steps-tls
  rules:
  - host: api.sevenstepstopoem.com
    http:
      paths:
      - path: /v1
        pathType: Prefix
        backend:
          service:
            name: orchestrator
            port:
              number: 80
      - path: /health
        pathType: Prefix
        backend:
          service:
            name: orchestrator
            port:
              number: 80
```

### Application Load Balancer (AWS)
```yaml
# k8s/alb-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: seven-steps-alb
  annotations:
    kubernetes.io/ingress.class: alb
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/load-balancer-attributes: routing.http2.enabled=true
    alb.ingress.kubernetes.io/certificate-arn: arn:aws:acm:us-west-2:123456789:certificate/your-cert-id
spec:
  rules:
  - host: api.sevenstepstopoem.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: orchestrator
            port:
              number: 80
```

## Monitoring & Logging

### Prometheus Setup
```yaml
# k8s/prometheus.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    scrape_configs:
    - job_name: 'seven-steps-services'
      kubernetes_sd_configs:
      - role: endpoints
      relabel_configs:
      - source_labels: [__meta_kubernetes_service_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_service_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      containers:
      - name: prometheus
        image: prom/prometheus:latest
        ports:
        - containerPort: 9090
        volumeMounts:
        - name: config-volume
          mountPath: /etc/prometheus
        - name: storage-volume
          mountPath: /prometheus
      volumes:
      - name: config-volume
        configMap:
          name: prometheus-config
      - name: storage-volume
        emptyDir: {}
```

### ELK Stack Deployment
```yaml
# k8s/elasticsearch.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: elasticsearch
spec:
  serviceName: elasticsearch
  replicas: 3
  selector:
    matchLabels:
      app: elasticsearch
  template:
    metadata:
      labels:
        app: elasticsearch
    spec:
      containers:
      - name: elasticsearch
        image: docker.elastic.co/elasticsearch/elasticsearch:8.5.0
        ports:
        - containerPort: 9200
        - containerPort: 9300
        env:
        - name: discovery.type
          value: "zen"
        - name: cluster.name
          value: "seven-steps-logs"
        - name: bootstrap.memory_lock
          value: "true"
        - name: ES_JAVA_OPTS
          value: "-Xms2g -Xmx2g"
        resources:
          requests:
            memory: 4Gi
            cpu: 1000m
          limits:
            memory: 4Gi
            cpu: 2000m
        volumeMounts:
        - name: elasticsearch-data
          mountPath: /usr/share/elasticsearch/data
  volumeClaimTemplates:
  - metadata:
      name: elasticsearch-data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 200Gi
```

## Security Configuration

### Network Policies
```yaml
# k8s/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: seven-steps-network-policy
  namespace: seven-steps-prod
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: seven-steps-prod
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: seven-steps-prod
  - to: []  # Allow external API calls (LLM providers)
    ports:
    - protocol: TCP
      port: 443
    - protocol: TCP
      port: 80
```

### Pod Security Standards
```yaml
# k8s/pod-security.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: seven-steps-prod
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
```

### RBAC Configuration
```yaml
# k8s/rbac.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: seven-steps-prod
  name: seven-steps-role
rules:
- apiGroups: [""]
  resources: ["pods", "services", "configmaps", "secrets"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: seven-steps-binding
  namespace: seven-steps-prod
subjects:
- kind: ServiceAccount
  name: seven-steps-sa
  namespace: seven-steps-prod
roleRef:
  kind: Role
  name: seven-steps-role
  apiGroup: rbac.authorization.k8s.io
```

## Deployment Scripts

### Automated Deployment
```bash
#!/bin/bash
# scripts/deploy.sh

set -e

ENVIRONMENT=${1:-staging}
NAMESPACE="seven-steps-${ENVIRONMENT}"

echo "Deploying Seven Steps to Poem - Environment: ${ENVIRONMENT}"

# Apply namespace and RBAC
kubectl apply -f k8s/namespaces.yaml
kubectl apply -f k8s/rbac.yaml

# Apply secrets (from sealed secrets or external secrets operator)
kubectl apply -f k8s/secrets/${ENVIRONMENT}-secrets.yaml

# Apply ConfigMaps
envsubst < k8s/configmap-template.yaml | kubectl apply -f -

# Deploy databases
kubectl apply -f k8s/postgresql.yaml
kubectl apply -f k8s/redis-cluster.yaml
kubectl apply -f k8s/neo4j.yaml

# Wait for databases to be ready
kubectl wait --for=condition=ready pod -l app=postgresql -n ${NAMESPACE} --timeout=300s
kubectl wait --for=condition=ready pod -l app=redis-cluster -n ${NAMESPACE} --timeout=300s

# Run database migrations
kubectl run migration --image=seven-steps/migrations:latest --restart=Never -n ${NAMESPACE} \
  --overrides='{"spec":{"containers":[{"name":"migration","image":"seven-steps/migrations:latest","env":[{"name":"DATABASE_URL","valueFrom":{"secretKeyRef":{"name":"seven-steps-secrets","key":"DATABASE_URL"}}}]}]}}'

kubectl wait --for=condition=complete job/migration -n ${NAMESPACE} --timeout=600s

# Deploy application services
for service in problem-framer issue-tree prioritization planner analysis synthesizer presentation orchestrator; do
  echo "Deploying ${service}..."
  envsubst < k8s/deployments/${service}.yaml | kubectl apply -f -
done

# Deploy ingress
kubectl apply -f k8s/ingress.yaml

# Deploy monitoring
kubectl apply -f k8s/prometheus.yaml
kubectl apply -f k8s/grafana.yaml

# Wait for all deployments to be ready
kubectl rollout status deployment/problem-framer -n ${NAMESPACE}
kubectl rollout status deployment/orchestrator -n ${NAMESPACE}

echo "Deployment complete!"
echo "API endpoint: https://api-${ENVIRONMENT}.sevenstepstopoem.com"
echo "Monitoring: https://monitoring-${ENVIRONMENT}.sevenstepstopoem.com"
```

### Health Check Script
```bash
#!/bin/bash
# scripts/health-check.sh

ENVIRONMENT=${1:-staging}
API_URL="https://api-${ENVIRONMENT}.sevenstepstopoem.com"

echo "Running health checks for ${ENVIRONMENT}..."

# API Gateway health
response=$(curl -s -o /dev/null -w "%{http_code}" ${API_URL}/health)
if [ $response -eq 200 ]; then
  echo "✓ API Gateway: Healthy"
else
  echo "✗ API Gateway: Unhealthy (${response})"
  exit 1
fi

# Database connectivity
kubectl exec -n seven-steps-${ENVIRONMENT} deployment/orchestrator -- \
  python -c "import psycopg2; psycopg2.connect(host='postgresql', database='sevensteps', user='sevensteps_user', password='password').close(); print('✓ PostgreSQL: Connected')"

# Redis connectivity  
kubectl exec -n seven-steps-${ENVIRONMENT} deployment/orchestrator -- \
  redis-cli -h redis-cluster ping | grep -q PONG && echo "✓ Redis: Connected"

# Neo4j connectivity
kubectl exec -n seven-steps-${ENVIRONMENT} deployment/orchestrator -- \
  python -c "from neo4j import GraphDatabase; driver = GraphDatabase.driver('bolt://neo4j:7687', auth=('neo4j', 'password')); driver.verify_connectivity(); print('✓ Neo4j: Connected')"

echo "All health checks passed!"
```

## Backup & Disaster Recovery

### Database Backups
```bash
#!/bin/bash
# scripts/backup.sh

DATE=$(date +"%Y%m%d_%H%M%S")
BACKUP_BUCKET="seven-steps-backups"

# PostgreSQL backup
kubectl exec deployment/postgresql -- pg_dump -U sevensteps_user sevensteps | \
  aws s3 cp - s3://${BACKUP_BUCKET}/postgresql/backup_${DATE}.sql

# Neo4j backup
kubectl exec statefulset/neo4j-0 -- neo4j-admin backup \
  --backup-dir=/tmp/backup --name=graph_${DATE}

kubectl cp neo4j-0:/tmp/backup ./neo4j_backup_${DATE}
tar czf neo4j_backup_${DATE}.tar.gz neo4j_backup_${DATE}
aws s3 cp neo4j_backup_${DATE}.tar.gz s3://${BACKUP_BUCKET}/neo4j/

# Application data backup (artifacts, deliverables)
kubectl exec deployment/orchestrator -- \
  aws s3 sync s3://seven-steps-artifacts s3://${BACKUP_BUCKET}/artifacts_${DATE} --storage-class GLACIER
```

### Disaster Recovery Runbook
```markdown
# Disaster Recovery Procedures

## RTO/RPO Targets
- Recovery Time Objective (RTO): 4 hours
- Recovery Point Objective (RPO): 1 hour

## Recovery Steps

### 1. Infrastructure Recovery
1. Provision new Kubernetes cluster in alternate region
2. Restore networking and security groups
3. Deploy monitoring and logging first

### 2. Database Recovery  
1. Restore PostgreSQL from latest backup
2. Restore Neo4j from latest backup
3. Verify data integrity and consistency

### 3. Application Recovery
1. Deploy all services from last known good images
2. Run health checks and integration tests
3. Update DNS to point to new endpoints

### 4. Data Validation
1. Compare critical metrics pre/post recovery
2. Validate end-to-end problem processing
3. Check all integrations (LLM providers, object storage)

### 5. Go-Live Checklist
- [ ] All services responding to health checks
- [ ] Database connections stable
- [ ] External API integrations working
- [ ] Monitoring and alerting active
- [ ] Load balancer health checks passing
- [ ] SSL certificates valid
- [ ] DNS propagation complete
```

This deployment guide provides a comprehensive foundation for running the Seven Steps to Poem system in production with high availability, security, and observability.