#!/bin/bash
# Development environment setup script for Seven Steps to Poem

set -e

echo "🚀 Setting up Seven Steps to Poem development environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed and running
check_docker() {
    print_status "Checking Docker installation..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running. Please start Docker."
        exit 1
    fi
    
    print_success "Docker is installed and running"
}

# Check if Docker Compose is available
check_docker_compose() {
    print_status "Checking Docker Compose..."
    
    if command -v docker-compose &> /dev/null; then
        DOCKER_COMPOSE="docker-compose"
    elif docker compose version &> /dev/null; then
        DOCKER_COMPOSE="docker compose"
    else
        print_error "Docker Compose is not available. Please install Docker Compose."
        exit 1
    fi
    
    print_success "Docker Compose is available"
}

# Check if Poetry is installed
check_poetry() {
    print_status "Checking Poetry installation..."
    
    if ! command -v poetry &> /dev/null; then
        print_warning "Poetry is not installed. Installing Poetry..."
        curl -sSL https://install.python-poetry.org | python3 -
        export PATH="$HOME/.local/bin:$PATH"
        
        if ! command -v poetry &> /dev/null; then
            print_error "Poetry installation failed. Please install Poetry manually."
            exit 1
        fi
    fi
    
    print_success "Poetry is available"
}

# Setup environment file
setup_env() {
    print_status "Setting up environment configuration..."
    
    if [ ! -f .env ]; then
        cp .env.example .env
        print_success "Created .env file from .env.example"
        print_warning "Please edit .env file and add your OpenAI API key!"
    else
        print_warning ".env file already exists. Skipping..."
    fi
}

# Install Python dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    
    poetry install
    print_success "Python dependencies installed"
}

# Start infrastructure services
start_infrastructure() {
    print_status "Starting infrastructure services..."
    
    $DOCKER_COMPOSE -f docker-compose.dev.yml up -d postgres redis neo4j minio
    
    print_status "Waiting for services to be ready..."
    sleep 10
    
    # Check if services are running
    if $DOCKER_COMPOSE -f docker-compose.dev.yml ps | grep -q "Up"; then
        print_success "Infrastructure services started successfully"
    else
        print_error "Some services failed to start. Check with: $DOCKER_COMPOSE -f docker-compose.dev.yml logs"
        exit 1
    fi
}

# Run database migrations
run_migrations() {
    print_status "Running database migrations..."
    
    # Generate initial migration if it doesn't exist
    if [ ! -f "migrations/versions/001_initial_schema.py" ]; then
        poetry run alembic revision --autogenerate -m "Initial schema"
    fi
    
    # Run migrations
    poetry run alembic upgrade head
    
    print_success "Database migrations completed"
}

# Setup monitoring (optional)
setup_monitoring() {
    read -p "Do you want to set up monitoring services (Prometheus, Grafana)? [y/N]: " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Setting up monitoring services..."
        
        # Create monitoring directory
        mkdir -p monitoring
        
        # Create basic Prometheus config if it doesn't exist
        if [ ! -f "monitoring/prometheus.yml" ]; then
            cat > monitoring/prometheus.yml << EOF
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'seven-steps-api'
    static_configs:
      - targets: ['host.docker.internal:8090']
    scrape_interval: 5s
    metrics_path: /metrics
EOF
        fi
        
        $DOCKER_COMPOSE -f docker-compose.dev.yml up -d prometheus grafana
        
        print_success "Monitoring services started"
        print_status "Grafana URL: http://localhost:3000 (admin/admin)"
        print_status "Prometheus URL: http://localhost:9090"
    fi
}

# Create a simple test script
create_test_script() {
    print_status "Creating test script..."
    
    cat > scripts/test-api.sh << 'EOF'
#!/bin/bash
# Simple API test script

API_URL="http://localhost:8000"

echo "Testing Seven Steps to Poem API..."

# Test health endpoint
echo "1. Testing health endpoint..."
curl -s "$API_URL/v1/health" | python -m json.tool

echo -e "\n2. Testing version endpoint..."
curl -s "$API_URL/v1/version" | python -m json.tool

echo -e "\n3. API is ready for testing!"
echo "   - API Documentation: $API_URL/docs"
echo "   - Health Check: $API_URL/v1/health" 
echo "   - Metrics: $API_URL/v1/metrics"
EOF

    chmod +x scripts/test-api.sh
    print_success "Test script created at scripts/test-api.sh"
}

# Display final information
show_final_info() {
    print_success "🎉 Development environment setup complete!"
    echo
    print_status "Services running:"
    echo "  - PostgreSQL: localhost:5432"
    echo "  - Redis: localhost:6379"
    echo "  - Neo4j: localhost:7474 (HTTP), localhost:7687 (Bolt)"
    echo "  - MinIO: localhost:9000 (API), localhost:9001 (Console)"
    echo
    print_status "Next steps:"
    echo "  1. Edit .env file and add your OpenAI API key"
    echo "  2. Start the API server: poetry run python -m seven_steps.api.main"
    echo "  3. Visit http://localhost:8000/docs for API documentation"
    echo "  4. Run tests: ./scripts/test-api.sh"
    echo
    print_status "Useful commands:"
    echo "  - View logs: $DOCKER_COMPOSE -f docker-compose.dev.yml logs -f [service]"
    echo "  - Stop services: $DOCKER_COMPOSE -f docker-compose.dev.yml down"
    echo "  - Reset data: $DOCKER_COMPOSE -f docker-compose.dev.yml down -v"
}

# Main execution
main() {
    echo "================================================="
    echo "Seven Steps to Poem - Development Setup"
    echo "================================================="
    
    check_docker
    check_docker_compose
    check_poetry
    setup_env
    install_dependencies
    start_infrastructure
    run_migrations
    setup_monitoring
    create_test_script
    show_final_info
}

# Run main function
main