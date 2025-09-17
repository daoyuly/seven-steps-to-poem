#!/bin/bash
# Start all microservices for Seven Steps to Poem

set -e

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

# Check if Docker Compose is available
check_docker_compose() {
    if command -v docker-compose &> /dev/null; then
        DOCKER_COMPOSE="docker-compose"
    elif docker compose version &> /dev/null; then
        DOCKER_COMPOSE="docker compose"
    else
        print_error "Docker Compose is not available."
        exit 1
    fi
}

# Check environment
check_environment() {
    print_status "Checking environment..."
    
    if [ ! -f .env ]; then
        print_warning ".env file not found. Creating from .env.example..."
        cp .env.example .env
        print_warning "Please edit .env file and add your OpenAI API key!"
        print_warning "Set OPENAI_API_KEY in .env before continuing."
        read -p "Press Enter to continue after setting up .env file..."
    fi
    
    # Source environment variables
    if [ -f .env ]; then
        export $(cat .env | grep -v '#' | xargs)
    fi
    
    if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your-openai-api-key-here" ]; then
        print_error "OPENAI_API_KEY is not set in .env file!"
        print_error "Please set a valid OpenAI API key in the .env file."
        exit 1
    fi
    
    print_success "Environment configuration validated"
}

# Build and start services
start_services() {
    print_status "Building and starting microservices..."
    
    # Build all service images
    print_status "Building Docker images..."
    $DOCKER_COMPOSE -f docker-compose.services.yml build
    
    if [ $? -ne 0 ]; then
        print_error "Failed to build Docker images"
        exit 1
    fi
    
    # Start infrastructure services first
    print_status "Starting infrastructure services..."
    $DOCKER_COMPOSE -f docker-compose.services.yml up -d postgres redis neo4j
    
    # Wait for infrastructure services to be ready
    print_status "Waiting for infrastructure services to be ready..."
    sleep 15
    
    # Check if infrastructure services are healthy
    print_status "Checking infrastructure services health..."
    
    for service in postgres redis neo4j; do
        if $DOCKER_COMPOSE -f docker-compose.services.yml ps $service | grep -q "Up (healthy)"; then
            print_success "$service is ready"
        else
            print_warning "$service may not be fully ready yet"
        fi
    done
    
    # Start application services
    print_status "Starting application services..."
    $DOCKER_COMPOSE -f docker-compose.services.yml up -d problem-framer issue-tree orchestrator
    
    # Wait for application services
    print_status "Waiting for application services to start..."
    sleep 20
    
    # Start API gateway
    print_status "Starting API gateway..."
    $DOCKER_COMPOSE -f docker-compose.services.yml up -d api-gateway
    
    print_success "All services started!"
}

# Check service health
check_service_health() {
    print_status "Checking service health..."
    
    services=("orchestrator:8000" "problem-framer:8001" "issue-tree:8002")
    
    for service in "${services[@]}"; do
        IFS=':' read -ra ADDR <<< "$service"
        service_name="${ADDR[0]}"
        port="${ADDR[1]}"
        
        if curl -s -f http://localhost:$port/health > /dev/null; then
            print_success "$service_name service is healthy"
        else
            print_error "$service_name service is not responding"
        fi
    done
}

# Show service information
show_service_info() {
    print_success "🎉 Seven Steps to Poem microservices are running!"
    echo
    print_status "Service URLs:"
    echo "  🌐 API Gateway: http://localhost"
    echo "  📖 API Documentation: http://localhost/docs"
    echo "  🏥 Health Check: http://localhost/health"
    echo
    echo "  🔧 Orchestrator: http://localhost:8000"
    echo "  🧠 Problem Framer: http://localhost:8001"
    echo "  🌳 Issue Tree: http://localhost:8002"
    echo
    echo "  📊 Infrastructure:"
    echo "    - PostgreSQL: localhost:5432"
    echo "    - Redis: localhost:6379"  
    echo "    - Neo4j: http://localhost:7474"
    echo "    - Prometheus: http://localhost:9090"
    echo "    - Grafana: http://localhost:3000"
    echo
    print_status "Useful commands:"
    echo "  📋 View logs: $DOCKER_COMPOSE -f docker-compose.services.yml logs -f [service]"
    echo "  🔄 Restart service: $DOCKER_COMPOSE -f docker-compose.services.yml restart [service]"
    echo "  🛑 Stop services: $DOCKER_COMPOSE -f docker-compose.services.yml down"
    echo "  🗑️  Reset data: $DOCKER_COMPOSE -f docker-compose.services.yml down -v"
    echo
    print_status "Test the API:"
    echo '  curl -X POST "http://localhost/v1/workflows" \\'
    echo '    -H "Content-Type: application/json" \\'
    echo '    -d '"'"'{"problem_id":"test-123","user_id":"user-1","organization_id":"org-1"}'"'"
}

# Main execution
main() {
    echo "================================================="
    echo "Seven Steps to Poem - Microservices Startup"
    echo "================================================="
    
    check_docker_compose
    check_environment
    start_services
    
    # Wait a bit for services to fully start
    sleep 10
    
    check_service_health
    show_service_info
}

# Handle script arguments
case "${1:-start}" in
    start)
        main
        ;;
    stop)
        print_status "Stopping all services..."
        $DOCKER_COMPOSE -f docker-compose.services.yml down
        print_success "All services stopped"
        ;;
    restart)
        print_status "Restarting all services..."
        $DOCKER_COMPOSE -f docker-compose.services.yml restart
        print_success "All services restarted"
        ;;
    logs)
        service_name=${2:-""}
        if [ -z "$service_name" ]; then
            $DOCKER_COMPOSE -f docker-compose.services.yml logs -f
        else
            $DOCKER_COMPOSE -f docker-compose.services.yml logs -f $service_name
        fi
        ;;
    status)
        $DOCKER_COMPOSE -f docker-compose.services.yml ps
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|logs [service]|status}"
        exit 1
        ;;
esac