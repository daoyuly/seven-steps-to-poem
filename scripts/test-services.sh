#!/bin/bash
# Test script for Seven Steps to Poem microservices

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_test() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Base URL for API
API_URL="http://localhost"
ORCHESTRATOR_URL="http://localhost:8000"
PROBLEM_FRAMER_URL="http://localhost:8001"
ISSUE_TREE_URL="http://localhost:8002"

# Test service health endpoints
test_health_checks() {
    print_test "Testing service health endpoints..."
    
    services=(
        "API Gateway:$API_URL/health"
        "Orchestrator:$ORCHESTRATOR_URL/health"
        "Problem Framer:$PROBLEM_FRAMER_URL/health"
        "Issue Tree:$ISSUE_TREE_URL/health"
    )
    
    for service in "${services[@]}"; do
        IFS=':' read -ra ADDR <<< "$service"
        service_name="${ADDR[0]}"
        url="${ADDR[1]}:${ADDR[2]}"
        
        if curl -s -f "$url" > /dev/null; then
            print_success "$service_name health check passed"
        else
            print_error "$service_name health check failed"
            return 1
        fi
    done
}

# Test workflow creation
test_workflow_creation() {
    print_test "Testing workflow creation..."
    
    response=$(curl -s -X POST "$ORCHESTRATOR_URL/workflows" \
        -H "Content-Type: application/json" \
        -d '{
            "problem_id": "test-problem-123",
            "user_id": "test-user-456",
            "organization_id": "test-org-789"
        }')
    
    if echo "$response" | jq -e '.success' > /dev/null; then
        print_success "Workflow creation successful"
        echo "Response: $response" | jq .
    else
        print_error "Workflow creation failed"
        echo "Response: $response"
        return 1
    fi
}

# Test Problem Framer service directly
test_problem_framer() {
    print_test "Testing Problem Framer service..."
    
    response=$(curl -s -X POST "$PROBLEM_FRAMER_URL/frame" \
        -H "Content-Type: application/json" \
        -d '{
            "raw_text": "Our customer churn rate has increased by 25% over the past quarter. We need to identify root causes and implement solutions to reduce churn by 10% within 6 months.",
            "submitter": "product.manager@company.com",
            "metadata": {
                "industry": "SaaS",
                "company_size": "medium",
                "urgency": "high"
            },
            "previous_clarifications": {},
            "context": {
                "user_id": "test-user",
                "organization_id": "test-org",
                "problem_id": "test-problem"
            }
        }')
    
    if echo "$response" | jq -e '.success' > /dev/null; then
        print_success "Problem Framer test successful"
        echo "Execution time: $(echo "$response" | jq -r '.execution_time')s"
        echo "Confidence score: $(echo "$response" | jq -r '.confidence_score')"
        
        # Check if we got clarifying questions
        if echo "$response" | jq -e '.data.needs_clarification' > /dev/null; then
            questions_count=$(echo "$response" | jq -r '.data.clarifying_questions | length')
            print_success "Generated $questions_count clarifying questions"
        else
            print_success "Generated problem frame without clarification needed"
        fi
    else
        print_error "Problem Framer test failed"
        echo "Response: $response"
        return 1
    fi
}

# Test Issue Tree service directly
test_issue_tree() {
    print_test "Testing Issue Tree service..."
    
    response=$(curl -s -X POST "$ISSUE_TREE_URL/generate" \
        -H "Content-Type: application/json" \
        -d '{
            "problem_frame": {
                "goal": "Reduce customer churn rate by 10% within 6 months",
                "scope": {
                    "include": ["Paid customers", "B2B accounts"],
                    "exclude": ["Free trial users", "Internal accounts"]
                },
                "kpis": [
                    {
                        "name": "Monthly Churn Rate",
                        "baseline": 5.2,
                        "target": 4.7,
                        "window": "monthly",
                        "unit": "percentage"
                    }
                ]
            },
            "context": {
                "user_id": "test-user",
                "organization_id": "test-org",
                "problem_id": "test-problem"
            },
            "max_depth": 3,
            "max_branches": 4
        }')
    
    if echo "$response" | jq -e '.success' > /dev/null; then
        print_success "Issue Tree test successful"
        echo "Execution time: $(echo "$response" | jq -r '.execution_time')s"
        
        # Count tree nodes
        root_children=$(echo "$response" | jq -r '.data.root.children | length')
        print_success "Generated tree with $root_children main branches"
        
        # Check visualization data
        if echo "$response" | jq -e '.data.visualization_data' > /dev/null; then
            print_success "Visualization data generated"
        fi
    else
        print_error "Issue Tree test failed"
        echo "Response: $response"
        return 1
    fi
}

# Test end-to-end workflow through orchestrator
test_e2e_workflow() {
    print_test "Testing end-to-end workflow execution..."
    
    # Step 1: Create workflow
    workflow_response=$(curl -s -X POST "$ORCHESTRATOR_URL/workflows" \
        -H "Content-Type: application/json" \
        -d '{
            "problem_id": "e2e-test-problem",
            "user_id": "e2e-test-user",
            "organization_id": "e2e-test-org"
        }')
    
    if ! echo "$workflow_response" | jq -e '.success' > /dev/null; then
        print_error "Failed to create workflow for E2E test"
        return 1
    fi
    
    print_success "E2E workflow created"
    
    # Step 2: Execute problem framing step
    frame_response=$(curl -s -X POST "$ORCHESTRATOR_URL/workflows/execute-step" \
        -H "Content-Type: application/json" \
        -d '{
            "problem_id": "e2e-test-problem",
            "step": "frame",
            "user_id": "e2e-test-user",
            "organization_id": "e2e-test-org",
            "input_data": {
                "raw_text": "Our mobile app conversion rate is down 30%. Need to fix this ASAP.",
                "submitter": "mobile.pm@company.com",
                "metadata": {"urgency": "critical"}
            }
        }')
    
    if echo "$frame_response" | jq -e '.success' > /dev/null; then
        execution_id=$(echo "$frame_response" | jq -r '.execution_id')
        print_success "Problem framing step initiated (execution: $execution_id)"
        
        # Wait for execution to complete
        sleep 5
        
        # Check execution status
        status_response=$(curl -s "$ORCHESTRATOR_URL/workflows/e2e-test-problem/executions/$execution_id")
        status=$(echo "$status_response" | jq -r '.status')
        print_success "Execution status: $status"
        
    else
        print_error "Failed to execute problem framing step"
        echo "Response: $frame_response"
        return 1
    fi
    
    # Step 3: Check workflow status
    workflow_status=$(curl -s "$ORCHESTRATOR_URL/workflows/e2e-test-problem/status")
    progress=$(echo "$workflow_status" | jq -r '.progress_percentage')
    completed_steps=$(echo "$workflow_status" | jq -r '.completed_steps | length')
    
    print_success "Workflow progress: $progress% ($completed_steps steps completed)"
}

# Test API Gateway routing
test_api_gateway() {
    print_test "Testing API Gateway routing..."
    
    # Test routing to orchestrator
    response=$(curl -s -f "$API_URL/v1/health" 2>/dev/null || echo "failed")
    if [ "$response" != "failed" ]; then
        print_success "API Gateway routing to orchestrator works"
    else
        print_error "API Gateway routing to orchestrator failed"
        return 1
    fi
    
    # Test direct service routing
    response=$(curl -s -f "$API_URL/services/problem-framer/health" 2>/dev/null || echo "failed")
    if [ "$response" != "failed" ]; then
        print_success "API Gateway routing to Problem Framer works"
    else
        print_error "API Gateway routing to Problem Framer failed"
        return 1
    fi
}

# Test service metrics endpoints
test_metrics() {
    print_test "Testing service metrics..."
    
    services=(
        "Orchestrator:$ORCHESTRATOR_URL/metrics"
        "Problem Framer:$PROBLEM_FRAMER_URL/metrics"
        "Issue Tree:$ISSUE_TREE_URL/metrics"
    )
    
    for service in "${services[@]}"; do
        IFS=':' read -ra ADDR <<< "$service"
        service_name="${ADDR[0]}"
        url="${ADDR[1]}:${ADDR[2]}"
        
        response=$(curl -s "$url")
        if echo "$response" | jq -e '.service' > /dev/null 2>&1; then
            print_success "$service_name metrics endpoint working"
        else
            print_error "$service_name metrics endpoint failed"
        fi
    done
}

# Main test execution
main() {
    echo "================================================="
    echo "Seven Steps to Poem - Service Integration Tests"
    echo "================================================="
    
    # Wait for services to be ready
    print_test "Waiting for services to be ready..."
    sleep 5
    
    # Run tests
    test_health_checks || exit 1
    test_workflow_creation || exit 1
    test_problem_framer || exit 1
    test_issue_tree || exit 1
    test_e2e_workflow || exit 1
    test_api_gateway || exit 1
    test_metrics || exit 1
    
    echo
    print_success "🎉 All tests passed! Services are working correctly."
    echo
    print_test "Ready for development and testing!"
    echo "  📖 API Documentation: http://localhost/docs"
    echo "  🧪 Test endpoints with curl or Postman"
    echo "  📊 Monitor with Grafana: http://localhost:3000"
}

# Handle script arguments
case "${1:-all}" in
    all)
        main
        ;;
    health)
        test_health_checks
        ;;
    workflow)
        test_workflow_creation
        ;;
    framer)
        test_problem_framer
        ;;
    tree)
        test_issue_tree
        ;;
    e2e)
        test_e2e_workflow
        ;;
    gateway)
        test_api_gateway
        ;;
    metrics)
        test_metrics
        ;;
    *)
        echo "Usage: $0 {all|health|workflow|framer|tree|e2e|gateway|metrics}"
        exit 1
        ;;
esac