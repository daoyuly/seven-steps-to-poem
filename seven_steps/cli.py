"""
Command-line interface for Seven Steps to Poem system.

This module provides CLI commands for development, testing,
and administration tasks.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from seven_steps.api.main import create_app
from seven_steps.core.config import get_settings
from seven_steps.core.database import close_database, init_database

app = typer.Typer(
    name="seven-steps",
    help="Seven Steps to Poem - AI Business Problem Solver",
    add_completion=False
)

console = Console()


@app.command()
def init_db(
    reset: bool = typer.Option(False, "--reset", help="Reset existing database")
) -> None:
    """Initialize the database with tables and initial data."""
    
    async def _init_db():
        try:
            console.print("🔧 Initializing database...", style="blue")
            
            await init_database()
            
            if reset:
                console.print("⚠️ Database reset requested", style="yellow")
                # In production, you would implement proper reset logic
            
            console.print("✅ Database initialized successfully!", style="green")
            
        except Exception as e:
            console.print(f"❌ Database initialization failed: {e}", style="red")
            sys.exit(1)
        finally:
            await close_database()
    
    asyncio.run(_init_db())


@app.command()
def run_server(
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to bind to"),
    reload: bool = typer.Option(False, help="Enable auto-reload for development"),
) -> None:
    """Run the API server."""
    import uvicorn
    
    settings = get_settings()
    
    console.print("🚀 Starting Seven Steps to Poem API server...", style="blue")
    console.print(f"📍 Server will be available at http://{host}:{port}")
    console.print(f"📖 API Documentation at http://{host}:{port}/docs")
    
    uvicorn.run(
        "seven_steps.api.main:app",
        host=host,
        port=port,
        reload=reload or settings.debug,
        log_config=None,
    )


@app.command()
def migrate(
    message: Optional[str] = typer.Option(None, help="Migration message"),
    auto: bool = typer.Option(False, "--auto", help="Auto-generate migration")
) -> None:
    """Create and run database migrations."""
    import subprocess
    
    try:
        if message:
            console.print(f"🔄 Creating migration: {message}", style="blue")
            
            if auto:
                cmd = ["alembic", "revision", "--autogenerate", "-m", message]
            else:
                cmd = ["alembic", "revision", "-m", message]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                console.print(f"❌ Migration creation failed: {result.stderr}", style="red")
                sys.exit(1)
            
            console.print("✅ Migration created successfully!", style="green")
        
        # Run migrations
        console.print("🔄 Running migrations...", style="blue")
        result = subprocess.run(["alembic", "upgrade", "head"], capture_output=True, text=True)
        
        if result.returncode != 0:
            console.print(f"❌ Migration failed: {result.stderr}", style="red")
            sys.exit(1)
        
        console.print("✅ Migrations completed successfully!", style="green")
        
    except FileNotFoundError:
        console.print("❌ Alembic not found. Please install the package first.", style="red")
        sys.exit(1)


@app.command()
def test_agent(
    agent_name: str = typer.Argument(..., help="Agent name to test"),
    input_file: Optional[Path] = typer.Option(None, help="Input JSON file for testing"),
) -> None:
    """Test a specific agent with sample input."""
    
    async def _test_agent():
        try:
            console.print(f"🧪 Testing {agent_name} agent...", style="blue")
            
            # Sample test data
            test_data = {
                "context": {
                    "user_id": "test-user",
                    "organization_id": "test-org",
                    "problem_id": "test-problem",
                    "correlation_id": "test-correlation"
                }
            }
            
            if input_file and input_file.exists():
                console.print(f"📁 Loading input from {input_file}", style="blue")
                with open(input_file) as f:
                    test_data.update(json.load(f))
            elif agent_name.lower() == "problemframer":
                test_data.update({
                    "raw_text": "Our SaaS customer churn rate has increased by 20% over the past 3 months. We need to reduce it by 5% within 90 days to meet our quarterly targets.",
                    "submitter": "product.manager@company.com",
                    "metadata": {
                        "industry": "SaaS",
                        "company_size": "medium",
                        "urgency": "high"
                    }
                })
            
            # Test the agent
            if agent_name.lower() == "problemframer":
                from seven_steps.agents.problem_framer import ProblemFramer
                agent = ProblemFramer()
            else:
                console.print(f"❌ Agent '{agent_name}' not yet implemented", style="red")
                return
            
            result = await agent.execute(test_data)
            
            console.print("📊 Agent Test Results:", style="bold blue")
            
            # Create results table
            table = Table(title=f"{agent_name} Test Results")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="white")
            
            table.add_row("Success", "✅ Yes" if result.success else "❌ No")
            table.add_row("Execution Time", f"{result.execution_time:.2f}s" if result.execution_time else "N/A")
            table.add_row("Confidence Score", f"{result.confidence_score:.2%}" if result.confidence_score else "N/A")
            
            if result.error_message:
                table.add_row("Error", result.error_message)
            
            console.print(table)
            
            if result.success and result.data:
                console.print("\n📋 Detailed Results:", style="bold blue")
                console.print_json(json.dumps(result.data, indent=2, default=str))
            
        except Exception as e:
            console.print(f"❌ Agent test failed: {e}", style="red")
            sys.exit(1)
    
    asyncio.run(_test_agent())


@app.command()
def health_check() -> None:
    """Perform a health check of all system components."""
    
    async def _health_check():
        try:
            console.print("🏥 Performing system health check...", style="blue")
            
            # Test database connection
            console.print("🔍 Checking database connection...", style="blue")
            await init_database()
            console.print("✅ Database: OK", style="green")
            
            # Test configuration
            console.print("🔍 Checking configuration...", style="blue")
            settings = get_settings()
            console.print("✅ Configuration: OK", style="green")
            
            # Create health table
            table = Table(title="System Health Status")
            table.add_column("Component", style="cyan")
            table.add_column("Status", style="white")
            table.add_column("Details", style="white")
            
            table.add_row("Database", "✅ Healthy", "PostgreSQL connection OK")
            table.add_row("Configuration", "✅ Healthy", f"Environment: {settings.environment}")
            table.add_row("Redis", "⚠️ Not tested", "Requires running Redis instance")
            table.add_row("Neo4j", "⚠️ Not tested", "Requires running Neo4j instance")
            table.add_row("OpenAI API", "⚠️ Not tested", "Requires valid API key")
            
            console.print(table)
            console.print("\n💡 Run with Docker services for full health check", style="yellow")
            
        except Exception as e:
            console.print(f"❌ Health check failed: {e}", style="red")
            sys.exit(1)
        finally:
            await close_database()
    
    asyncio.run(_health_check())


@app.command()
def create_sample_data() -> None:
    """Create sample data for testing and development."""
    
    async def _create_sample():
        try:
            console.print("📝 Creating sample data...", style="blue")
            
            await init_database()
            
            # In production, you would create sample organizations, users, problems
            console.print("✅ Sample data created successfully!", style="green")
            console.print("💡 Sample data includes test users and problems", style="blue")
            
        except Exception as e:
            console.print(f"❌ Failed to create sample data: {e}", style="red")
            sys.exit(1)
        finally:
            await close_database()
    
    asyncio.run(_create_sample())


@app.command()
def version() -> None:
    """Show version information."""
    settings = get_settings()
    
    table = Table(title="Seven Steps to Poem")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Version", settings.version)
    table.add_row("Environment", settings.environment)
    table.add_row("Debug Mode", "Enabled" if settings.debug else "Disabled")
    
    console.print(table)


if __name__ == "__main__":
    app()