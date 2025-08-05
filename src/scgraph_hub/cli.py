"""Command-line interface for Single-Cell Graph Hub."""

import click
import sys
import os
from pathlib import Path
from typing import Optional, List
import json
import asyncio

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from .catalog import DatasetCatalog, get_default_catalog
from .utils import check_dependencies, validate_dataset_config


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def main(ctx, verbose):
    """Single-Cell Graph Hub - Graph Neural Networks for Single-Cell Omics."""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    
    if verbose:
        click.echo("Single-Cell Graph Hub CLI")
        click.echo("=" * 40)


@main.group()
def catalog():
    """Dataset catalog operations."""
    pass


@catalog.command()
@click.option('--format', 'output_format', 
              type=click.Choice(['table', 'json', 'csv']), 
              default='table',
              help='Output format')
@click.option('--modality', help='Filter by data modality')
@click.option('--organism', help='Filter by organism') 
@click.option('--min-cells', type=int, help='Minimum number of cells')
@click.option('--has-spatial/--no-spatial', default=None, help='Filter by spatial data availability')
def list(output_format, modality, organism, min_cells, has_spatial):
    """List available datasets."""
    try:
        cat = get_default_catalog()
        
        # Apply filters
        if any([modality, organism, min_cells, has_spatial is not None]):
            datasets = cat.filter(
                modality=modality,
                organism=organism,
                min_cells=min_cells,
                has_spatial=has_spatial
            )
        else:
            datasets = cat.list_datasets()
        
        if output_format == 'json':
            dataset_info = {}
            for name in datasets:
                dataset_info[name] = cat.get_info(name)
            click.echo(json.dumps(dataset_info, indent=2))
            
        elif output_format == 'csv':
            click.echo("name,cells,genes,modality,organism,tissue")
            for name in datasets:
                info = cat.get_info(name)
                click.echo(f"{name},{info['n_cells']},{info['n_genes']},{info['modality']},{info['organism']},{info['tissue']}")
                
        else:  # table format
            click.echo(f"Found {len(datasets)} datasets:")
            click.echo("")
            click.echo(f"{'Name':<20} {'Cells':<8} {'Genes':<8} {'Modality':<15} {'Organism':<10} {'Tissue':<15}")
            click.echo("-" * 85)
            
            for name in datasets:
                info = cat.get_info(name)
                click.echo(f"{name:<20} {info['n_cells']:<8} {info['n_genes']:<8} {info['modality']:<15} {info['organism']:<10} {info['tissue']:<15}")
                
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@catalog.command()
@click.argument('dataset_name')
def info(dataset_name):
    """Show detailed information about a dataset."""
    try:
        cat = get_default_catalog()
        dataset_info = cat.get_info(dataset_name)
        
        if not dataset_info:
            click.echo(f"Dataset '{dataset_name}' not found", err=True)
            sys.exit(1)
        
        click.echo(f"Dataset: {dataset_name}")
        click.echo("=" * (len(dataset_name) + 9))
        
        for key, value in dataset_info.items():
            if isinstance(value, list):
                value = ', '.join(str(v) for v in value)
            click.echo(f"{key.replace('_', ' ').title()}: {value}")
            
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@catalog.command()
@click.argument('query')
@click.option('--max-results', default=10, help='Maximum number of results')
def search(query, max_results):
    """Search for datasets by keyword."""
    try:
        cat = get_default_catalog()
        results = cat.search(query)
        
        if not results:
            click.echo(f"No datasets found matching '{query}'")
            return
        
        # Limit results
        results = results[:max_results]
        
        click.echo(f"Found {len(results)} datasets matching '{query}':")
        click.echo("")
        
        for name in results:
            info = cat.get_info(name)
            click.echo(f"• {name} - {info['description']}")
            click.echo(f"  {info['n_cells']} cells, {info['n_genes']} genes, {info['modality']}")
            click.echo("")
            
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@catalog.command()
@click.argument('dataset_name')
@click.option('--max-results', default=5, help='Maximum number of recommendations')
def recommend(dataset_name, max_results):
    """Get dataset recommendations based on similarity."""
    try:
        cat = get_default_catalog()
        recommendations = cat.get_recommendations(dataset_name, max_results=max_results)
        
        if not recommendations:
            click.echo(f"No recommendations found for '{dataset_name}'")
            return
        
        click.echo(f"Datasets similar to '{dataset_name}':")
        click.echo("")
        
        for name in recommendations:
            info = cat.get_info(name)
            click.echo(f"• {name} - {info['description']}")
            click.echo(f"  {info['n_cells']} cells, {info['n_genes']} genes, {info['modality']}")
            click.echo("")
            
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@catalog.command()
def stats():
    """Show catalog statistics."""
    try:
        cat = get_default_catalog()
        summary = cat.get_summary_stats()
        
        click.echo("Catalog Statistics")
        click.echo("=" * 18)
        
        for key, value in summary.items():
            if isinstance(value, dict):
                click.echo(f"{key.replace('_', ' ').title()}:")
                for sub_key, sub_value in value.items():
                    click.echo(f"  {sub_key}: {sub_value}")
            else:
                click.echo(f"{key.replace('_', ' ').title()}: {value}")
            
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.group()
def data():
    """Data management operations."""
    pass


@data.command()
@click.argument('dataset_name')
@click.option('--output-dir', default='./data', help='Output directory for downloaded data')
@click.option('--force', is_flag=True, help='Force redownload if file exists')
def download(dataset_name, output_dir, force):
    """Download a dataset."""
    try:
        # Check if core functionality is available
        from . import _CORE_AVAILABLE
        if not _CORE_AVAILABLE:
            click.echo("Error: Core functionality not available.", err=True)
            click.echo("Install with: pip install single-cell-graph-hub[full]", err=True)
            sys.exit(1)
            
        from .data_manager import get_data_manager
        
        click.echo(f"Downloading dataset: {dataset_name}")
        click.echo(f"Output directory: {output_dir}")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Note: This is a simplified implementation
        # In reality, we'd use the async download functionality
        click.echo("Download functionality requires async implementation")
        click.echo("Use Python API for full download capabilities")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def quick_start():
    """Interactive quick start guide."""
    click.echo("Single-Cell Graph Hub - Quick Start Guide")
    click.echo("=" * 45)
    click.echo("")
    
    # Check dependencies
    click.echo("1. Checking dependencies...")
    deps_available = check_dependencies()
    if deps_available:
        click.echo("✅ All dependencies available")
    else:
        click.echo("❌ Missing dependencies")
        click.echo("Install with: pip install single-cell-graph-hub[full]")
        return
    
    # List available datasets
    click.echo("\n2. Available datasets:")
    try:
        cat = get_default_catalog()
        datasets = cat.list_datasets()[:5]  # Show first 5
        for name in datasets:
            info = cat.get_info(name)
            click.echo(f"  • {name} ({info['n_cells']} cells, {info['modality']})")
        click.echo(f"  ... and {len(cat.list_datasets()) - 5} more")
    except Exception as e:
        click.echo(f"Error loading catalog: {e}")
        return
    
    click.echo("\n3. Next steps:")
    click.echo("  - Use 'scgraph catalog list' to see all datasets")
    click.echo("  - Use 'scgraph catalog info <dataset>' for details")
    click.echo("  - Use 'scgraph data download <dataset>' to download")
    click.echo("  - Check examples/ directory for usage patterns")
    click.echo("")
    click.echo("For full API usage, see the documentation or examples/")


@main.command()
def version():
    """Show version information."""
    try:
        from . import __version__
        click.echo(f"Single-Cell Graph Hub version {__version__}")
    except ImportError:
        click.echo("Version information not available")


@main.command(name='quick-start')
def quick_start_command():
    """Interactive quick start guide."""
    quick_start()


@main.command()
@click.option('--detailed', is_flag=True, help='Show detailed health information')
@click.option('--json-output', is_flag=True, help='Output results in JSON format')
def check(detailed, json_output):
    """Check system dependencies and configuration."""
    import asyncio
    
    async def run_check():
        if json_output:
            # Import health check functionality
            try:
                from .health_checks import run_health_check
                results = await run_health_check()
                click.echo(json.dumps(results, indent=2))
                return
            except ImportError:
                click.echo('{"error": "Health check functionality not available"}')
                return
        
        click.echo("Single-Cell Graph Hub - System Check")
        click.echo("=" * 35)
        click.echo("")
        
        # Check Python version
        click.echo(f"Python version: {sys.version}")
        
        # Check dependencies
        click.echo("\nDependency check:")
        deps_available = check_dependencies()
        if deps_available:
            click.echo("✅ All core dependencies available")
        else:
            click.echo("❌ Missing core dependencies")
            click.echo("Install with: pip install single-cell-graph-hub[full]")
        
        # Test catalog
        click.echo("\nCatalog check:")
        try:
            cat = get_default_catalog()
            datasets = cat.list_datasets()
            click.echo(f"✅ Catalog loaded: {len(datasets)} datasets available")
        except Exception as e:
            click.echo(f"❌ Catalog error: {e}")
        
        # Check core functionality
        click.echo("\nCore functionality check:")
        try:
            from . import _CORE_AVAILABLE
            if _CORE_AVAILABLE:
                click.echo("✅ Core functionality available")
            else:
                click.echo("⚠️  Core functionality limited (missing dependencies)")
        except Exception as e:
            click.echo(f"❌ Core functionality error: {e}")
        
        # Run detailed health check if requested
        if detailed:
            try:
                from .health_checks import run_health_check
                click.echo("\nRunning detailed health check...")
                health_results = await run_health_check()
                
                click.echo(f"\nOverall status: {health_results['overall_status'].upper()}")
                click.echo(f"System uptime: {health_results['uptime_seconds']:.1f}s")
                
                click.echo("\nComponent health:")
                for component, status in health_results['components'].items():
                    status_icon = "✅" if status['status'] == 'healthy' else "⚠️" if status['status'] == 'degraded' else "❌"
                    click.echo(f"{status_icon} {component}: {status['status']} - {status['message']}")
                    
                    # Show resource details
                    if component == 'system_resources' and status.get('details'):
                        details = status['details']
                        click.echo(f"   Memory: {details.get('memory_percent', 0):.1f}%, CPU: {details.get('cpu_percent', 0):.1f}%")
            
            except ImportError:
                click.echo("\n⚠️  Detailed health checks not available (missing dependencies)")
        
        click.echo("\nSystem check complete.")
    
    # Run the async check
    asyncio.run(run_check())


if __name__ == '__main__':
    main()