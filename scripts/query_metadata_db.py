#!/usr/bin/env python3
"""Query CAD vector metadata from OceanBase/MySQL database

This script provides a command-line interface for querying the metadata database.

Usage:
    python scripts/query_metadata_db.py stats
    python scripts/query_metadata_db.py get "0000/00000001.h5"
    python scripts/query_metadata_db.py subset 0000 --limit 10
    python scripts/query_metadata_db.py seqlen --min 10 --max 20
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cad_vectordb.database.metadata import MetadataDB
from config import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME


def format_record(record: Dict) -> str:
    """Format a single record for display"""
    return (f"  ID: {record['id']}\n"
            f"  File: {record['file_path']}\n"
            f"  Subset: {record['subset']}\n"
            f"  Seq Len: {record['seq_len']}\n"
            f"  Label: {record.get('label', 'N/A')}\n"
            f"  Source: {record.get('source', 'N/A')}")


def cmd_stats(db: MetadataDB, args):
    """Display database statistics"""
    print("=" * 60)
    print("Database Statistics")
    print("=" * 60)
    print()
    
    stats = db.get_stats()
    
    print(f"📊 Total Vectors: {stats['total_vectors']:,}")
    print(f"📁 Unique Subsets: {stats['unique_subsets']}")
    print(f"📏 Sequence Length:")
    print(f"   Min: {stats['min_seq_len']}")
    print(f"   Max: {stats['max_seq_len']}")
    print(f"   Avg: {stats['avg_seq_len']:.2f}")
    print()


def cmd_get(db: MetadataDB, args):
    """Get a specific record by ID"""
    print("=" * 60)
    print(f"Get Record: {args.id}")
    print("=" * 60)
    print()
    
    results = db.query(limit=1000000)  # Get all and filter
    record = None
    for r in results:
        if r['id'] == args.id:
            record = r
            break
    
    if record:
        print("✅ Found record:")
        print()
        print(format_record(record))
    else:
        print(f"❌ Record not found: {args.id}")
        print()
        print("Available IDs (first 5):")
        for r in results[:5]:
            print(f"  - {r['id']}")
    
    print()


def cmd_subset(db: MetadataDB, args):
    """Query records by subset"""
    print("=" * 60)
    print(f"Query Subset: {args.subset}")
    print("=" * 60)
    print()
    
    results = db.query(subset=args.subset, limit=args.limit)
    
    print(f"✅ Found {len(results)} records")
    print()
    
    if results:
        for i, record in enumerate(results[:args.show], 1):
            print(f"[{i}]")
            print(format_record(record))
            print()
        
        if len(results) > args.show:
            print(f"... and {len(results) - args.show} more records")
            print(f"Use --show {len(results)} to see all")
    else:
        print("No records found")
    
    print()


def cmd_seqlen(db: MetadataDB, args):
    """Query records by sequence length range"""
    print("=" * 60)
    print(f"Query Sequence Length: {args.min} - {args.max}")
    print("=" * 60)
    print()
    
    results = db.query(
        min_seq_len=args.min,
        max_seq_len=args.max,
        limit=args.limit
    )
    
    print(f"✅ Found {len(results)} records")
    print()
    
    if results:
        # Group by subset
        by_subset = {}
        for r in results:
            subset = r['subset']
            by_subset.setdefault(subset, []).append(r)
        
        print(f"Distribution by subset:")
        for subset, records in sorted(by_subset.items()):
            print(f"  {subset}: {len(records)} records")
        print()
        
        print(f"Sample records (showing first {args.show}):")
        for i, record in enumerate(results[:args.show], 1):
            print(f"[{i}]")
            print(format_record(record))
            print()
        
        if len(results) > args.show:
            print(f"... and {len(results) - args.show} more records")
    else:
        print("No records found")
    
    print()


def cmd_export(db: MetadataDB, args):
    """Export query results to JSON"""
    print("=" * 60)
    print(f"Export to: {args.output}")
    print("=" * 60)
    print()
    
    # Build query
    query_params = {}
    if args.subset:
        query_params['subset'] = args.subset
    if args.min_seqlen:
        query_params['min_seq_len'] = args.min_seqlen
    if args.max_seqlen:
        query_params['max_seq_len'] = args.max_seqlen
    
    results = db.query(**query_params, limit=args.limit)
    
    print(f"✅ Queried {len(results)} records")
    
    # Export to JSON
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Exported to: {output_path}")
    print(f"   File size: {output_path.stat().st_size:,} bytes")
    print()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Query CAD vector metadata from OceanBase/MySQL',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  stats              Show database statistics
  get ID             Get a specific record by ID
  subset SUBSET      Query records by subset
  seqlen             Query records by sequence length range
  export             Export query results to JSON

Examples:
  # Show statistics
  python scripts/query_metadata_db.py stats
  
  # Get specific record
  python scripts/query_metadata_db.py get "0000/00000001.h5"
  
  # Query by subset
  python scripts/query_metadata_db.py subset 0000 --limit 10
  python scripts/query_metadata_db.py subset 0001 --show 20
  
  # Query by sequence length
  python scripts/query_metadata_db.py seqlen --min 10 --max 20
  python scripts/query_metadata_db.py seqlen --max 15 --limit 100
  
  # Export results
  python scripts/query_metadata_db.py export --subset 0000 --output results.json
  python scripts/query_metadata_db.py export --min-seqlen 10 --max-seqlen 20 --output filtered.json

For more information, see docs/OCEANBASE_GUIDE.md
        """
    )
    
    # Database connection
    parser.add_argument('--host', type=str, default=DB_HOST, help=f'Database host (default: {DB_HOST})')
    parser.add_argument('--port', type=int, default=DB_PORT, help=f'Database port (default: {DB_PORT})')
    parser.add_argument('--user', type=str, default=DB_USER, help=f'Database user (default: {DB_USER})')
    parser.add_argument('--password', type=str, default=DB_PASSWORD, help='Database password')
    parser.add_argument('--database', '-d', type=str, default=DB_NAME, help=f'Database name (default: {DB_NAME})')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # stats command
    subparsers.add_parser('stats', help='Show database statistics')
    
    # get command
    get_parser = subparsers.add_parser('get', help='Get a specific record by ID')
    get_parser.add_argument('id', type=str, help='Record ID (e.g., "0000/00000001.h5")')
    
    # subset command
    subset_parser = subparsers.add_parser('subset', help='Query records by subset')
    subset_parser.add_argument('subset', type=str, help='Subset ID (e.g., "0000")')
    subset_parser.add_argument('--limit', type=int, default=100, help='Maximum results (default: 100)')
    subset_parser.add_argument('--show', type=int, default=5, help='Number of records to display (default: 5)')
    
    # seqlen command
    seqlen_parser = subparsers.add_parser('seqlen', help='Query records by sequence length range')
    seqlen_parser.add_argument('--min', type=int, help='Minimum sequence length')
    seqlen_parser.add_argument('--max', type=int, help='Maximum sequence length')
    seqlen_parser.add_argument('--limit', type=int, default=100, help='Maximum results (default: 100)')
    seqlen_parser.add_argument('--show', type=int, default=5, help='Number of records to display (default: 5)')
    
    # export command
    export_parser = subparsers.add_parser('export', help='Export query results to JSON')
    export_parser.add_argument('--output', '-o', type=str, required=True, help='Output JSON file')
    export_parser.add_argument('--subset', type=str, help='Filter by subset')
    export_parser.add_argument('--min-seqlen', type=int, help='Minimum sequence length')
    export_parser.add_argument('--max-seqlen', type=int, help='Maximum sequence length')
    export_parser.add_argument('--limit', type=int, default=10000, help='Maximum results (default: 10000)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    return args


def main():
    """Main query process"""
    args = parse_args()
    
    # Connect to database
    db_config = {
        'host': args.host,
        'port': args.port,
        'user': args.user,
        'password': args.password,
        'database': args.database
    }
    
    try:
        db = MetadataDB(**db_config)
        db.connect()
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print()
        print("Troubleshooting:")
        print("1. Is OceanBase/MySQL running?")
        print("2. Have you imported metadata?")
        print(f"   python scripts/import_metadata_to_oceanbase.py --metadata data/indices/metadata.json")
        print("3. Check connection parameters in config.py")
        sys.exit(1)
    
    # Execute command
    try:
        if args.command == 'stats':
            cmd_stats(db, args)
        elif args.command == 'get':
            cmd_get(db, args)
        elif args.command == 'subset':
            cmd_subset(db, args)
        elif args.command == 'seqlen':
            cmd_seqlen(db, args)
        elif args.command == 'export':
            cmd_export(db, args)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    finally:
        db.close()


if __name__ == '__main__':
    main()
