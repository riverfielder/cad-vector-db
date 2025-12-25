"""
Query metadata from OceanBase database
Provides utility functions for filtering and searching vectors
"""
import pymysql
import argparse
import json
from typing import List, Dict, Optional
import sys
import os

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD


class MetadataDB:
    """OceanBase metadata database client"""
    
    def __init__(self, host: str, port: int, user: str, password: str, database: str):
        self.conn = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            charset='utf8mb4'
        )
    
    def close(self):
        """Close database connection"""
        self.conn.close()
    
    def get_by_id(self, vector_id: str) -> Optional[Dict]:
        """Get metadata by vector ID"""
        with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(
                "SELECT * FROM `cad_vectors` WHERE `id` = %s",
                (vector_id,)
            )
            return cur.fetchone()
    
    def filter_by_subset(self, subset: str, limit: int = 100) -> List[Dict]:
        """Filter vectors by subset"""
        with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(
                "SELECT * FROM `cad_vectors` WHERE `subset` = %s LIMIT %s",
                (subset, limit)
            )
            return cur.fetchall()
    
    def filter_by_seq_len(self, min_len: int = None, max_len: int = None, 
                          limit: int = 100) -> List[Dict]:
        """Filter vectors by sequence length range"""
        conditions = []
        params = []
        
        if min_len is not None:
            conditions.append("`seq_len` >= %s")
            params.append(min_len)
        if max_len is not None:
            conditions.append("`seq_len` <= %s")
            params.append(max_len)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)
        
        with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(
                f"SELECT * FROM `cad_vectors` WHERE {where_clause} LIMIT %s",
                params
            )
            return cur.fetchall()
    
    def get_ids_by_filter(self, subset: str = None, min_len: int = None, 
                          max_len: int = None, label: str = None) -> List[str]:
        """Get vector IDs matching filters (for retrieval filtering)"""
        conditions = []
        params = []
        
        if subset:
            conditions.append("`subset` = %s")
            params.append(subset)
        if min_len is not None:
            conditions.append("`seq_len` >= %s")
            params.append(min_len)
        if max_len is not None:
            conditions.append("`seq_len` <= %s")
            params.append(max_len)
        if label:
            conditions.append("`label` = %s")
            params.append(label)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT `id` FROM `cad_vectors` WHERE {where_clause}",
                params
            )
            return [row[0] for row in cur.fetchall()]
    
    def batch_get_metadata(self, vector_ids: List[str]) -> Dict[str, Dict]:
        """Batch retrieve metadata for multiple vector IDs"""
        if not vector_ids:
            return {}
        
        placeholders = ','.join(['%s'] * len(vector_ids))
        
        with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(
                f"SELECT * FROM `cad_vectors` WHERE `id` IN ({placeholders})",
                vector_ids
            )
            results = cur.fetchall()
            return {row['id']: row for row in results}
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        stats = {}
        
        with self.conn.cursor() as cur:
            # Total count
            cur.execute("SELECT COUNT(*) FROM `cad_vectors`")
            stats['total'] = cur.fetchone()[0]
            
            # Subset counts
            cur.execute("""
                SELECT `subset`, COUNT(*) as count 
                FROM `cad_vectors` 
                GROUP BY `subset` 
                ORDER BY count DESC
            """)
            stats['subsets'] = {row[0]: row[1] for row in cur.fetchall()}
            
            # Sequence length stats
            cur.execute("""
                SELECT 
                    MIN(`seq_len`) as min,
                    MAX(`seq_len`) as max,
                    AVG(`seq_len`) as avg
                FROM `cad_vectors`
            """)
            min_len, max_len, avg_len = cur.fetchone()
            stats['seq_len'] = {
                'min': min_len,
                'max': max_len,
                'avg': float(avg_len) if avg_len else 0
            }
        
        return stats


def main():
    parser = argparse.ArgumentParser(description='Query metadata from OceanBase')
    parser.add_argument('--host', type=str, default=DB_HOST)
    parser.add_argument('--port', type=int, default=DB_PORT)
    parser.add_argument('--user', type=str, default=DB_USER)
    parser.add_argument('--password', type=str, default=DB_PASSWORD)
    parser.add_argument('--database', type=str, default=DB_NAME)
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Stats command
    subparsers.add_parser('stats', help='Show database statistics')
    
    # Get by ID command
    get_parser = subparsers.add_parser('get', help='Get metadata by ID')
    get_parser.add_argument('id', type=str, help='Vector ID')
    
    # Filter by subset command
    subset_parser = subparsers.add_parser('subset', help='Filter by subset')
    subset_parser.add_argument('subset', type=str, help='Subset name')
    subset_parser.add_argument('--limit', type=int, default=10)
    
    # Filter by sequence length command
    seqlen_parser = subparsers.add_parser('seqlen', help='Filter by sequence length')
    seqlen_parser.add_argument('--min', type=int, help='Minimum length')
    seqlen_parser.add_argument('--max', type=int, help='Maximum length')
    seqlen_parser.add_argument('--limit', type=int, default=10)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Create database client
    db = MetadataDB(args.host, args.port, args.user, args.password, args.database)
    
    try:
        if args.command == 'stats':
            stats = db.get_stats()
            print(json.dumps(stats, indent=2, ensure_ascii=False))
        
        elif args.command == 'get':
            result = db.get_by_id(args.id)
            if result:
                print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
            else:
                print(f"No record found for ID: {args.id}")
        
        elif args.command == 'subset':
            results = db.filter_by_subset(args.subset, args.limit)
            print(f"Found {len(results)} records:")
            print(json.dumps(results, indent=2, ensure_ascii=False, default=str))
        
        elif args.command == 'seqlen':
            results = db.filter_by_seq_len(args.min, args.max, args.limit)
            print(f"Found {len(results)} records:")
            print(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    
    finally:
        db.close()


if __name__ == '__main__':
    main()
