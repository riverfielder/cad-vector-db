"""
Import metadata.json to OceanBase database
"""
import json
import argparse
import pymysql
from typing import List, Dict
import sys
import os

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD


def create_connection(host: str, port: int, user: str, password: str, database: str = None):
    """Create OceanBase/MySQL connection"""
    try:
        conn = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            charset='utf8mb4'
        )
        print(f"✓ Successfully connected to OceanBase at {host}:{port}")
        return conn
    except Exception as e:
        print(f"✗ Failed to connect to OceanBase: {e}")
        raise


def create_database(conn, db_name: str):
    """Create database if not exists"""
    with conn.cursor() as cur:
        cur.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}` DEFAULT CHARSET=utf8mb4")
        print(f"✓ Database `{db_name}` ready")


def create_table(conn):
    """Create cad_vectors table"""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS `cad_vectors` (
        `id` VARCHAR(255) PRIMARY KEY COMMENT 'Unique identifier (subset/filename)',
        `file_path` TEXT NOT NULL COMMENT 'Absolute path to h5 file',
        `subset` VARCHAR(50) NOT NULL COMMENT 'Subset directory (e.g., 0000, 0001)',
        `seq_len` INT NOT NULL COMMENT 'Sequence length of macro vector',
        `label` VARCHAR(100) DEFAULT NULL COMMENT 'Optional label (for future use)',
        `source` VARCHAR(50) DEFAULT 'WHUCAD' COMMENT 'Data source',
        `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Import timestamp',
        INDEX `idx_subset` (`subset`),
        INDEX `idx_seq_len` (`seq_len`),
        INDEX `idx_label` (`label`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='CAD vector metadata table'
    """
    
    with conn.cursor() as cur:
        cur.execute(create_table_sql)
        print("✓ Table `cad_vectors` created/verified")


def load_metadata(metadata_file: str) -> List[Dict]:
    """Load metadata from JSON file"""
    print(f"Loading metadata from {metadata_file}...")
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    print(f"✓ Loaded {len(metadata)} records")
    return metadata


def batch_insert(conn, records: List[Dict], batch_size: int = 1000):
    """Batch insert records into database"""
    total = len(records)
    inserted = 0
    duplicates = 0
    
    insert_sql = """
    INSERT INTO `cad_vectors` (`id`, `file_path`, `subset`, `seq_len`, `label`, `source`)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        file_path = VALUES(file_path),
        seq_len = VALUES(seq_len),
        label = VALUES(label)
    """
    
    with conn.cursor() as cur:
        for i in range(0, total, batch_size):
            batch = records[i:i+batch_size]
            values = [
                (
                    rec['id'],
                    rec['file_path'],
                    rec['subset'],
                    rec['seq_len'],
                    rec.get('label'),
                    rec.get('source', 'WHUCAD')
                )
                for rec in batch
            ]
            
            affected = cur.executemany(insert_sql, values)
            conn.commit()
            
            # Estimate duplicates (affected rows = inserts + 2*updates)
            batch_inserted = min(affected, len(batch))
            inserted += batch_inserted
            
            progress = min(i + batch_size, total)
            print(f"Progress: {progress}/{total} ({progress*100//total}%)")
    
    print(f"\n✓ Import completed!")
    print(f"  - Total records: {total}")
    print(f"  - Inserted/Updated: {inserted}")


def query_stats(conn):
    """Query and display database statistics"""
    print("\n" + "="*60)
    print("DATABASE STATISTICS")
    print("="*60)
    
    with conn.cursor() as cur:
        # Total count
        cur.execute("SELECT COUNT(*) FROM `cad_vectors`")
        total = cur.fetchone()[0]
        print(f"Total vectors: {total}")
        
        # Count by subset
        cur.execute("""
            SELECT `subset`, COUNT(*) as count 
            FROM `cad_vectors` 
            GROUP BY `subset` 
            ORDER BY `subset`
            LIMIT 10
        """)
        print("\nTop 10 subsets:")
        for subset, count in cur.fetchall():
            print(f"  {subset}: {count}")
        
        # Sequence length statistics
        cur.execute("""
            SELECT 
                MIN(`seq_len`) as min_len,
                MAX(`seq_len`) as max_len,
                AVG(`seq_len`) as avg_len
            FROM `cad_vectors`
        """)
        min_len, max_len, avg_len = cur.fetchone()
        print(f"\nSequence length:")
        print(f"  Min: {min_len}")
        print(f"  Max: {max_len}")
        print(f"  Avg: {avg_len:.2f}")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Import metadata.json to OceanBase')
    parser.add_argument('--metadata', type=str, required=True,
                        help='Path to metadata.json file')
    parser.add_argument('--host', type=str, default=DB_HOST,
                        help=f'OceanBase host (default: {DB_HOST})')
    parser.add_argument('--port', type=int, default=DB_PORT,
                        help=f'OceanBase port (default: {DB_PORT})')
    parser.add_argument('--user', type=str, default=DB_USER,
                        help=f'Database user (default: {DB_USER})')
    parser.add_argument('--password', type=str, default=DB_PASSWORD,
                        help='Database password')
    parser.add_argument('--database', type=str, default=DB_NAME,
                        help=f'Database name (default: {DB_NAME})')
    parser.add_argument('--batch-size', type=int, default=1000,
                        help='Batch insert size (default: 1000)')
    parser.add_argument('--drop-table', action='store_true',
                        help='Drop existing table before import')
    
    args = parser.parse_args()
    
    # Step 1: Connect to OceanBase
    print("\n" + "="*60)
    print("STEP 1: Connect to OceanBase")
    print("="*60)
    conn = create_connection(args.host, args.port, args.user, args.password)
    
    # Step 2: Create database
    print("\n" + "="*60)
    print("STEP 2: Create Database")
    print("="*60)
    create_database(conn, args.database)
    conn.close()
    
    # Reconnect to the specific database
    conn = create_connection(args.host, args.port, args.user, args.password, args.database)
    
    # Step 3: Drop table if requested
    if args.drop_table:
        print("\n" + "="*60)
        print("STEP 3: Drop Existing Table")
        print("="*60)
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS `cad_vectors`")
            print("✓ Table dropped")
    
    # Step 4: Create table
    print("\n" + "="*60)
    print(f"STEP {'4' if args.drop_table else '3'}: Create Table")
    print("="*60)
    create_table(conn)
    
    # Step 5: Load metadata
    print("\n" + "="*60)
    print(f"STEP {'5' if args.drop_table else '4'}: Load Metadata")
    print("="*60)
    metadata = load_metadata(args.metadata)
    
    # Step 6: Import data
    print("\n" + "="*60)
    print(f"STEP {'6' if args.drop_table else '5'}: Import Data")
    print("="*60)
    batch_insert(conn, metadata, args.batch_size)
    
    # Step 7: Display statistics
    print("\n" + "="*60)
    print(f"STEP {'7' if args.drop_table else '6'}: Statistics")
    print("="*60)
    query_stats(conn)
    
    # Close connection
    conn.close()
    print("\n✓ All done! Connection closed.")


if __name__ == '__main__':
    main()
