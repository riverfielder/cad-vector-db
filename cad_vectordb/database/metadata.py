"""Metadata database operations for CAD Vector Database

Provides OceanBase/MySQL database operations for vector metadata:
- Creating and managing database schema
- Importing metadata from JSON
- Querying metadata with filters
- Hybrid search support
"""
import json
import pymysql
from typing import List, Dict, Optional
from pathlib import Path


class MetadataDB:
    """Metadata database manager"""
    
    def __init__(self, host: str, port: int, user: str, password: str, database: str):
        """Initialize database connection parameters
        
        Args:
            host: Database host
            port: Database port
            user: Database user
            password: Database password
            database: Database name
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.conn = None
    
    def connect(self, create_db: bool = False):
        """Connect to database
        
        Args:
            create_db: Whether to create database if not exists
        """
        try:
            if create_db:
                # Connect without database to create it
                self.conn = pymysql.connect(
                    host=self.host,
                    port=self.port,
                    user=self.user,
                    password=self.password,
                    charset='utf8mb4'
                )
                self._create_database()
                self.conn.close()
            
            # Connect to database
            self.conn = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                charset='utf8mb4'
            )
            print(f"✅ Connected to {self.host}:{self.port}/{self.database}")
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            raise
    
    def _create_database(self):
        """Create database if not exists"""
        with self.conn.cursor() as cur:
            cur.execute(f"CREATE DATABASE IF NOT EXISTS `{self.database}` DEFAULT CHARSET=utf8mb4")
            self.conn.commit()
    
    def create_table(self):
        """Create cad_vectors table"""
        if not self.conn:
            raise ValueError("Not connected to database")
        
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS `cad_vectors` (
            `id` VARCHAR(255) PRIMARY KEY COMMENT 'Unique identifier (subset/filename)',
            `file_path` TEXT NOT NULL COMMENT 'Absolute path to h5 file',
            `subset` VARCHAR(50) NOT NULL COMMENT 'Subset directory (e.g., 0000, 0001)',
            `seq_len` INT NOT NULL COMMENT 'Sequence length of macro vector',
            `label` VARCHAR(100) DEFAULT NULL COMMENT 'Optional label',
            `source` VARCHAR(50) DEFAULT 'WHUCAD' COMMENT 'Data source',
            `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Import timestamp',
            INDEX `idx_subset` (`subset`),
            INDEX `idx_seq_len` (`seq_len`),
            INDEX `idx_label` (`label`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='CAD vector metadata table'
        """
        
        with self.conn.cursor() as cur:
            cur.execute(create_table_sql)
            self.conn.commit()
        
        print("✅ Table `cad_vectors` created/verified")
    
    def import_metadata(self, metadata: List[Dict], batch_size: int = 1000) -> Dict:
        """Import metadata records
        
        Args:
            metadata: List of metadata dicts
            batch_size: Batch size for insertion
            
        Returns:
            stats: Import statistics
        """
        if not self.conn:
            raise ValueError("Not connected to database")
        
        total = len(metadata)
        inserted = 0
        updated = 0
        
        insert_sql = """
        INSERT INTO `cad_vectors` (`id`, `file_path`, `subset`, `seq_len`, `label`, `source`)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            file_path = VALUES(file_path),
            seq_len = VALUES(seq_len),
            label = VALUES(label)
        """
        
        with self.conn.cursor() as cur:
            for i in range(0, total, batch_size):
                batch = metadata[i:i+batch_size]
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
                
                cur.executemany(insert_sql, values)
                inserted += cur.rowcount
                
                if (i + batch_size) % 10000 == 0:
                    print(f"  Processed {min(i + batch_size, total)}/{total} records...")
            
            self.conn.commit()
        
        stats = {
            'total': total,
            'inserted': inserted,
            'status': 'success'
        }
        
        print(f"✅ Imported {total} records")
        return stats
    
    def query(self, 
             subset: Optional[str] = None,
             min_seq_len: Optional[int] = None,
             max_seq_len: Optional[int] = None,
             label: Optional[str] = None,
             limit: int = 1000) -> List[Dict]:
        """Query metadata with filters
        
        Args:
            subset: Filter by subset
            min_seq_len: Minimum sequence length
            max_seq_len: Maximum sequence length
            label: Filter by label
            limit: Maximum results
            
        Returns:
            records: List of matching records
        """
        if not self.conn:
            raise ValueError("Not connected to database")
        
        conditions = []
        params = []
        
        if subset:
            if isinstance(subset, list):
                placeholders = ','.join(['%s'] * len(subset))
                conditions.append(f"`subset` IN ({placeholders})")
                params.extend(subset)
            else:
                conditions.append("`subset` = %s")
                params.append(subset)
        
        if min_seq_len is not None:
            conditions.append("`seq_len` >= %s")
            params.append(min_seq_len)
        
        if max_seq_len is not None:
            conditions.append("`seq_len` <= %s")
            params.append(max_seq_len)
        
        if label:
            conditions.append("`label` = %s")
            params.append(label)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        query_sql = f"""
        SELECT `id`, `file_path`, `subset`, `seq_len`, `label`, `source`
        FROM `cad_vectors`
        WHERE {where_clause}
        LIMIT %s
        """
        params.append(limit)
        
        with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(query_sql, params)
            results = cur.fetchall()
        
        return results
    
    def get_stats(self) -> Dict:
        """Get database statistics
        
        Returns:
            stats: Statistics dict
        """
        if not self.conn:
            raise ValueError("Not connected to database")
        
        with self.conn.cursor() as cur:
            # Total count
            cur.execute("SELECT COUNT(*) FROM `cad_vectors`")
            total_count = cur.fetchone()[0]
            
            # Subset count
            cur.execute("SELECT COUNT(DISTINCT `subset`) FROM `cad_vectors`")
            subset_count = cur.fetchone()[0]
            
            # Sequence length stats
            cur.execute("SELECT MIN(`seq_len`), MAX(`seq_len`), AVG(`seq_len`) FROM `cad_vectors`")
            min_len, max_len, avg_len = cur.fetchone()
        
        return {
            'total_vectors': total_count,
            'unique_subsets': subset_count,
            'min_seq_len': min_len,
            'max_seq_len': max_len,
            'avg_seq_len': float(avg_len) if avg_len else 0
        }
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


def import_from_json(db_config: Dict, json_file: str, batch_size: int = 1000):
    """Import metadata from JSON file
    
    Args:
        db_config: Database configuration dict
        json_file: Path to metadata JSON file
        batch_size: Batch size for import
    """
    # Load JSON
    print(f"Loading {json_file}...")
    with open(json_file, 'r') as f:
        metadata = json.load(f)
    print(f"Loaded {len(metadata)} records")
    
    # Connect and import
    db = MetadataDB(**db_config)
    db.connect(create_db=True)
    db.create_table()
    stats = db.import_metadata(metadata, batch_size)
    db.close()
    
    return stats
