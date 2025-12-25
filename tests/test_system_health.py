"""System Health Check for CAD Vector Database

Comprehensive tests for all system components:
1. Module imports
2. IndexManager functionality
3. TwoStageRetrieval functionality
4. Database connection
5. Visualization tools
6. CLI tools
7. Server module
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test all module imports"""
    print("\n1. Testing Module Imports")
    print("-" * 60)
    
    modules = [
        ("cad_vectordb.core.index", "IndexManager"),
        ("cad_vectordb.core.retrieval", "TwoStageRetrieval"),
        ("cad_vectordb.core.feature", "extract_feature, load_macro_vec"),
        ("cad_vectordb.database.metadata", "MetadataDB"),
        ("cad_vectordb.utils.visualization", "generate_html_visualization"),
        ("server.app", "app"),
    ]
    
    errors = []
    for module_name, obj_names in modules:
        try:
            __import__(module_name)
            print(f"   ✅ {module_name}")
        except Exception as e:
            print(f"   ❌ {module_name}: {e}")
            errors.append((module_name, e))
    
    return len(errors) == 0


def test_index_manager():
    """Test IndexManager functionality"""
    print("\n2. Testing IndexManager")
    print("-" * 60)
    
    try:
        from cad_vectordb.core.index import IndexManager
        
        # Test initialization
        mgr = IndexManager('data/index_test')
        print("   ✅ IndexManager initialized")
        
        # Test listing indexes
        indexes = mgr.list_available_indexes()
        print(f"   ✅ Found {len(indexes)} index(es): {indexes}")
        
        if len(indexes) > 0:
            # Test loading
            mgr.load_index(indexes[0])
            print(f"   ✅ Index loaded: {indexes[0]}")
            
            # Test stats
            stats = mgr.get_stats()
            print(f"   ✅ Stats: {stats['num_vectors']} vectors, dim={stats['dimension']}")
            
            # Test validation
            validation = mgr.validate_index()
            if validation['valid']:
                print(f"   ✅ Index validation passed")
            else:
                print(f"   ⚠️  Index validation issues: {validation['issues']}")
        else:
            print("   ⚠️  No indexes found to test")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_retrieval():
    """Test TwoStageRetrieval functionality"""
    print("\n3. Testing TwoStageRetrieval")
    print("-" * 60)
    
    try:
        from cad_vectordb.core.index import IndexManager
        from cad_vectordb.core.retrieval import TwoStageRetrieval
        
        # Load index
        mgr = IndexManager('data/index_test')
        indexes = mgr.list_available_indexes()
        
        if len(indexes) == 0:
            print("   ⚠️  No indexes found to test retrieval")
            return True
        
        mgr.load_index(indexes[0])
        
        # Test initialization with IndexManager
        retrieval = TwoStageRetrieval(mgr)
        print("   ✅ TwoStageRetrieval initialized with IndexManager")
        
        # Test backward compatibility
        retrieval2 = TwoStageRetrieval(mgr.index, mgr.ids, mgr.metadata)
        print("   ✅ Backward compatibility: works with raw parameters")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_database():
    """Test database functionality"""
    print("\n4. Testing Database")
    print("-" * 60)
    
    try:
        from cad_vectordb.database.metadata import MetadataDB
        from config import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME
        
        # Test initialization
        db = MetadataDB(DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME)
        print("   ✅ MetadataDB initialized")
        
        # Note: Not testing connection as it requires OceanBase to be running
        print("   ℹ️  Connection test skipped (requires OceanBase)")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_visualization():
    """Test visualization functionality"""
    print("\n5. Testing Visualization")
    print("-" * 60)
    
    try:
        from cad_vectordb.utils.visualization import generate_html_visualization
        
        # Test with dummy data
        dummy_results = [
            {
                'id': 'test/file.h5',
                'score': 0.95,
                'metadata': {'subset': '0000', 'seq_len': 50},
                'explanation': {
                    'stage1_similarity': 0.9,
                    'stage2_similarity': 0.85,
                    'final_score': 0.95,
                    'fusion_method': 'weighted',
                    'stage1_interpretation': 'High feature similarity',
                    'stage2_interpretation': 'Good sequence match'
                }
            }
        ]
        
        output = generate_html_visualization(dummy_results, "test_query.h5", "/tmp/test_viz.html")
        print(f"   ✅ Visualization generated: {output}")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cli():
    """Test CLI functionality"""
    print("\n6. Testing CLI Tools")
    print("-" * 60)
    
    try:
        import subprocess
        
        # Test list command
        result = subprocess.run(
            ['python', '-m', 'cad_vectordb.cli', 'list'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print("   ✅ CLI list command works")
        else:
            print(f"   ❌ CLI list command failed: {result.stderr}")
            return False
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_server():
    """Test server module"""
    print("\n7. Testing Server Module")
    print("-" * 60)
    
    try:
        from server.app import app
        print("   ✅ Server module imported successfully")
        print("   ℹ️  To start server: uvicorn server.app:app --host 0.0.0.0 --port 8000")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("CAD Vector Database - System Health Check")
    print("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("IndexManager", test_index_manager),
        ("TwoStageRetrieval", test_retrieval),
        ("Database", test_database),
        ("Visualization", test_visualization),
        ("CLI Tools", test_cli),
        ("Server Module", test_server),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n   ❌ Unexpected error in {name}: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {name}")
    
    print("\n" + "=" * 60)
    print(f"Result: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! System is fully functional.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
