"""Example: Using Semantic Search API

This script demonstrates how to use the FastAPI semantic search endpoints.

Prerequisites:
1. Start the API server:
   python server/app.py

2. Install httpx:
   pip install httpx
"""

import httpx
import json
from typing import List, Dict


API_BASE_URL = "http://localhost:8000"


def example_1_semantic_search():
    """Example 1: Basic semantic search via API"""
    print("=" * 80)
    print("Example 1: Semantic Search API")
    print("=" * 80)
    
    # Prepare request
    request_data = {
        "query_text": "cylindrical mechanical part",
        "k": 5,
        "encoder_type": "sentence-transformer",
        "explainable": False
    }
    
    print(f"\nPOST {API_BASE_URL}/search/semantic")
    print(f"Request: {json.dumps(request_data, indent=2)}\n")
    
    # Send request
    with httpx.Client() as client:
        response = client.post(
            f"{API_BASE_URL}/search/semantic",
            json=request_data,
            timeout=30.0
        )
    
    if response.status_code == 200:
        data = response.json()
        print(f"Status: {data['status']}")
        print(f"Query: '{data['query_text']}'")
        print(f"Results: {data['num_results']}\n")
        
        for i, result in enumerate(data['results'], 1):
            print(f"{i}. {result['id']}")
            print(f"   Score: {result['score']:.4f}")
            print(f"   Subset: {result['metadata'].get('subset', 'unknown')}")
            print()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


def example_2_multilingual_queries():
    """Example 2: Multilingual semantic search"""
    print("=" * 80)
    print("Example 2: Multilingual Queries")
    print("=" * 80)
    
    queries = [
        "圆柱形零件",                    # Chinese
        "cylindrical part",             # English
        "找一个机械齿轮",                # Chinese
        "mechanical gear component"     # English
    ]
    
    print("\nTesting multilingual queries...\n")
    
    with httpx.Client() as client:
        for query in queries:
            request_data = {
                "query_text": query,
                "k": 3,
                "encoder_type": "sentence-transformer"
            }
            
            response = client.post(
                f"{API_BASE_URL}/search/semantic",
                json=request_data,
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                if data['results']:
                    top = data['results'][0]
                    print(f"'{query}'")
                    print(f"  → {top['id']} (score: {top['score']:.4f})")
                else:
                    print(f"'{query}' → No results")
            else:
                print(f"'{query}' → Error: {response.status_code}")
            print()


def example_3_explainable_search():
    """Example 3: Semantic search with explanations"""
    print("=" * 80)
    print("Example 3: Explainable Semantic Search")
    print("=" * 80)
    
    request_data = {
        "query_text": "mechanical component with holes",
        "k": 5,
        "encoder_type": "sentence-transformer",
        "explainable": True
    }
    
    print(f"\nQuery: '{request_data['query_text']}'")
    print("Requesting with explainable=true...\n")
    
    with httpx.Client() as client:
        response = client.post(
            f"{API_BASE_URL}/search/semantic",
            json=request_data,
            timeout=30.0
        )
    
    if response.status_code == 200:
        data = response.json()
        
        # Display explanation
        if 'explanation' in data:
            exp = data['explanation']
            print("Explanation:")
            print(f"  Query: {exp['query_text']}")
            print(f"  Top Match: {exp['top_match']['id']}")
            print(f"  Score: {exp['top_match']['score']:.4f}")
            print(f"  Interpretation: {exp['interpretation']}")
            if 'recommendation' in exp:
                print(f"  Recommendation: {exp['recommendation']}")
        
        print(f"\nAll results:")
        for i, res in enumerate(data['results'], 1):
            print(f"  {i}. {res['id']} - {res['score']:.4f}")
    else:
        print(f"Error: {response.status_code}")


def example_4_hybrid_search():
    """Example 4: Hybrid search API"""
    print("=" * 80)
    print("Example 4: Hybrid Search (Text + Vector)")
    print("=" * 80)
    
    request_data = {
        "query_text": "cylindrical part with threads",
        "query_file_path": "data/vec/0000/00000000.h5",  # Optional CAD vector
        "k": 5,
        "semantic_weight": 0.6,
        "vector_weight": 0.4,
        "encoder_type": "sentence-transformer"
    }
    
    print(f"\nQuery Text: '{request_data['query_text']}'")
    print(f"Query File: {request_data.get('query_file_path', 'None')}")
    print(f"Weights: semantic={request_data['semantic_weight']}, vector={request_data['vector_weight']}\n")
    
    with httpx.Client() as client:
        response = client.post(
            f"{API_BASE_URL}/search/hybrid",
            json=request_data,
            timeout=30.0
        )
    
    if response.status_code == 200:
        data = response.json()
        print(f"Status: {data['status']}")
        print(f"Results: {data['num_results']}\n")
        
        for i, res in enumerate(data['results'], 1):
            print(f"{i}. {res['id']}")
            print(f"   Fused Score: {res['score']:.4f}")
            print(f"   Semantic: {res['semantic_score']:.4f}")
            print(f"   Vector: {res['vector_score']:.4f}")
            print()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


def example_5_metadata_filtering():
    """Example 5: Semantic search with metadata filters"""
    print("=" * 80)
    print("Example 5: Semantic Search with Metadata Filtering")
    print("=" * 80)
    
    query_text = "cylindrical part"
    
    # Without filter
    print(f"\nQuery: '{query_text}'")
    print("\n1. Without filters:")
    
    with httpx.Client() as client:
        response = client.post(
            f"{API_BASE_URL}/search/semantic",
            json={
                "query_text": query_text,
                "k": 5,
                "encoder_type": "sentence-transformer"
            },
            timeout=30.0
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Found {data['num_results']} results")
            for res in data['results'][:3]:
                print(f"   - {res['id']} (subset: {res['metadata'].get('subset', 'unknown')})")
    
    # With filter
    print("\n2. With subset filter (subset='0000'):")
    
    with httpx.Client() as client:
        response = client.post(
            f"{API_BASE_URL}/search/semantic",
            json={
                "query_text": query_text,
                "k": 5,
                "encoder_type": "sentence-transformer",
                "filters": {"subset": "0000"}
            },
            timeout=30.0
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Found {data['num_results']} results")
            for res in data['results'][:3]:
                print(f"   - {res['id']} (subset: {res['metadata'].get('subset', 'unknown')})")


def example_6_batch_queries():
    """Example 6: Batch semantic queries"""
    print("=" * 80)
    print("Example 6: Batch Semantic Queries")
    print("=" * 80)
    
    queries = [
        "圆形零件",
        "cylindrical part",
        "mechanical gear",
        "threaded shaft",
        "component with holes"
    ]
    
    print(f"\nProcessing {len(queries)} queries...\n")
    
    with httpx.Client() as client:
        for query in queries:
            response = client.post(
                f"{API_BASE_URL}/search/semantic",
                json={
                    "query_text": query,
                    "k": 1,
                    "encoder_type": "sentence-transformer"
                },
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                if data['results']:
                    top = data['results'][0]
                    print(f"'{query}' → {top['id']} ({top['score']:.4f})")
                else:
                    print(f"'{query}' → No results")
            else:
                print(f"'{query}' → Error: {response.status_code}")


def check_api_health():
    """Check if API server is running"""
    print("Checking API server...")
    try:
        with httpx.Client() as client:
            response = client.get(f"{API_BASE_URL}/", timeout=5.0)
            if response.status_code == 200:
                print(f"✓ API server is running at {API_BASE_URL}")
                data = response.json()
                print(f"  Version: {data.get('version', 'unknown')}")
                return True
            else:
                print(f"✗ API server returned status {response.status_code}")
                return False
    except httpx.ConnectError:
        print(f"✗ Cannot connect to API server at {API_BASE_URL}")
        print("  Please start the server: python server/app.py")
        return False
    except Exception as e:
        print(f"✗ Error checking API: {e}")
        return False


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 22 + "Semantic Search API Examples" + " " * 28 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")
    
    # Check if API is running
    if not check_api_health():
        print("\nPlease start the API server first:")
        print("  cd /Users/he.tian/bs/db")
        print("  python server/app.py")
        exit(1)
    
    print("\n")
    
    # Run examples
    try:
        example_1_semantic_search()
        input("\nPress Enter to continue to Example 2...\n")
        
        example_2_multilingual_queries()
        input("\nPress Enter to continue to Example 3...\n")
        
        example_3_explainable_search()
        input("\nPress Enter to continue to Example 4...\n")
        
        example_4_hybrid_search()
        input("\nPress Enter to continue to Example 5...\n")
        
        example_5_metadata_filtering()
        input("\nPress Enter to continue to Example 6...\n")
        
        example_6_batch_queries()
        
        print("\n" + "=" * 80)
        print("All API examples completed!")
        print("=" * 80 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user.")
    except Exception as e:
        print(f"\n\nError running examples: {e}")
        import traceback
        traceback.print_exc()
