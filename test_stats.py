import requests
import json
from datetime import datetime
import time

def print_separator(char="=", length=70):
    print(f"\n{char * length}\n")

def check_server_health():
    """Check if the server is running and healthy"""
    base_url = "http://localhost:8000"
    try:
        health_response = requests.get(f"{base_url}/health")
        if health_response.status_code == 200:
            return True, "Server is healthy"
        return False, f"Health check failed: {health_response.text}"
    except Exception as e:
        return False, f"Error checking server health: {str(e)}"

def format_vector_info(vector):
    """Format vector information for display"""
    return f"""
  Name: {vector['name']}
  Tool ID: {vector['tool_id']}
  Vector ID: {vector['vector_id']}
  Description: {vector['description']}
  Categories: {vector['categories']}
  Pricing: {vector['pricing']}"""

def test_stats():
    """Test the /stats endpoint with detailed vector information"""
    base_url = "http://localhost:8000"
    
    print_separator()
    print("Testing Stats Endpoint")
    print(f"Timestamp: [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
    print_separator()
    
    # Check server health
    is_healthy, health_message = check_server_health()
    print(f"Server Health: {'✓' if is_healthy else '✗'} - {health_message}")
    
    if not is_healthy:
        print("\nServer is not healthy. Please check if it's running properly.")
        return False, None
    
    try:
        print("\nFetching vector store statistics...")
        stats_response = requests.get(f"{base_url}/stats")
        print(f"Status Code: {stats_response.status_code}")
        
        if stats_response.status_code == 200:
            stats = stats_response.json()
            
            print("\nVECTOR STORE STATISTICS:")
            print_separator("-")
            print(f"Total Vectors: {stats['total_vectors']}")
            print(f"Dimension: {stats['dimension']}")
            print(f"Index Fullness: {stats['index_fullness']}")
            
            if stats['namespaces']:
                print("\nNamespaces:")
                for namespace, details in stats['namespaces'].items():
                    print(f"  {namespace}: {details}")
            
            # Display vector details
            if 'vectors' in stats and stats['vectors']:
                print("\nSTORED VECTORS:")
                print_separator("-")
                for i, vector in enumerate(stats['vectors'], 1):
                    print(f"\nVector {i}:{format_vector_info(vector)}")
            else:
                print("\nNo vectors found in the store.")
            
            return True, stats
        else:
            print(f"\nError Response: {stats_response.text}")
            return False, None
            
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        return False, None

def main():
    print("\n=== Vector Store Stats Tester ===")
    success, stats = test_stats()
    print_separator()
    print(f"Test {'passed' if success else 'failed'}")

if __name__ == "__main__":
    main()