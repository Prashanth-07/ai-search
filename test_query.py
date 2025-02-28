import requests
import json
from datetime import datetime
import time
import re

def print_separator(char="=", length=70):
    print(f"\n{char * length}\n")

def extract_match_details(log_lines):
    """Extract full match details from log lines"""
    matches = []
    current_match = {}
    
    for line in log_lines:
        if 'Match' in line and ':' in line and not any(detail in line for detail in ['Tool:', 'Tool ID:', 'Vector ID:', 'Score:']):
            if current_match:
                matches.append(current_match)
            current_match = {}
        elif 'Tool:' in line:
            current_match['tool'] = line.split('Tool:')[-1].strip()
        elif 'Tool ID:' in line:
            current_match['tool_id'] = line.split('Tool ID:')[-1].strip()
        elif 'Vector ID:' in line:
            current_match['vector_id'] = line.split('Vector ID:')[-1].strip()
        elif 'Score:' in line:
            current_match['score'] = line.split('Score:')[-1].strip()
    
    if current_match:
        matches.append(current_match)
    
    return matches

def show_retriever_output(query: str):
    """Get and show retriever output for the query"""
    base_url = "http://localhost:8000"
    
    try:
        # Make query request
        response = requests.post(
            f"{base_url}/query",
            json={"query": query}
        )
        
        if response.status_code == 200:
            # Get vector store logs
            try:
                with open('vectorstore.log', 'r') as f:
                    log_content = f.read()
                
                # Find the relevant log section for this query
                query_section = None
                sections = log_content.split("Processing query:")
                for section in sections:
                    if query in section:
                        query_section = section
                        break
                
                if query_section:
                    print("\nRETRIEVER OUTPUT:")
                    print_separator("-")
                    
                    # Extract number of retrieved documents
                    docs_match = re.search(r"Retrieved (\d+) relevant documents", query_section)
                    if docs_match:
                        print(f"Retrieved {docs_match.group(1)} relevant documents")
                    
                    # Extract match details
                    matches = extract_match_details(query_section.split('\n'))
                    
                    if matches:
                        for i, match in enumerate(matches, 1):
                            print(f"\nMatch {i}:")
                            if 'tool' in match:
                                print(f"  Tool: {match['tool']}")
                            if 'tool_id' in match:
                                print(f"  Tool ID: {match['tool_id']}")
                            if 'vector_id' in match:
                                print(f"  Vector ID: {match['vector_id']}")
                            if 'score' in match:
                                print(f"  Score: {match['score']}")
                    else:
                        print("No detailed match information found in logs")
                    
                    print_separator("-")
                    return response.json()
            except Exception as e:
                print(f"\nError reading retriever logs: {str(e)}")
                return None
    except Exception as e:
        print(f"Error getting retriever output: {str(e)}")
        return None

def show_llm_response(data):
    """Show LLM response and analysis"""
    if data:
        print("\nLLM RESPONSE:")
        print_separator("-")
        print(data['response'])
        print_separator("-")
        
        # Response analysis
        response_length = len(data['response'])
        print("\nResponse Analysis:")
        print(f"Character count: {response_length}")
        print(f"Word count: {len(data['response'].split())}")
        print(f"Line count: {len(data['response'].splitlines())}")
        return True
    return False

def test_query(query: str):
    """Test a single query showing retriever output first, then LLM response"""
    print_separator()
    print(f"QUERY TEST: '{query}'")
    print(f"Timestamp: [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
    print_separator()
    
    # First show retriever output
    data = show_retriever_output(query)
    
    if data:
        # Wait for user to review retriever output
        input("\nPress Enter to see LLM response...")
        show_llm_response(data)

def main():
    print("\n=== RAG System Query Tester ===")
    print("Enter your queries to test the system")
    print("Commands:")
    print("  'exit' or 'quit' - Exit the program")
    print("  'clear' - Clear the screen")
    
    while True:
        try:
            print_separator("-")
            query = input("Enter your query: ").strip()
            
            if not query:
                continue
                
            if query.lower() in ['exit', 'quit']:
                print("\nExiting query tester...")
                break
                
            if query.lower() == 'clear':
                import os
                os.system('cls' if os.name == 'nt' else 'clear')
                continue
            
            test_query(query)
            
        except KeyboardInterrupt:
            print("\nExiting query tester...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()