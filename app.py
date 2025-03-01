import streamlit as st
import requests
import json
import pandas as pd
from io import StringIO
import time

# Set page configuration
st.set_page_config(
    page_title="AI Tool Search Interface",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.5rem;
        font-weight: bold;
        color: #333;
        margin-bottom: 1rem;
    }
    .search-results {
        color: #FF5252;
        font-size: 1.8rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .tool-result {
        background-color: #1A1E23;
        border-left: 4px solid #1E88E5;
        padding: 1rem;
        margin-bottom: 0.5rem;
        border-radius: 0.3rem;
    }
    .tool-title {
        font-size: 1.3rem;
        font-weight: bold;
        color: #4FC3F7;
        margin-bottom: 0.5rem;
    }
    .tool-id {
        font-size: 0.9rem;
        color: #B0BEC5;
        background-color: #263238;
        padding: 0.2rem 0.5rem;
        border-radius: 0.2rem;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    .tool-description {
        color: #E0E0E0;
        font-size: 1rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .success-banner {
        padding: 1rem;
        border-radius: 0.3rem;
        background-color: #d4edda;
        color: #155724;
        margin-bottom: 1rem;
    }
    .error-banner {
        padding: 1rem;
        border-radius: 0.3rem;
        background-color: #f8d7da;
        color: #721c24;
        margin-bottom: 1rem;
    }
    .tool-card {
        padding: 1rem;
        border-radius: 0.3rem;
        background-color: white;
        border-left: 4px solid #1E88E5;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre-wrap;
        border-radius: 4px 4px 0 0;
        gap: 1rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'api_url' not in st.session_state:
    st.session_state.api_url = "http://localhost:8000"
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""
if 'last_result' not in st.session_state:
    st.session_state.last_result = None
if 'fetched_tool' not in st.session_state:
    st.session_state.fetched_tool = None

# Sidebar for configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    st.session_state.api_url = st.text_input("API URL", value=st.session_state.api_url)
    st.session_state.api_key = st.text_input("Pinecone API Key (for admin functions)", 
                                          value=st.session_state.api_key, 
                                          type="password")
    
    # Model selection
    st.divider()
    st.subheader("Model Selection")
    
    # Initialize model choice if not in session state
    if 'model_choice' not in st.session_state:
        st.session_state.model_choice = "DEV_MODEL"
    
    # Radio button for model selection
    st.session_state.model_choice = st.radio(
        "Select Model Environment Variable:",
        options=["DEV_MODEL", "PROD_MODEL"],
        index=0 if st.session_state.model_choice == "DEV_MODEL" else 1,
        horizontal=True
    )
    
    # Display current model information
    if st.session_state.model_choice == "DEV_MODEL":
        st.info("Using DEV_MODEL from .env file (deepseek-r1:1.5b)")
    else:
        st.info("Using PROD_MODEL from .env file (deepseek-r1:7b)")
        
    st.caption("Note: This will tell the backend which environment variable to use for the model selection.")
    
    st.divider()
    st.markdown("### About")
    st.markdown("""
    This interface allows you to interact with the AI Tool Search API.
    
    You can:
    - Add new AI tools (single or bulk)
    - Search for tools based on queries
    - Update existing tools
    - Delete tools
    - View statistics
    """)
    
    st.divider()
    if st.button("Check API Health"):
        try:
            # Prepare model selection to send
            headers = {}
            headers["MODEL_CHOICE"] = st.session_state.model_choice
                
            response = requests.get(
                f"{st.session_state.api_url}/health", 
                headers=headers
            )
            if response.status_code == 200:
                st.success("API is healthy! ✅")
                st.json(response.json())
            else:
                st.error(f"API returned status code: {response.status_code}")
        except Exception as e:
            st.error(f"Error connecting to API: {str(e)}")

# Main content
st.markdown('<div class="main-header">AI Tool Search</div>', unsafe_allow_html=True)

# Create tabs for different functionalities
tabs = st.tabs(["🔍 Search", "➕ Add Tools", "🔄 Update Tools", "🗑️ Delete Tools", "📊 Statistics"])

# 1. SEARCH TAB
with tabs[0]:
    st.markdown('<div class="subheader">Search AI Tools</div>', unsafe_allow_html=True)
    
    query = st.text_input("Enter your search query:", 
                          placeholder="e.g., code generation tools for JavaScript",
                          value=st.session_state.last_query)
    
    if st.button("Search", type="primary", key="search_button"):
        st.session_state.last_query = query
        with st.spinner("Searching..."):
            try:
                # Prepare environment variables to send
                headers = {}
                headers["MODEL_CHOICE"] = st.session_state.model_choice
                
                response = requests.post(
                    f"{st.session_state.api_url}/query",
                    json={"query": query},
                    headers=headers
                )
                
                if response.status_code == 200:
                    result = response.json()
                    st.session_state.last_result = result
                    
                    # Parse the response JSON string into a Python object
                    try:
                        result_data = json.loads(result["response"])
                        
                        # Display the tools in a visually appealing way
                        st.markdown('<div class="search-results">Search Results</div>', unsafe_allow_html=True)
                        
                        if "tools" in result_data and len(result_data["tools"]) > 0:
                            for tool in result_data["tools"]:
                                st.markdown(f"""
                                <div class="tool-result">
                                    <div class="tool-title">{tool.get('name', 'No Name')}</div>
                                    <div class="tool-id">ID: {tool.get('id', 'No ID')}</div>
                                    <div class="tool-description">{tool.get('description', 'No description available.')}</div>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No matching tools found.")
                            
                        # Show raw response in an expander
                        with st.expander("View Raw JSON Response"):
                            st.json(result_data)
                    except json.JSONDecodeError:
                        st.error("Failed to parse response JSON.")
                        st.text(result["response"])
                else:
                    st.error(f"Error: API returned status code {response.status_code}")
                    st.text(response.text)
            except Exception as e:
                st.error(f"Error connecting to API: {str(e)}")

# 2. ADD TOOLS TAB
with tabs[1]:
    st.markdown('<div class="subheader">Add New AI Tools</div>', unsafe_allow_html=True)
    
    add_option = st.radio("Choose an option:", ["Add Single Tool", "Bulk Upload"])
    
    if add_option == "Add Single Tool":
        with st.form(key="add_tool_form"):
            st.markdown("### Tool Information")
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Tool Name*", help="The official name of the AI tool")
                tool_id = st.text_input("Tool ID*", help="A unique identifier for this tool (e.g., tool-name-001)")
                categories = st.text_input("Categories", help="Comma-separated list of categories (e.g., Text Generation, Code Assistance)")
                pricing = st.text_input("Pricing", help="Information about pricing tiers (e.g., Free, Freemium, $10/month)")
            
            with col2:
                description = st.text_area("Description*", help="A brief description of what the tool does")
                
                pros_input = st.text_area("Pros (one per line)", 
                                     help="List the advantages of this tool, one per line")
                cons_input = st.text_area("Cons (one per line)", 
                                     help="List limitations or disadvantages, one per line")
            
            st.markdown("### Additional Details")
            usage = st.text_area("Usage Examples", help="How this tool can be used effectively")
            unique_features = st.text_area("Unique Features", help="What makes this tool stand out from others")
            
            submit_button = st.form_submit_button("Add Tool", type="primary")
            
            if submit_button:
                if not name or not tool_id or not description:
                    st.error("Please fill in all required fields (marked with *).")
                else:
                    # Process pros and cons lists
                    pros = [p.strip() for p in pros_input.split('\n') if p.strip()]
                    cons = [c.strip() for c in cons_input.split('\n') if c.strip()]
                    
                    # Create the tool object
                    tool = {
                        "name": name,
                        "tool_id": tool_id,
                        "description": description,
                        "pros": pros,
                        "cons": cons,
                        "categories": categories,
                        "usage": usage,
                        "unique_features": unique_features,
                        "pricing": pricing
                    }
                    
                    # Create the request payload
                    payload = {"tools": [tool]}
                    
                    try:
                        with st.spinner("Adding tool..."):
                            # Prepare environment variables to send
                            headers = {}
                            if st.session_state.environment == "DEV":
                                headers["ENVIRONMENT"] = "DEV"
                                headers["DEV_MODEL"] = st.session_state.dev_model
                            else:
                                headers["ENVIRONMENT"] = "PROD"
                                headers["PROD_MODEL"] = st.session_state.prod_model
                                
                            response = requests.post(
                                f"{st.session_state.api_url}/add-tools",
                                json=payload,
                                headers=headers
                            )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"Tool '{name}' added successfully!")
                            
                            with st.expander("View Details"):
                                st.json(result)
                        else:
                            st.error(f"Error: API returned status code {response.status_code}")
                            st.text(response.text)
                    except Exception as e:
                        st.error(f"Error connecting to API: {str(e)}")
    
    else:  # Bulk Upload
        st.markdown("### Bulk Upload Tools")
        
        st.info("""
        Upload a JSON file with multiple tools. The file should have this structure:
        ```json
        {
            "tools": [
                {
                    "name": "Tool Name",
                    "tool_id": "tool-name-001",
                    "description": "Tool description",
                    "pros": ["Pro 1", "Pro 2"],
                    "cons": ["Con 1", "Con 2"],
                    "categories": "Category1, Category2",
                    "usage": "Usage examples",
                    "unique_features": "What makes this tool unique",
                    "pricing": "Pricing information"
                },
                // More tools...
            ]
        }
        ```
        """)
        
        uploaded_file = st.file_uploader("Upload JSON file", type="json")
        
        if uploaded_file is not None:
            try:
                # Load JSON data
                data = json.load(uploaded_file)
                
                # Preview the data
                with st.expander("Preview Upload Data"):
                    st.write(f"Found {len(data.get('tools', []))} tools in the uploaded file.")
                    st.json(data)
                
                if st.button("Process Bulk Upload", type="primary"):
                    with st.spinner("Uploading tools..."):
                        try:
                            # Prepare environment variables to send
                            headers = {}
                            if st.session_state.environment == "DEV":
                                headers["ENVIRONMENT"] = "DEV"
                                headers["DEV_MODEL"] = st.session_state.dev_model
                            else:
                                headers["ENVIRONMENT"] = "PROD"
                                headers["PROD_MODEL"] = st.session_state.prod_model
                                
                            response = requests.post(
                                f"{st.session_state.api_url}/add-tools",
                                json=data,
                                headers=headers
                            )
                            
                            if response.status_code == 200:
                                result = response.json()
                                st.success(f"Successfully processed {len(result['results'])} tools!")
                                
                                # Show results in a table
                                results_data = []
                                for item in result["results"]:
                                    results_data.append({
                                        "Name": item["tool"]["name"],
                                        "ID": item["tool"]["tool_id"],
                                        "Status": item["status"]
                                    })
                                
                                results_df = pd.DataFrame(results_data)
                                st.dataframe(results_df)
                                
                                with st.expander("View Full Response"):
                                    st.json(result)
                            else:
                                st.error(f"Error: API returned status code {response.status_code}")
                                st.text(response.text)
                        except Exception as e:
                            st.error(f"Error connecting to API: {str(e)}")
                
            except json.JSONDecodeError:
                st.error("Invalid JSON file. Please check the format.")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

# 3. UPDATE TOOLS TAB
with tabs[2]:
    st.markdown('<div class="subheader">Update Existing Tools</div>', unsafe_allow_html=True)
    
    # Step 1: Input tool ID
    tool_id_to_update = st.text_input("Enter Tool ID to update:", 
                                      key="update_tool_id_input",
                                      help="Enter the unique identifier of the tool you want to update")
    
    col1, col2 = st.columns([1, 5])
    with col1:
        fetch_button = st.button("Fetch Tool", key="fetch_tool_button")
    
    if fetch_button and tool_id_to_update:
        with st.spinner("Fetching tool data..."):
            # In a real implementation, you would have an endpoint to fetch a single tool
            # For now, we'll simulate fetching by querying with the tool ID
            try:
                # Prepare environment variables to send
                headers = {}
                if st.session_state.environment == "DEV":
                    headers["ENVIRONMENT"] = "DEV"
                    headers["DEV_MODEL"] = st.session_state.dev_model
                else:
                    headers["ENVIRONMENT"] = "PROD"
                    headers["PROD_MODEL"] = st.session_state.prod_model
                    
                response = requests.post(
                    f"{st.session_state.api_url}/query",
                    json={"query": f"tool_id:{tool_id_to_update}"},
                    headers=headers
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    try:
                        result_data = json.loads(result["response"])
                        
                        if "tools" in result_data and len(result_data["tools"]) > 0:
                            # Find the matching tool
                            matching_tool = None
                            for tool in result_data["tools"]:
                                if tool.get("id") == tool_id_to_update:
                                    matching_tool = tool
                                    break
                            
                            if matching_tool:
                                st.session_state.fetched_tool = matching_tool
                                st.success(f"Found tool: {matching_tool.get('name', 'Unnamed Tool')}")
                            else:
                                st.warning(f"Tool with ID '{tool_id_to_update}' not found in search results.")
                                st.session_state.fetched_tool = None
                        else:
                            st.warning(f"No tool found with ID: {tool_id_to_update}")
                            st.session_state.fetched_tool = None
                    except json.JSONDecodeError:
                        st.error("Failed to parse response JSON.")
                        st.session_state.fetched_tool = None
                else:
                    st.error(f"Error: API returned status code {response.status_code}")
                    st.session_state.fetched_tool = None
            except Exception as e:
                st.error(f"Error connecting to API: {str(e)}")
                st.session_state.fetched_tool = None
    
    # Step 2: If a tool was fetched, show the update form
    if st.session_state.fetched_tool:
        with st.form(key="update_tool_form"):
            st.markdown("### Update Tool Information")
            col1, col2 = st.columns(2)
            
            # Pre-fill form with existing data
            tool = st.session_state.fetched_tool
            
            with col1:
                name = st.text_input("Tool Name*", 
                                    value=tool.get("name", ""),
                                    help="The official name of the AI tool")
                tool_id = st.text_input("Tool ID*", 
                                       value=tool.get("id", ""),
                                       help="A unique identifier for this tool",
                                       disabled=True)
                categories = st.text_input("Categories", 
                                         value=tool.get("categories", ""),
                                         help="Comma-separated list of categories")
                pricing = st.text_input("Pricing", 
                                      value=tool.get("pricing", ""),
                                      help="Information about pricing tiers")
            
            with col2:
                description = st.text_area("Description*", 
                                         value=tool.get("description", ""),
                                         help="A brief description of what the tool does")
                
                # Join pros and cons with newlines for the text area
                pros_text = "\n".join(tool.get("pros", []))
                cons_text = "\n".join(tool.get("cons", []))
                
                pros_input = st.text_area("Pros (one per line)", 
                                        value=pros_text,
                                        help="List the advantages of this tool, one per line")
                cons_input = st.text_area("Cons (one per line)", 
                                        value=cons_text,
                                        help="List limitations or disadvantages, one per line")
            
            st.markdown("### Additional Details")
            usage = st.text_area("Usage Examples", 
                               value=tool.get("usage", ""),
                               help="How this tool can be used effectively")
            unique_features = st.text_area("Unique Features", 
                                         value=tool.get("unique_features", ""),
                                         help="What makes this tool stand out from others")
            
            update_button = st.form_submit_button("Update Tool", type="primary")
            
            if update_button:
                if not name or not tool_id or not description:
                    st.error("Please fill in all required fields (marked with *).")
                else:
                    # Process pros and cons lists
                    pros = [p.strip() for p in pros_input.split('\n') if p.strip()]
                    cons = [c.strip() for c in cons_input.split('\n') if c.strip()]
                    
                    # Create the updated tool object
                    updated_tool = {
                        "name": name,
                        "tool_id": tool_id,
                        "description": description,
                        "pros": pros,
                        "cons": cons,
                        "categories": categories,
                        "usage": usage,
                        "unique_features": unique_features,
                        "pricing": pricing
                    }
                    
                    # Create the request payload
                    payload = {"tools": [updated_tool]}
                    
                    try:
                        with st.spinner("Updating tool..."):
                            # Prepare environment variables to send
                            headers = {}
                            if st.session_state.environment == "DEV":
                                headers["ENVIRONMENT"] = "DEV"
                                headers["DEV_MODEL"] = st.session_state.dev_model
                            else:
                                headers["ENVIRONMENT"] = "PROD"
                                headers["PROD_MODEL"] = st.session_state.prod_model
                                
                            response = requests.put(
                                f"{st.session_state.api_url}/update-tools",
                                json=payload,
                                headers=headers
                            )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"Tool '{name}' updated successfully!")
                            
                            with st.expander("View Details"):
                                st.json(result)
                                
                            # Reset the fetched tool to show the form is complete
                            st.session_state.fetched_tool = None
                            st.rerun()
                        else:
                            st.error(f"Error: API returned status code {response.status_code}")
                            st.text(response.text)
                    except Exception as e:
                        st.error(f"Error connecting to API: {str(e)}")

# 4. DELETE TOOLS TAB
with tabs[3]:
    st.markdown('<div class="subheader">Delete Tools</div>', unsafe_allow_html=True)
    
    st.warning("⚠️ Warning: Deletion is permanent and cannot be undone.")
    
    tool_id_to_delete = st.text_input("Enter Tool ID to delete:", 
                                     key="delete_tool_id_input",
                                     help="Enter the unique identifier of the tool you want to delete")
    
    confirm_delete = st.checkbox("I confirm that I want to delete this tool permanently")
    
    if st.button("Delete Tool", type="primary", disabled=not confirm_delete or not tool_id_to_delete):
        with st.spinner("Deleting tool..."):
            try:
                # Prepare environment variables to send
                headers = {}
                if st.session_state.environment == "DEV":
                    headers["ENVIRONMENT"] = "DEV"
                    headers["DEV_MODEL"] = st.session_state.dev_model
                else:
                    headers["ENVIRONMENT"] = "PROD"
                    headers["PROD_MODEL"] = st.session_state.prod_model
                    
                response = requests.delete(
                    f"{st.session_state.api_url}/delete-tool/{tool_id_to_delete}",
                    headers=headers
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        st.success(f"Tool '{result.get('deleted_tool', 'unknown')}' was deleted successfully!")
                    else:
                        st.error("Deletion failed.")
                elif response.status_code == 404:
                    st.error(f"Tool with ID '{tool_id_to_delete}' not found.")
                else:
                    st.error(f"Error: API returned status code {response.status_code}")
                    st.text(response.text)
            except Exception as e:
                st.error(f"Error connecting to API: {str(e)}")
    
    st.divider()
    
    st.markdown("### Clear Entire Index")
    st.error("⚠️ DANGER: This will delete ALL tools from the index. This action cannot be undone.")
    
    if not st.session_state.api_key:
        st.info("Please enter your Pinecone API Key in the sidebar to use this function.")
    
    confirm_clear = st.checkbox("I understand that this will delete ALL data from the index permanently")
    
    if st.button("Clear Index", type="primary", disabled=not confirm_clear or not st.session_state.api_key):
        with st.spinner("Clearing index..."):
            try:
                # Prepare environment variables to send
                headers = {}
                if st.session_state.environment == "DEV":
                    headers["ENVIRONMENT"] = "DEV"
                    headers["DEV_MODEL"] = st.session_state.dev_model
                else:
                    headers["ENVIRONMENT"] = "PROD"
                    headers["PROD_MODEL"] = st.session_state.prod_model
                    
                response = requests.delete(
                    f"{st.session_state.api_url}/clear-index",
                    json={"api_key": st.session_state.api_key},
                    headers=headers
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        st.success(f"Index cleared successfully! Deleted {result.get('deleted_count', 0)} tools.")
                    else:
                        st.error("Operation failed.")
                elif response.status_code == 401:
                    st.error("Unauthorized: Invalid API Key.")
                else:
                    st.error(f"Error: API returned status code {response.status_code}")
                    st.text(response.text)
            except Exception as e:
                st.error(f"Error connecting to API: {str(e)}")

# 5. STATISTICS TAB
with tabs[4]:
    st.markdown('<div class="subheader">Index Statistics</div>', unsafe_allow_html=True)
    
    if st.button("Refresh Statistics", key="refresh_stats"):
        with st.spinner("Fetching statistics..."):
            try:
                # Prepare environment variables to send
                headers = {}
                if st.session_state.environment == "DEV":
                    headers["ENVIRONMENT"] = "DEV"
                    headers["DEV_MODEL"] = st.session_state.dev_model
                else:
                    headers["ENVIRONMENT"] = "PROD"
                    headers["PROD_MODEL"] = st.session_state.prod_model
                    
                response = requests.get(
                    f"{st.session_state.api_url}/stats",
                    headers=headers
                )
                
                if response.status_code == 200:
                    stats = response.json()
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Tools", stats.get("total_vectors", 0))
                    with col2:
                        st.metric("Vector Dimension", stats.get("dimension", "-"))
                    with col3:
                        st.metric("Index Fullness", f"{stats.get('index_fullness', 0):.2%}")
                    
                    # Display vector information
                    st.markdown("### Tools in Index")
                    if "vectors" in stats and len(stats["vectors"]) > 0:
                        # Convert to DataFrame for better display
                        vectors_df = pd.DataFrame(stats["vectors"])
                        
                        # Add category counts
                        if "categories" in vectors_df.columns:
                            # Extract categories and count occurrences
                            all_categories = []
                            for cats in vectors_df["categories"]:
                                if cats and cats != "N/A":
                                    categories_list = [c.strip() for c in cats.split(",")]
                                    all_categories.extend(categories_list)
                            
                            category_counts = pd.Series(all_categories).value_counts()
                            
                            # Show category distribution
                            st.markdown("### Category Distribution")
                            st.bar_chart(category_counts)
                        
                        # Show the main table
                        st.dataframe(vectors_df, use_container_width=True)
                    else:
                        st.info("No tools found in the index.")
                    
                    # Show raw JSON for detailed inspection
                    with st.expander("View Raw Statistics"):
                        st.json(stats)
                else:
                    st.error(f"Error: API returned status code {response.status_code}")
                    st.text(response.text)
            except Exception as e:
                st.error(f"Error connecting to API: {str(e)}")