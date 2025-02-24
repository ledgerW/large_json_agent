import json
from agents.smolagent import sql_query_agent

def test_hierarchical_data_tool():
    """
    Test the get_hierarchical_data_info tool to display information about the hierarchical structure
    of a database in a similar way to test_hierarchical_data.py.
    """
    print("Testing the get_hierarchical_data_info tool...")
    
    # Define a prompt that will use the get_hierarchical_data_info tool
    prompt = """
    I need to understand the structure of the llamacloud_hierarchical database.
    Please use the get_hierarchical_data_info tool to show me the hierarchical structure,
    including table relationships, statistics, and an overview of the data.
    """
    
    # Run the agent with the prompt
    result = sql_query_agent.run(prompt)
    
    # Print the result
    print("\nResult from the agent:")
    print(result)
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_hierarchical_data_tool()
