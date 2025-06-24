import os
from autogen import ConversableAgent
import re

# Configuration for Qwen 1.5B via Ollama
QWEN_MODEL = "qwen2:1.5b"  
llm_config = {
    "config_list": [
        {
            "model": QWEN_MODEL,
            "base_url": "http://localhost:11434/v1",  # Ollama's API endpoint
            "api_key": "ollama",  # Placeholder (not actually used by Ollama)
            "price": [0, 0]  # Suppress the pricing warning
        }
    ],
    "temperature": 0.7,
    "timeout": 180,  # Increased timeout for local model 
     "max_tokens": 512  # Limit response length
}

# Define termination condition function
def is_termination_msg(message):
    """Check if a message contains a termination signal"""
    # Extract content from message dictionary
    content = message.get("content", "") if isinstance(message, dict) else str(message)
    
    # Convert to lowercase for case-insensitive matching
    content = content.lower()
    
    termination_phrases = [
        "terminate",
        "report complete",
        "final report",
        "research concluded",
        "end of report",
        "output complete"
    ]
    
    return any(phrase in content for phrase in termination_phrases)

# Define your agents
try:
    research_agent = ConversableAgent(
        "researcher",
        llm_config=llm_config,
        system_message="""You are an expert web researcher. Your role is to find and extract relevant, up-to-date information on a given topic. You must cite your sources.
        When you have gathered sufficient information, say 'RESEARCH COMPLETE. TERMINATE.'""",
        max_consecutive_auto_reply=2,  # Limit auto-replies
        human_input_mode="NEVER",
    )

    writer_agent = ConversableAgent(
        "writer",
        llm_config=llm_config,
        system_message="""You are an expert technical writer. You take the information provided by the researcher and craft a well-structured, easy-to-read report.
        When your report is complete, say 'REPORT COMPLETE. TERMINATE.'""",
        max_consecutive_auto_reply=2,  # Limit auto-replies
        human_input_mode="NEVER",
        is_termination_msg=is_termination_msg,  # Termination condition
    )
    print("Agents created successfully! 🤖✍️")
    print("Enter a research topic to begin. Type 'exit' or 'quit' to end the session.")

    # --- Interactive Loop ---
    while True:
        try:
            # Get user input
            user_prompt = input("\n▶️  Enter your research topic: ")

            # Initiate chat with termination condition
            chat_result = research_agent.initiate_chat(
                writer_agent,
                message=user_prompt,
                max_turns=4,  # Max total turns in conversation
                summary_method="last_msg",
                summary_args={"use_cache": True}   # Return the final response
            )

            # Print the final result
            print("\n\n--- FINAL REPORT ---")
            print(chat_result.summary)
            print("--- END OF REPORT ---\n")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting the interactive session. Goodbye! 👋")
            break
        except Exception as e:
            print(f"An error occurred during the chat: {e}")
            

except Exception as e:
    print(f"An error occurred during agent creation: {e}")