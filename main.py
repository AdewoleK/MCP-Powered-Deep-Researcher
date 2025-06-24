# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from autogen import GroupChat, GroupChatManager
from research import research_agent, writer_agent, llm_config

app = FastAPI()

class Task(BaseModel):
    description: str

@app.post("/start_task")
async def start_task(task: Task):
    """
    This endpoint kicks off a multi-agent chat to accomplish a task.
    """
    # Define the group chat with our agents
    groupchat = GroupChat(
        agents=[research_agent, writer_agent],
        messages=[],
        max_round=10
    )

    # The manager is the orchestrator (our MCP)
    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config
    )

    # Initiate the chat with the user's task
    chat_result = manager.run_chat(
        messages=[{"role": "user", "content": task.description}]
    )

    # In a real app, you'd save this result to a database
    # and return a task ID. For simplicity, we return the final message.
    return {"response": chat_result.summary}