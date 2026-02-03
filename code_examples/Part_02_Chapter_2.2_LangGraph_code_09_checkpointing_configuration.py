from langgraph.checkpoint.sqlite import SqliteSaver

# Create checkpointer with SQLite backend
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

# Compile graph with checkpointing enabled
app = workflow.compile(checkpointer=checkpointer)

# Execute with thread_id for checkpoint tracking
config = {"configurable": {"thread_id": "debug-session-001"}}
final_state = app.invoke(initial_state, config=config)
