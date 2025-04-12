from rag_agent import agent_with_knowledge
from agno.playground import Playground, serve_playground_app


app = Playground(
    agents=[agent_with_knowledge]
).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)