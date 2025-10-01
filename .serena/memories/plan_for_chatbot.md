# Proposed Next Steps
- Extend Settings with chat backend configuration (e.g. CHAT_MODEL={openai_gpt4, llama_cpp}).
- Implement a ChatService abstraction that wraps OpenAI responses and local Llama 3.2 3B Instruct via llama-cpp-python.
- Add retrieval-augmented prompt template combining query + top Qdrant hits.
- Update FastAPI UI to include a chat view with streaming or aggregated responses.
- Add tests mocking both backends to validate prompt construction and response flow.
- Document setup for local Llama weights (download, model path env var).