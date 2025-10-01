# Implementation Plan – Support Agent Chatbot & Glossary
1. Extend Settings/config to include chat backend selection (OpenAI Responses vs local Llama), chat model parameters, and glossary configuration (file path + top_k injection).
2. Build glossary module: load/save definitions (YAML/JSON), expose lookup by term and fuzzy matching, plus helper to format definitions for prompts.
3. Implement ChatService abstraction handling retrieval (Qdrant), glossary injection, prompt templating, and backend-specific generation via OpenAI Responses and llama-cpp-python.
4. Add FastAPI endpoints for chat (`POST /support/chat` returning structured answer) and glossary management (read-only for now).
5. Update Jinja UI with a Support Agent tab and basic conversation display.
6. Add tests covering glossary loader, chat prompt assembly, backend dispatch (mocked), and API responses.