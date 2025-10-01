# Chatbot Implementation Plan
1. Extend `Settings` with chat backend configuration (OpenAI "responses" API vs local Llama, plus parameters for each backend).
2. Add a ChatService abstraction with unified interface, using OpenAI Responses API (via `client.responses.create`) and llama-cpp for local model.
3. Implement retrieval pipeline: gather top-k documents from Qdrant and format prompt for both backends.
4. Create new FastAPI endpoint `/chat` (JSON) and optionally update UI for chat flow.
5. Write tests: mock OpenAI Responses client, stub llama-cpp, ensure prompt/context handling.
6. Document setup (env vars, llama model download, example usage).