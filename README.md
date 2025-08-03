# Assistant Chatbot
Python version = 3.12.3

To run this project,
- Start a Codespace from the `code` button
- Wait for Codespace to finish setting up
- Install uv (Python package and project manager) by running this command `curl -LsSf https://astral.sh/uv/install.sh | sh` in terminal (source: [uv installation docs](https://docs.astral.sh/uv/getting-started/installation/))
- Run `uv sync` to install packages. This will take time
- While the packages are getting installed, create `.env` file using the `.env.example` file format. Save your API tokens to file
- Put your files for RAG in `documents/` folder
- After packages are installed, run `uv run initialize_vectorstore.py`, to create vectorstore database
- Run `uv run app.py` to start app
- Ask questions on your documents and files
- When you want to close the app, type 'exit' in question
