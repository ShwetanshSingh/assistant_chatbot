# Assistant Chatbot
Python version = 3.12.3

To run this project, 
- Start a Codespace from the `code` button
- Wait for Codespace to finish setting up
- Run the following commands in terminal:
    - `pip install -r requirements-1.txt`
    - `pip install -r requirements-2.txt`
- Create a `.env` file in directory, and save your Huggingface API token as it is saved in `.env.example`
- Run `python app.py` 
- You will see a link from Gradio in the terminal. `Ctrl+click` the link to open the application
- The current data (in `./data`) for RAG is `The Wizard of Oz`. You can submit your questions about the story, and the chatbot will provide responses with context gained from the document.
- Alternatively, you can put your own `.txt` document in `./data` folder and run the app over it.
