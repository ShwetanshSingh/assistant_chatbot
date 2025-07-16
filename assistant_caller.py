from utils.llm_init import assistant

while True:
    query = input("\n=>Enter your query: ")
    if query.lower() == "exit":
        break
    try:
        response = assistant.get_answer(query)
        answer = response["answer"]
        source = response["context"]
        source_list = [src.metadata["source"] for src in source]
        print(f"Answer: {answer}")
        print(f"Sources: {source_list}")
    except Exception as e:
        print(f"An error occurred: {e}")