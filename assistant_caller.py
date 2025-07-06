from llm_util.llm_init import assistant

while True:
    query = input("=>Enter your query: ")
    if query.lower() == "exit":
        break
    try:
        answer = assistant.get_answer(query)
        answer.pretty_print()
    except Exception as e:
        print(f"An error occurred: {e}")