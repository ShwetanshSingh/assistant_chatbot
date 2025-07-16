from utils.llm_init import assistant

questions = [
    "What is the Constitution of India?",
    "How many articles are there in the Constitution of India?",
    "What is the significance of the Preamble in the Constitution of India?",
    "What are Fundamental Rights in the Constitution of India?",
    "What is the procedure for amending the Constitution of India?",
    "Who is known as the Father of the Indian Constitution?",
    "What is the role of the President in the Indian Constitution?",
    "How does the Constitution of India ensure separation of powers?",
    "What are Directive Principles of State Policy in the Constitution of India?",
    "How does the Constitution of India protect minority rights?"
]
for question in questions:
        print(f"\nType: {type(question)} == Question: {question}")
        response = assistant.get_answer(question)
        answer = response["answer"]
        source = response["context"]
        source = [src.metadata["source"] for src in source]
        print(f"Type: {type(answer)} == Answer: {answer}")
        print(f"Type: {type(source)} == Source: {source}")
        print("\n=====================================================")