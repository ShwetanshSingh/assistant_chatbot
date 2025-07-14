from chatbot import graph, log_message
import gradio as gr

def ask_question(question: str):
    log_message("Chatbot", "Chatbot started")
    log_message("Chatbot", f"Question to chatbot:\n\t{question}")
    response = graph.invoke({"question": question})
    log_message("Chatbot", f"Response from Chatbot:\n\t{response["answer"]}")
    return response["answer"]

if __name__=="__main__":
    # gradio function
    ui = gr.Interface(
        fn=ask_question,
        inputs=["text"],
        outputs=["text"]
    )

    # launch ui
    try:
        ui.launch()
    except Exception as e:
        log_message("Error", f"Exception: {e}")

    # seperation for logs
    with open("logs.txt", "a") as f:
        f.write("\n----------------------------------------------\n\n")