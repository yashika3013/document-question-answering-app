import torch
import gradio as gr
# Use a pipeline as a high-level helper

from transformers import pipeline


model_path = model_path = "C:\\Users\\yashika\\.cache\\huggingface\\hub\\models--deepset--roberta-base-squad2\\snapshots\\adc3b06f79f797d1c575d5479d6f5efe54a9e3b4"
question_answer = pipeline("question-answering", model="deepset/roberta-base-squad2")
#question_answer = pipeline("question-answering", model=model_path)

# context = "Artificial intelligence (AI) is the capability of computational systems to perform tasks typically associated with human intelligence, such as learning, reasoning, problem-solving, perception, and decision-making. It is a field of research in computer science that develops and studies methods and software that enable machines to perceive their environment and use learning and intelligence to take actions that maximize their chances of achieving defined goals."
# question = "What is artificial intelligence?"

def read_file_content(file_obj):
    try:
        with open(file_obj.name,'r',encoding='utf-8') as file:
            context = file.read()
            return context
    except Exception as e:
        return f'An error occurred: {e}'


def get_answer(file, question):
    context = read_file_content(file)
    answer = question_answer(question=question, context=context)
    confidence = f"{answer['score'] * 100:.2f}%"  # Convert to percentage
    return answer['answer'], confidence

demo = gr.Interface(
    fn=get_answer,
    inputs=[
        gr.File(label="Upload your .txt file"),
        gr.Textbox(label="Ask your question", lines=1)
    ],
    outputs=[
        gr.Textbox(label="Answer", lines=1),
        gr.Textbox(label="Confidence Score (%)", lines=1)
    ],
    title="Document Question Answering",
    description="Upload a document and ask a question to extract an answer. Powered by Hugging Face Transformers 🤖💬"
)
demo.launch()