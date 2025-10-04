import gradio as gr


def add_numbers(number1, number2):
    return number1+number2

# gradio interface
interface = gr.Interface(
    fn=add_numbers,
    inputs=[gr.Number(), gr.Number()],
    outputs=gr.Number()
    
    )

interface.launch(server_name="127.0.0.1", server_port= 7860)