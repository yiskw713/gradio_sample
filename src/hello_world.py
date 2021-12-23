import gradio as gr


def greet(name):
    return "Hello " + name + "!!"


def main():
    iface = gr.Interface(fn=greet, inputs="text", outputs="text")
    iface.launch(server_name="0.0.0.0")


if __name__ == "__main__":
    main()
