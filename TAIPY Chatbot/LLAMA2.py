import os
import replicate
from taipy.gui import Gui

# Ensure REPLICATE_API_TOKEN is properly set as an environment variable
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "r8_1OHNFO7JXcY8bEKNA7X5aGCNQ4245ku25i7b9")

# Set up Replicate client
replicate.api_token = REPLICATE_API_TOKEN

# Define the GUI layout and functionality
def create_gui():
    models = {
        'Llama2-7B': 'a16z-infra/llama7b-v2-chat:latest',
        'Llama2-13B': 'a16z-infra/llama13b-v2-chat:latest',
        'Llama2-70B': 'replicate/llama70b-v2-chat:latest',
    }

    # Initial state setup
    gui = Gui()

    # GUI layout definition
    gui.page = """
    <|Welcome to the simplified LLaMA chatbot!|>
    <|button|text='Click me'|>
    """
    
    # Initialize state properties
    gui.selected_model = 'Llama2-7B'
    gui.temperature = 0.1
    gui.top_p = 0.9
    gui.max_length = 512
    gui.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

    # Response generation function
    def generate_response(sender, value):
        llm = models[gui.selected_model]
        dialogue = "You are a helpful assistant."
        for msg in gui.messages:
            dialogue += f"\n{msg['role'].capitalize()}: {msg['content']}\n"
        dialogue += f"\nUser: {value}\nAssistant: "

        try:
            output = replicate.run(llm, input={"prompt": dialogue,
                                               "temperature": gui.temperature,
                                               "top_p": gui.top_p,
                                               "max_length": gui.max_length,
                                               "repetition_penalty": 1})
            gui.messages += [{"role": "user", "content": value},
                             {"role": "assistant", "content": output}]
        except Exception as e:
            print(f"An error occurred: {e}")

    return gui

if __name__ == "__main__":
    # Check for API token
    if not REPLICATE_API_TOKEN:
        print("Please set your REPLICATE_API_TOKEN environment variable.")
    else:
        create_gui().run(dark_mode=True)
