import gradio as gr
from huggingface_hub import InferenceClient
import random
import textwrap

# Define the model to be used
model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
client = InferenceClient(model)

# Embedded system prompt
system_prompt_text = "Act like a compassionate and helpful Health consultant and professional therapist named CareNetAI owned by YAiC. You help and support with any kind of request and provide a detailed answer or suggestion to the question. You are friendly and willing to help depressed people and also help people identify manipultors and how to protect themselves. But if you are asked about something unethical or dangerous, you must provide a safe and respectful way to handle that. If someone has sucidal thought, you must try your best to explain that they matter and motivate that life is full of up and down and remember that Luck is when consistency meets opportunity! Also failure is also a part of growth, and there is so much more to life. Never say that you cannot help them, that will make them even more depressed or worse! Be sure to ask for specific problem and do your best to give professional advices, remember you are a preoifessional."

# Read the content of the info.md file
with open("info.md", "r") as file:
    info_md_content = file.read()

# Chunk the info.md content into smaller sections
chunk_size = 2000  # Adjust this size as needed
info_md_chunks = textwrap.wrap(info_md_content, chunk_size)

def get_all_chunks(chunks):
    return "\n\n".join(chunks)

def format_prompt_mixtral(message, history, info_md_chunks):
    prompt = "<s>"
    all_chunks = get_all_chunks(info_md_chunks)
    prompt += f"{all_chunks}\n\n"  # Add all chunks of info.md at the beginning
    prompt += f"{system_prompt_text}\n\n"  # Add the system prompt

    if history:
        for user_prompt, bot_response in history:
            prompt += f"[INST] {user_prompt} [/INST]"
            prompt += f" {bot_response}</s> "
    prompt += f"[INST] {message} [/INST]"
    return prompt

def chat_inf(prompt, history, seed, temp, tokens, top_p, rep_p):
    generate_kwargs = dict(
        temperature=temp,
        max_new_tokens=tokens,
        top_p=top_p,
        repetition_penalty=rep_p,
        do_sample=True,
        seed=seed,
    )

    formatted_prompt = format_prompt_mixtral(prompt, history, info_md_chunks)
    stream = client.text_generation(formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=False)
    output = ""
    for response in stream:
        output += response.token.text
        yield [(prompt, output)]
    history.append((prompt, output))
    yield history

def clear_fn():
    return None, None

rand_val = random.randint(1, 1111111111111111)

def check_rand(inp, val):
    if inp:
        return gr.Slider(label="Seed", minimum=1, maximum=1111111111111111, value=random.randint(1, 1111111111111111))
    else:
        return gr.Slider(label="Seed", minimum=1, maximum=1111111111111111, value=int(val))

with gr.Blocks() as app:  # Add auth here
    gr.HTML("""<center><h1 style='font-size:xx-large;'>CareNetAI</h1><br><h3> made with love by YAiC </h3><br><h7>EXPERIMENTAL</center>""")
    with gr.Row():
        chat = gr.Chatbot(height=500)
    with gr.Group():
        with gr.Row():
            with gr.Column(scale=3):
                inp = gr.Textbox(label="Prompt", lines=5, interactive=True)  # Increased lines and interactive
                with gr.Row():
                    with gr.Column(scale=2):
                        btn = gr.Button("Chat")
                    with gr.Column(scale=1):
                        with gr.Group():
                            stop_btn = gr.Button("Stop")
                            clear_btn = gr.Button("Clear")
            with gr.Column(scale=1):
                with gr.Group():
                    rand = gr.Checkbox(label="Random Seed", value=True)
                    seed = gr.Slider(label="Seed", minimum=1, maximum=1111111111111111, step=1, value=rand_val)
                    tokens = gr.Slider(label="Max new tokens", value=3840, minimum=0, maximum=8000, step=64, interactive=True, visible=True, info="The maximum number of tokens")
                    temp = gr.Slider(label="Temperature", step=0.01, minimum=0.01, maximum=1.0, value=0.9)
                    top_p = gr.Slider(label="Top-P", step=0.01, minimum=0.01, maximum=1.0, value=0.9)
                    rep_p = gr.Slider(label="Repetition Penalty", step=0.1, minimum=0.1, maximum=2.0, value=1.0)

    hid1 = gr.Number(value=1, visible=False)

    go = btn.click(check_rand, [rand, seed], seed).then(chat_inf, [inp, chat, seed, temp, tokens, top_p, rep_p], chat)

    stop_btn.click(None, None, None, cancels=[go])
    clear_btn.click(clear_fn, None, [inp, chat])

app.queue(default_concurrency_limit=10).launch(share=True, auth=("admin", "0112358"))
