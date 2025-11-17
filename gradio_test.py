# # # # # # # # # # # import gradio as gr

# # # # # # # # # # # def greet(name, intensity):
# # # # # # # # # # #     return "Hello, " + name + "!" * int(intensity)

# # # # # # # # # # # demo = gr.Interface(
# # # # # # # # # # #     fn=greet,
# # # # # # # # # # #     inputs=["text", "slider"],
# # # # # # # # # # #     outputs=["text"],
# # # # # # # # # # # )

# # # # # # # # # # # demo.launch()




# # # # # # # # # # import gradio as gr

# # # # # # # # # # def greet(name):
# # # # # # # # # #     return "Hello " + name + "!"

# # # # # # # # # # demo = gr.Interface(fn=greet, inputs="textbox", outputs="textbox")
    
# # # # # # # # # # demo.launch(share=True)  # Share your demo with just 1 extra parameter 🚀



# # # # # # # # # import gradio as gr

# # # # # # # # # def greet(name, intensity):
# # # # # # # # #     return "Hello, " + name + "!" * intensity

# # # # # # # # # demo = gr.Interface(
# # # # # # # # #     fn=greet,
# # # # # # # # #     inputs=["text", gr.Slider(value=2, minimum=1, maximum=10, step=1)],
# # # # # # # # #     outputs=[gr.Textbox(label="greeting", lines=3)],
# # # # # # # # # )

# # # # # # # # # demo.launch()



# # # # # # # # import gradio as gr

# # # # # # # # def greet(name, is_morning, temperature):
# # # # # # # #     salutation = "Good morning" if is_morning else "Good evening"
# # # # # # # #     greeting = f"{salutation} {name}. It is {temperature} degrees today"
# # # # # # # #     celsius = (temperature - 32) * 5 / 9
# # # # # # # #     return greeting, round(celsius, 2)

# # # # # # # # demo = gr.Interface(
# # # # # # # #     fn=greet,
# # # # # # # #     inputs=["text", "checkbox", gr.Slider(0, 100)],
# # # # # # # #     outputs=["text", "number"],
# # # # # # # # )
# # # # # # # # demo.launch()




# # # # # # # import numpy as np
# # # # # # # import gradio as gr

# # # # # # # def sepia(input_img):
# # # # # # #     sepia_filter = np.array([
# # # # # # #         [0.393, 0.769, 0.189],
# # # # # # #         [0.349, 0.686, 0.168],
# # # # # # #         [0.272, 0.534, 0.131]
# # # # # # #     ])
# # # # # # #     sepia_img = input_img.dot(sepia_filter.T)
# # # # # # #     sepia_img /= sepia_img.max()
# # # # # # #     return sepia_img

# # # # # # # demo = gr.Interface(sepia, gr.Image(), "image")
# # # # # # # demo.launch()


# # # # # # import gradio as gr

# # # # # # def calculator(num1, operation, num2):
# # # # # #     if operation == "add":
# # # # # #         return num1 + num2
# # # # # #     elif operation == "subtract":
# # # # # #         return num1 - num2
# # # # # #     elif operation == "multiply":
# # # # # #         return num1 * num2
# # # # # #     elif operation == "divide":
# # # # # #         if num2 == 0:
# # # # # #             raise gr.Error("Cannot divide by zero!")
# # # # # #         return num1 / num2
    

# # # # # # demo = gr.Interface(
# # # # # #     calculator,
# # # # # #     [
# # # # # #         "number",
# # # # # #         gr.Radio(["add", "subtract", "multiply", "divide"]),
# # # # # #         "number"
# # # # # #     ],
# # # # # #     "number",
# # # # # #     examples=[
# # # # # #         [45, "add", 3],
# # # # # #         [3.14, "divide", 2],
# # # # # #         [144, "multiply", 2.5],
# # # # # #         [0, "subtract", 1.2],
# # # # # #     ],
    
# # # # # #     title="Toy Calculator",
# # # # # #     description="Here's a sample toy calculator.",
# # # # # # )

# # # # # # demo.launch()



# # # # # import gradio as gr

# # # # # def calculate_bmi(age, height, weight):
# # # # #     bmi = weight / (height **2)
# # # # #     return f"Age: {age}, BMI: {bmi:.2f}"

# # # # # demo = gr.Interface(
# # # # #     fn=calculate_bmi,
# # # # #     inputs=[
# # # # #         gr.Number(label='Age', info='In years, must be greater than 0'),  # 年龄输入
# # # # #         gr.Number(label='Height', info='In meters, e.g., 1.75'),  # 身高输入
# # # # #         gr.Number(label='Weight', info='In kilograms, e.g., 65')  # 体重输入
# # # # #     ],
# # # # #     outputs="text"
# # # # # )
# # # # # demo.launch()




# # # # # import gradio as gr

# # # # # def generate_fake_image(prompt, seed, initial_image=None):
# # # # #     return f"Used seed: {seed}", "https://dummyimage.com/300/09f.png"

# # # # # demo = gr.Interface(
# # # # #     generate_fake_image,
# # # # #     inputs=["textbox"],
# # # # #     outputs=["textbox", "image"],
# # # # #     additional_inputs=[
# # # # #         gr.Slider(0, 1000),
# # # # #         "image"
# # # # #     ]
# # # # # )

# # # # # demo.launch()




# # # # import gradio as gr


# # # # def greet(name):
# # # #     return "Hello " + name + "!"


# # # # with gr.Blocks() as demo:
# # # #     name = gr.Textbox(label="Name")
# # # #     output = gr.Textbox(label="Output Box")
# # # #     greet_btn = gr.Button("Greet")
# # # #     greet_btn.click(fn=greet, inputs=name, outputs=output, api_name="greet")

# # # # demo.launch()




# # # import gradio as gr

# # # with gr.Blocks() as demo:
# # #     name = gr.Textbox(label="Name")
# # #     output = gr.Textbox(label="Output Box")
# # #     greet_btn = gr.Button("Greet")

# # #     @greet_btn.click(inputs=name, outputs=output)
# # #     def greet(name):
# # #         return "Hello " + name + "!"

# # # demo.launch()



# # import gradio as gr

# # def welcome(name):
# #     return f"Welcome to Gradio, {name}!"

# # with gr.Blocks() as demo:
# #     gr.Markdown(
# #     """
# #     # Hello World!
# #     Start typing below to see the output.
# #     """)
# #     inp = gr.Textbox(placeholder="What is your name?")
# #     out = gr.Textbox()
# #     inp.change(welcome, inp, out)

# # demo.launch()




# import gradio as gr

# def increase(num):
#     return num + 1

# with gr.Blocks() as demo:
#     a = gr.Number(label="a")
#     b = gr.Number(label="b")
#     atob = gr.Button("a > b")
#     btoa = gr.Button("b > a")
#     atob.click(increase, a, b)
#     btoa.click(increase, b, a)

# demo.launch()



from transformers import pipeline

import gradio as gr

asr = pipeline("automatic-speech-recognition", "facebook/wav2vec2-base-960h")
classifier = pipeline("text-classification")

def speech_to_text(speech):
    text = asr(speech)["text"]  
    return text

def text_to_sentiment(text):
    return classifier(text)[0]["label"]  

demo = gr.Blocks()

with demo:
    audio_file = gr.Audio(type="filepath")
    text = gr.Textbox()
    label = gr.Label()

    b1 = gr.Button("Recognize Speech")
    b2 = gr.Button("Classify Sentiment")

    b1.click(speech_to_text, inputs=audio_file, outputs=text)
    b2.click(text_to_sentiment, inputs=text, outputs=label)

demo.launch()
