import streamlit as st
from PIL import Image, ImageOps
import io
import base64
import requests
from streamlit_drawable_canvas import st_canvas
import json
import ollama

# Sidebar Configuration
st.sidebar.header("Configuration")
API_URL = st.sidebar.text_input("SageMaker API Endpoint URL", value="URL")
SECRET_TOKEN = st.sidebar.text_input("Bearer Token", value="SECRET")
MODEL_NAME = st.sidebar.text_input("Ollama Model", value="mistral")

# Main Streamlit UI
st.title("Sketch Classifier with Ollama")
st.header("(Tool Calling + SageMaker)")
st.markdown("Draw a digit, and ask the LLM figure it out!")


# Define the tool function
def classify_image(image: str) -> dict:
    """
    Classifies a base64-encoded image as one of the digits 0-9.

    Args:
        image (str): Base64-encoded image.

    Returns:
        result dict: Classification result with prediction and confidence. (e.g. {'confidence': 0.9928, 'prediction': 4})
    """
    try:
        print("-----------", image,"-----------")
        headers = {"Authorization": f"Bearer {SECRET_TOKEN}", "Content-Type": "application/json"}
        res = requests.post(API_URL, json={"image": image}, headers=headers)
        print("-----------", res.text,"-----------")
        returnJson = res.json()
        return {
           'confidence': returnJson['confidence'],
            'prediction': returnJson['prediction']
        }
        # return res.text
    except Exception as e:
        return {"error": str(e)}

# Here we could have multiple functions
available_functions = {
    'classify_image': classify_image
}


canvas_result = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    # Convert image to 28x28 grayscale PNG
    img = Image.fromarray(canvas_result.image_data.astype("uint8"))
    img = ImageOps.grayscale(img)
    img = ImageOps.invert(img)
    img = img.resize((28, 28))

    # Convert to base64
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    st.image(img, caption="Processed Image (28x28)", width=100)
    promptText = st.text_area("Prompt", value=f"Use the classify_image tool to classify this base64 encoded sketch image: {img_base64}", height=200)

    if st.button("Prompt AI"):
        with st.spinner("Running Ollama with tool calling..."):
            # Initiate conversation with tool calling
            try:
                response = ollama.chat(
                    model=MODEL_NAME,
                    messages=[
                        {
                            "role": "user",
                            "content": promptText
                        }
                    ],
                    tools=[classify_image]
                )

                # Handle tool calls
                if response.message.tool_calls:
                    for tool_call in response.message.tool_calls:
                        function_name = tool_call.function.name
                        arguments = tool_call.function.arguments
                        function_to_call = available_functions.get(function_name)
                        if function_to_call:
                            tool_result = function_to_call(**arguments)
                            st.caption("Tool Result:")
                            st.code(json.dumps(tool_result, indent=2))
                            # Append the tool result to the messages
                            response = ollama.chat(
                                model=MODEL_NAME,
                                messages=[
                                    {
                                        "role": "user",
                                        "content": f"The tool {function_name} returned: {tool_result}"
                                    }
                                ]
                            )
                            st.success("LLM Response:")
                            st.write(response.message.content)
                        else:
                            st.error(f"Function {function_name} not found.")
                else:
                    st.success("LLM Response:")
                    st.write(response.message.content)

            except Exception as e:
                st.error("Ollama model failed to run:")
                st.text(str(e))
else:
    st.info("Use your mouse to draw above.")
