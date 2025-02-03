
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load DeepSeek model and tokenizer
@st.cache_resource
def load_model():
    model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    return model, tokenizer

model, tokenizer = load_model()

# Streamlit app
st.title("DeepSeek Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Format prompt for DeepSeek
    formatted_prompt = f"User: {prompt}\n\nAssistant: "
    
    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the assistant's response
        assistant_response = response.split("Assistant: ")[-1]
        
        message_placeholder.markdown(assistant_response)
    
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})