import os
import streamlit as st
from image import extract_text_from_image, model
from main import generate_text, summarize_text, create_topic_prompt, create_question_prompt, call_cohere_api, parse_response, generate_text_about_topic, ask_question_about_text, memory, cohere_client
from difflib import SequenceMatcher
from main import evaluate_prediction
from main import extract_text_from_pdf


# Streamlit user interface setup with custom styling
st.markdown(
    """
    <style>
    .main {
        background-color: #000000;
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 24px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
    }
    .stTextArea>div>div>textarea {
        border-radius: 5px;
    }
    .stFileUploader>div>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 24px;
        cursor: pointer;
    }
    .stFileUploader>div>button:hover {
        background-color: #45a049;
    }
    .stTextArea>div>div>textarea, .stTextInput>div>div>input {
        background-color: #333333;
        color: white;
    }
    </style>
    """, 
    unsafe_allow_html=True     # to allow rendering of raw HTML content within Streamlit applications."render" refers to the process of generating and displaying content on the screen
)

st.title("🌟 Text Generation and Summarization with Cohere 🌟")

# Upload image for text extraction
st.subheader("🖼️ Extract Text from Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    with open("temp_image.jpg", "wb") as f:     #wb write-binary mode. This is typically done when you want to save binary data, such as an image, to a file.
        f.write(uploaded_file.getbuffer())     #used to write the contents of an uploaded file to the opened file in binary mode. This is commonly used in web applications to save uploaded files to the server
    extracted_text = extract_text_from_image("temp_image.jpg", model)
    st.write("**Extracted Text:**", extracted_text)
    memory.add_to_history("Extracted Text", extracted_text)

    # Generate text about the extracted topic
    generated_text = generate_text_about_topic(extracted_text, cohere_client)
    st.write("**Generated Text:**", generated_text)
    memory.add_to_history("Generated Text", generated_text)

    # Provide actual text for evaluation
    st.subheader("🏷️ Provide Actual Text for Evaluation")
    actual_text = st.text_area("Enter the actual text to evaluate the accuracy:")
    if st.button("Evaluate Accuracy"):
        if actual_text:
            accuracy = evaluate_prediction(extracted_text, actual_text)
            st.write(f"**Accuracy:** {accuracy:.2f}%")    #accyracy is displayed with 2 dp
        else:
            st.warning("Please enter the actual text for evaluation.")

# Upload PDF for text summarization
st.subheader("📄 Summarize Text from PDF")
uploaded_pdf = st.file_uploader("Choose a PDF...", type=["pdf"])
if uploaded_pdf is not None:
    with open("temp_document.pdf", "wb") as f:
        f.write(uploaded_pdf.getbuffer())
    pdf_text = extract_text_from_pdf("temp_document.pdf")
    st.write("**Extracted Text from PDF:**", pdf_text[:1000] + "...")     #display a portion of the text extracted from a PDF file.  that selects the first 1000 characters of the pdf_text string.
    
    if st.button("Summarize PDF Text"):
        summary = summarize_text(pdf_text)
        st.text_area("Summary:", value=summary, height=200)  #height=200: This parameter sets the height of the text area in pixels.

# User input for further questions
st.subheader("❓ Ask Me A Question To Generate Text")
question = st.text_input("Enter your question:")
if st.button("Ask Question"):
    if question:
        response = ask_question_about_text(generated_text, question, cohere_client)
        st.write("**Answer:**", response)
        memory.add_to_history(f"Question: {question}", response)
    else:
        st.warning("Please enter a question.")

# User input for text summarization
st.subheader("📝 Summarize Text")
text_to_summarize = st.text_area("Enter text to summarize:")
if st.button("Summarize Text"):
    if text_to_summarize:
        summary = summarize_text(text_to_summarize)
        st.text_area("Summary:", value=summary, height=100)
    else:
        st.warning("Please enter text to summarize.")