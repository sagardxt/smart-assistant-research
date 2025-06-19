import streamlit as st
import pdfplumber
from transformers import pipeline

# Load Models
summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")
qa_model = pipeline("question-answering")

# Functions
def extract_text_from_pdf(file):
    """Extracts text from PDF using pdfplumber."""
    with pdfplumber.open(file) as pdf:
        text = ''.join(page.extract_text() for page in pdf.pages)
    return text

def summarize_text(text, max_length=150, min_length=50):
    """Generates a concise summary of the input text."""
    return summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]["summary_text"]

def answer_question(question, context):
    """Answers questions using a pre-trained QA model."""
    return qa_model(question=question, context=context)["answer"]

def generate_logic_questions(text):
    """Generates basic logic-based questions from the document."""
    # Simple example of logic-based question generation
    questions = [
        f"What is the main focus of this document?",
        f"Explain the key takeaway from the introduction.",
        f"How does the document conclude?"
    ]
    return questions

# Streamlit Frontend
st.set_page_config(page_title="Smart Assistant for Research", layout="wide")
st.title("📄 Smart Assistant for Research Summarization")

# File Upload Section
uploaded_file = st.file_uploader("Upload a PDF or TXT file:", type=["pdf", "txt"])

if uploaded_file:
    # Extract Text from File
    if uploaded_file.type == "application/pdf":
        document_text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "text/plain":
        document_text = uploaded_file.read().decode("utf-8")
    else:
        st.error("Unsupported file format!")

    # Generate and Display Summary
    with st.spinner("Generating summary..."):
        summary = summarize_text(document_text)
    st.subheader("Auto-Summary")
    st.write(summary)

    # Ask Anything Section
    st.subheader("Ask Anything")
    user_question = st.text_input("Enter your question:")
    if st.button("Get Answer"):
        with st.spinner("Finding answer..."):
            answer = answer_question(user_question, document_text)
        st.success(f"**Answer:** {answer}")

    # Challenge Me Section
    st.subheader("Challenge Me")
    if st.button("Generate Questions"):
        st.info("Answer the following questions based on the document:")
        logic_questions = generate_logic_questions(document_text)
        for i, question in enumerate(logic_questions):
            user_answer = st.text_area(f"Q{i+1}: {question}", key=f"q{i+1}")
            if st.button(f"Evaluate Q{i+1}", key=f"eval{i+1}"):
                st.write("Evaluation not implemented for demo purposes.")

else:
    st.info("Please upload a document to start.")

