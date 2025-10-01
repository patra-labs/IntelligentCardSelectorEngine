
import streamlit as st
from PyPDF2 import PdfReader
import requests
import base64

# Set up the Streamlit page
st.set_page_config(page_title="PDF Summarizer with Gemini", layout="wide")
st.title("📄 PDF Summarizer using Google Gemini")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    # Read the PDF
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    # Display extracted text preview
    st.subheader("Extracted Text (Preview)")
    st.text_area("Text Preview", text[:1000], height=200)

    # # Summarize the PDF
    # if st.button("Summarize PDF"):
    #     with st.spinner("Generating summary..."):
    #         try:
    #             # Prepare the PDF content as base64
    #             pdf_base64 = base64.b64encode(uploaded_file.read()).decode("utf-8")

    #             # Set up the Gemini API endpoint and headers
    #             api_url = "https://api.genai.google.com/v1/documents:analyze"
    #             headers = {
    #                 "Authorization": "Bearer YOUR_GEMINI_API_KEY",
    #                 "Content-Type": "application/json",
    #             }

    #             # Prepare the payload
    #             payload = {
    #                 "document": {
    #                     "content": pdf_base64,
    #                     "mimeType": "application/pdf",
    #                 },
    #                 "features": ["SUMMARY"],
    #             }

    #             # Make the API request
    #             response = requests.post(api_url, headers=headers, json=payload)
    #             response.raise_for_status()

    #             # Extract and display the summary
    #             summary = response.json().get("summary", "No summary available.")
    #             st.subheader("📌 Summary")
    #             st.write(summary)

    #         except requests.exceptions.RequestException as e:
    #             st.error(f"An error occurred: {e}")
    #         except Exception as e:
    #             st.error(f"An unexpected error occurred: {e}")
