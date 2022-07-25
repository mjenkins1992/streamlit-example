import pandas as pd
import streamlit as st
from PyPDF2 import PdfFileReader
import pdfplumber
import docx2txt
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

path = "./model/"
dev_id = '1' #Change to 0 for single GPU systems. Currently set as 1 to use free GPU

model = AutoModelForSeq2SeqLM.from_pretrained(path, local_files_only=True) # Load Model from local Path
tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True) # Load Tokenizer from local Path

# CALLBACKS

# Toggles params based on seelcted Input Methods
def update_input_params():
    st.session_state.input_text = ""
    st.session_state.file_up_off = not(st.session_state.file_up_off)
    st.session_state.text_in_off = not(st.session_state.text_in_off)
    return

# Reads raw text from PDF file
def read_pdf_old(file):
    pdfReader = PdfFileReader(file)
    count = pdfReader.numPages
    all_page_text = ""
    for i in range(count):
        page = pdfReader.getPage(i)
        all_page_text += page.extractText()
        all_page_text += "\n"
    return all_page_text

# Reads raw text from PDF file
def read_pdf(file):
    text = ''
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
            text += "\n"
    return text

# Extracts raw text from input depending on input method and file type
def get_raw_txt():
    with st.spinner('Extracting Text...'):
        if st.session_state["text_box"]:
            raw_text = st.session_state.input_text
        elif st.session_state["text_upload"]:
            if st.session_state.input_file.type == "text/plain":
                raw_text = str(st.session_state.input_file.read(),"utf-8")
            elif st.session_state.input_file.type == "application/pdf":
                raw_text = read_pdf(st.session_state.input_file)
            elif st.session_state.input_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" :
                raw_text = docx2txt.process(st.session_state.input_file)
        st.session_state.raw_text = raw_text

# Run the model to generate a summary
def run_model(data):
    to_pred = tokenizer(data, padding="max_length", max_length=4096, return_tensors="pt", truncation=True)

    input_ids=to_pred["input_ids"].cuda()
    attention_mask=to_pred["attention_mask"].cuda()
    #global attention on special tokens
    global_attention_mask = torch.zeros_like(attention_mask)
    global_attention_mask[:, 0] = 1

    predicted_ids = model.generate(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)

    output = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)

    return output

# Itterates through input file to generate full summary
def generate_summary():
    max_input = 4096
    raw_text = st.session_state.raw_text

    if len(raw_text) > max_input: # If length of input is > max model input length
        with st.spinner('Pre-Processing Input...'):
            # Split text into chunks of size max_input
            chunked_text = [raw_text[i:i+max_input] for i in range(0, len(raw_text), max_input)]
    else:
        # If input does not need split
        chunked_text = raw_text

    output = []
    # Generate Summary for each Input Chunk
    with st.spinner('Running Model...'):
        for i in range(0, len(chunked_text)):
            output.append(run_model(chunked_text[i]))

    temp_out = ''
    # Format output chunks as single string
    with st.spinner('Building Output...'):
        for i in output:
            temp_out += ' '+i[0]

    st.session_state.final_output = temp_out

def run_analysis():
    # Get the raw text from the input
    get_raw_txt()
    # Any preprocessing on raw text should happen here!!
    # Pass the model to the GPU
    model.cuda()
    # Run the function to generate summary
    generate_summary()
    #Activate Download Button
    st.session_state.download_button_off = False
    st.success('Complete!')
    return

# Controls the visibility of the Generate Summary button based on availability of input data
def update_button():
    if st.session_state["text_box"]:
            st.session_state.generate_button_off = False
    elif st.session_state["text_upload"]:
            st.session_state.generate_button_off = False
    else:
            st.session_state.generate_button_off = True
    return

# UI FLOW

# Page Config
st.set_page_config(
    page_title="Lay Summary Generator",
    page_icon="📄",
)

# Page Layout Config
def _max_width_():
    max_width_str = f"max-width: 1400px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )
_max_width_()

# Page Title
st.title("📄 Lay Summary Generator")

#About App Section
with st.expander("ℹ️ - About this app", expanded=True):
    st.write(
        """
        - This application uses the Longformer Transformer NLP Algorithm to generate Lay Summaries of Clinical Trial Reports.
        - Select Your Data Input Method
        - File Upload accepts PDF, DOCX and TXT
        - Click Generate Summary to run the model
        - The Download will become available when the summary is complete
        - To run a second summary or fix any GUI issues refresh the page
        """
    )

# Initialise Default Parameters
if 'in_type' not in st.session_state:
    st.session_state.in_type = 0                        # Input type. 0 = File Upload, 1 = Input Text
if 'input_text' not in st.session_state:
    st.session_state.input_text = ""                    # Variable for input via text box
if 'input_file' not in st.session_state:
    st.session_state.input_file = ""                    # Variable for input via file
if 'file_up_off' not in st.session_state:
    st.session_state.file_up_off = False                # Controls File Upload Visibility
if 'text_in_off' not in st.session_state:
    st.session_state.text_in_off = True                 # Controls Text Entry Visibility
if 'download_button_off' not in st.session_state:
    st.session_state.download_button_off = True         # Controls Download Button Visibility
if 'final_output' not in st.session_state:
    st.session_state.final_output = ""                  # Variable for final Summary
if 'generate_button_off' not in st.session_state:
    st.session_state.generate_button_off = True         # Controls Generate Summary Button Visibility
if 'raw_text' not in st.session_state:
    st.session_state.raw_text = True                    # Variable to store raw text

# Input Type Selection
st.session_state.in_type = st.radio("How would you like to input your data?",
     ('File Upload', 'Enter Text'), horizontal=True, on_change=update_input_params)

# Data Input Section
c30, c31 = st.columns([.5, 1])

# Col for Input by File
with c30:
    label = "📤 Upload File"
    st.session_state.input_file = st.file_uploader(label, disabled=st.session_state.file_up_off, on_change=update_button, type=['txt', 'pdf', 'docx'], key='text_upload')

# Col for Input by Text
with c31:
    label = "⌨️ Enter Text"
    st.session_state.input_text = st.text_area(label, height=175, placeholder="Type or Paste the text you would like to summarise here...", on_change=update_button, key='text_box', disabled=st.session_state.text_in_off)

# Data Processing Section
c30, c31, c32 = st.columns([.25, .25, 1])

# Col for Start Button
with c30:
    label = "Generate Summary"
    st.session_state.run = st.button(label, on_click=run_analysis, disabled=st.session_state.generate_button_off)

# Col for Download Button
with c31:
    label = "Download Summary"
    st.download_button(label, ''.join(st.session_state.final_output), file_name="summary.txt", disabled=st.session_state.download_button_off)

# Results Section
label = "✅ Lay Summary"
st.text_area(label, value=st.session_state.final_output, disabled=True, height=175, placeholder='Your summary will appear here when it has been generated...')
