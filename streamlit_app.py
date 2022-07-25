from collections import namedtuple
import altair as alt
import math
import pandas as pd
import numpy
import streamlit as st
import time
from io import StringIO
#from googletrans import Translator
from PyPDF2 import PdfFileReader
import docx2txt
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pickle
import os
#mps_device = torch.device("mps")
#os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

path = "./model/"
dev_id = 1 #Change to 0 for single GPU systems. Currently set as 1 to use free GPU

model = AutoModelForSeq2SeqLM.from_pretrained(path, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
#translator = Translator()

# CALLBACKS
def update_input_params():
    st.session_state.input_text = ""
    st.session_state.up_off = not(st.session_state.up_off)
    st.session_state.in_off = not(st.session_state.in_off)
    return

def read_pdf(file):
    pdfReader = PdfFileReader(file)
    count = pdfReader.numPages
    all_page_text = ""
    for i in range(count):
        page = pdfReader.getPage(i)
        all_page_text += page.extractText()
    return all_page_text

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
        st.session_state.box_value = raw_text

max_input = 4096
def generate_summary():

    raw_text = st.session_state.box_value
    if len(raw_text) < max_input:
        with st.spinner('Preprocessing Input...'):
            chunked_text = lambda raw_text, max_input: [raw_text[i:i+max_input] for i in range(0, len(raw_text), max_input)]
    else:
        chunked_text = 0


    with st.spinner('Running Tokenizer...'):
        to_pred = tokenizer(chunked_text, padding="max_length", max_length=4096, return_tensors="pt", truncation=True)

    with st.spinner('Pass tokens to GPU...'):
        input_ids=to_pred["input_ids"].cuda(dev_id)
        attention_mask=to_pred["attention_mask"].cuda(dev_id)

        #global attention on special tokens
        global_attention_mask = torch.zeros_like(attention_mask)
        global_attention_mask[:, 0] = 1

    with st.spinner('Running Model...'):
        predicted_ids = model.generate(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)

    with st.spinner('Decode Results...'):
        st.session_state.final_output = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)

def run_analysis():
    get_raw_txt()
    #Any preprocessing on raw text should happen here
    model.cuda(dev_id)
    generate_summary()
    st.session_state.download_off = False
    st.success('Complete!')
    return

def update_button():
    if st.session_state["text_box"]:
            st.session_state.generate_button = False
    elif st.session_state["text_upload"]:
            st.session_state.generate_button = False
    else:
            st.session_state.generate_button = True
    return

# UI FLOW
# Page Config
st.set_page_config(
    page_title="Lay Summary Generator",
    page_icon="📄",
)

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

# Initialise Default Parameters
if 'in_type' not in st.session_state:
    st.session_state.in_type = 0
if 'input_text' not in st.session_state:
    st.session_state.input_text = ""
if 'input_file' not in st.session_state:
    st.session_state.input_file = ""
if 'up_off' not in st.session_state:
    st.session_state.up_off = False
if 'in_off' not in st.session_state:
    st.session_state.in_off = True
if 'download_off' not in st.session_state:
    st.session_state.download_off = True
if 'stat_val' not in st.session_state:
    st.session_state.stat_val = 0
if 'output_text' not in st.session_state:
    st.session_state.output_text = ""
if 'final_output' not in st.session_state:
    st.session_state.final_output = ""
if 'generate_button' not in st.session_state:
    st.session_state.generate_button = True
if 'box_value' not in st.session_state:
    st.session_state.box_value = ""

# Input Type Selection
st.session_state.in_type = st.radio("How would you like to input your data?",
     ('File Upload', 'Free Text'), horizontal=True, on_change=update_input_params)

# Data Input Section
c30, c31 = st.columns([.5, 1])

# Col for Input by File
with c30:
    label = "📤 Upload File"
    st.session_state.input_file = st.file_uploader(label, disabled=st.session_state.up_off, on_change=update_button, type=['txt', 'pdf', 'docx'], key='text_upload')

# Col for Input by Text
with c31:
    label = "⌨️ Enter Text"
    st.session_state.input_text = st.text_area(label, value=st.session_state.box_value,  height=170, placeholder="Type or Paste the text you would like to summarise here...", on_change=update_button, key='text_box', disabled=st.session_state.in_off)

# Data Processing Section
c30, c31, c32 = st.columns([.25, .25, 1])

# Col for Start Button
with c30:
    label = "Generate Summary"
    st.session_state.run = st.button(label, on_click=run_analysis, disabled=st.session_state.generate_button)

with c31:
    label = "Download Summary"
    st.download_button(label, ''.join(st.session_state.final_output), file_name="summary.txt", disabled=st.session_state.download_off)

# Results Section
label = "✅ Lay Summary"
st.text_area(label, value=st.session_state.final_output, disabled=True, height=160, placeholder='Your summary will appear here when it has been generated...')
