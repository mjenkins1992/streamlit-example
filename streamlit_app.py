from collections import namedtuple
import altair as alt
import math
import pandas as pd
import numpy
import streamlit as st
import time
from io import StringIO
from googletrans import Translator
from PyPDF2 import PdfFileReader
import docx2txt
#import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pickle

path = "./model/"
model = AutoModelForSeq2SeqLM.from_pretrained(path, local_files_only=True)
#model.cuda()
tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)


translator = Translator()

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

def run_analysis():
    with st.spinner('Generating Summary...'):
        if st.session_state["text_box"]:
            raw_text = st.session_state.input_text
        elif st.session_state["text_upload"]:
            if st.session_state.input_file.type == "text/plain":
                raw_text = str(st.session_state.input_file.read(),"utf-8")
            elif st.session_state.input_file.type == "application/pdf":
                raw_text = read_pdf(st.session_state.input_file)
            elif st.session_state.input_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" :
                raw_text = docx2txt.process(st.session_state.input_file)

        st.session_state.output_text = translator.translate(raw_text, src='en', dest='fr')
        st.session_state.final_output = st.session_state.output_text.text
        st.session_state.download_off = False
        st.session_state.box_value = raw_text
        #st.success('Complete!')
    return

def run_analysis2():
    with st.spinner('Generating Summary...'):
        if st.session_state["text_box"]:
            raw_text = st.session_state.input_text
        elif st.session_state["text_upload"]:
            if st.session_state.input_file.type == "text/plain":
                raw_text = str(st.session_state.input_file.read(),"utf-8")
            elif st.session_state.input_file.type == "application/pdf":
                raw_text = read_pdf(st.session_state.input_file)
            elif st.session_state.input_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" :
                raw_text = docx2txt.process(st.session_state.input_file)

        #st.session_state.output_text = translator.translate(raw_text, src='en', dest='fr')
        #st.session_state.final_output = st.session_state.output_text.text
        st.session_state.download_off = False
        st.session_state.box_value = raw_text

        to_pred = tokenizer(raw_text, padding="max_length", max_length=4096, return_tensors="pt", truncation=True)
        #input_ids=to_pred["input_ids"].cuda()
        #attention_mask=to_pred["attention_mask"].cuda()
        #global attention on special tokens
        #global_attention_mask = torch.zeros_like(attention_mask)
        global_attention_mask = numpy.zeros_like(attention_mask)
        global_attention_mask[:, 0] = 1
        predicted_ids = model.generate(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
        st.session_state.final_output = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)

        #st.success('Complete!')
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
        - Some blurb here about the app
        - Usage instructions
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
    st.session_state.run = st.button(label, on_click=run_analysis2, disabled=st.session_state.generate_button)

with c31:
    label = "Download Summary"
    st.download_button(label, st.session_state.final_output, file_name="summary.txt", disabled=st.session_state.download_off)

# Results Section
with c32:
    label = "✅ Lay Summary"
    st.text_area(label, value=st.session_state.final_output, disabled=True, height=160, placeholder='Your summary will appear here when it has been generated...')
