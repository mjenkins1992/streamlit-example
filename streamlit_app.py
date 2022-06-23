from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import time
from io import StringIO
from googletrans import Translator

translator = Translator()
# CALLBACKS
def update_input_params():
    st.session_state.input_text = ""
    st.session_state.up_off = not(st.session_state.up_off)
    st.session_state.in_off = not(st.session_state.in_off)
    return

def run_analysis():
    with st.spinner('Generating Summary...'):
        if st.session_state["text_box"]:
            st.session_state.output_text = translator.translate(st.session_state["text_box"], dest='fr')
        if st.session_state.output_text:
            st.session_state.download_on = True
    st.success('Complete!')
    st.write(st.session_state.output_text)
    return

def update_button():
    if st.session_state["text_box"]:
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
if 'up_off' not in st.session_state:
	st.session_state.up_off = False
if 'in_off' not in st.session_state:
	st.session_state.in_off = True
if 'download_on' not in st.session_state:
	st.session_state.download_on = False
if 'stat_val' not in st.session_state:
	st.session_state.stat_val = 0
if 'output_text' not in st.session_state:
	st.session_state.output_text = ""
if 'generate_button' not in st.session_state:
	st.session_state.generate_button = True

# Input Type Selection
st.session_state.in_type = st.radio("How would you like to input your data?",
     ('File Upload', 'Free Text'), horizontal=True, on_change=update_input_params)

# Data Input Section
c30, c31 = st.columns([.5, 1])

# Col for Input by File
with c30:
    label = "📤 Upload File"
    st.session_state.input_text = st.file_uploader(label, disabled=st.session_state.up_off)

# Col for Input by text
with c31:
    label = "⌨️ Enter Text"
    st.session_state.input_text = st.text_area(label, height=160, placeholder="Type or Paste the text you would like to summarise here...", on_change=update_button, key='text_box', disabled=st.session_state.in_off)

# Data Processing Section
c30, c31 = st.columns([.25, 1])

# Col for Start Button
with c30:
    label = "Generate Summary"
    st.session_state.run = st.button(label, on_click=run_analysis, disabled=st.session_state.generate_button)

st.write(st.session_state.input_text)

# Results Section
#with c31:
#    st.header("📥 Summary Download")
#    label = "Download Summary"
#    st.download_button(label, st.session_state.output_text, file_name="summary.txt", disabled=st.session_state.download_on)
