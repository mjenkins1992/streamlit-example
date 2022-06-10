from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import time
from io import StringIO 

#Page Config
st.set_page_config(
    page_title="Lay Summary Generator",
    page_icon="📄",
)

#Page Title
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

# Input Type Selection
in_type = st.radio("How would you like to input your data?",
     ('File Upload', 'Free Text'), horizontal=True)
up_off=False
in_off=True
if in_type == "File Uplaod":
    up_off = False
    in_off = True
elif in_type == "Free Text":
    up_off = True
    in_off = False    

# Data Input Section
c30, c31 = st.columns([.5, 1])

# Col for Input by File
with c30:
    st.text("📤 Upload File to Summarise")
    label = ""
    input_text = st.file_uploader(label, disabled=up_off)
#    if input_text is not None:
#        for line in input_text:
#            st.write(line)
	
# Col for Input by text
with c31:
    st.text("⌨️ Enter Text to Summarise")
    label = ""
    input_text = st.text_area(label, height=160, placeholder="Type or Paste the text you would like to summarise here...", disabled=in_off)
    st.write(input_text)

	
# Temp for Prog Bar Demo
download_on = False
if 'stat_val' not in st.session_state:
	st.session_state.stat_val = 0
def run_analysis():
    if st.session_state.stat_val < 100:
        st.session_state.stat_val += 25
    else:
        st.session_state.stat_val = 0
    
    if input_file != NULL
	download_on = True

# Data Processing Section
c30, c31 = st.columns([.25, 1])

# Col for Start Button
with c30:
    label = "Generate Summary"
    run = False
    run = st.button(label, on_click=run_analysis)
    st.write(st.session_state.stat_val)
# Col for Prog Bars
with c31:
    st.text("⏳ Progress...")
    st.progress(st.session_state.stat_val)

# Results Section
st.subheader("📥 Summary Download")
st.header("")
label = "Summary Download"
data = "This is a temporary string"
st.download_button(label, data, file_name="summary.txt", disabled=download_on)

#with st.echo(code_location='below'):
#total_points = st.slider("Number of points in spiral", 1, 5000, 2000)
#num_turns = st.slider("Number of turns in spiral", 1, 100, 9)
#
#Point = namedtuple('Point', 'x y')
#data = []
#
#points_per_turn = total_points / num_turns
#
#for curr_point_num in range(total_points):
#    curr_turn, i = divmod(curr_point_num, points_per_turn)
#    angle = (curr_turn + 1) * 2 * math.pi * i / points_per_turn
#    radius = curr_point_num / total_points
#    x = radius * math.cos(angle)
#    y = radius * math.sin(angle)
#    data.append(Point(x, y))
#
#st.altair_chart(alt.Chart(pd.DataFrame(data), height=500, width=500)
#    .mark_circle(color='#0068c9', opacity=0.5)
#    .encode(x='x:Q', y='y:Q'))
