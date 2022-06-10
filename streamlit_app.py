from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import time

#"""
## Welcome to Streamlit!
#
#Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:
#
#If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
#forums](https://discuss.streamlit.io).
#
#In the meantime, below is an example of what you can do with just a few lines of code:
#"""



st.set_page_config(
    page_title="Lay Summary Generator",
    page_icon="📄",
)

st.title("📄 Lay Summary Generator")

with st.expander("ℹ️ - About this app", expanded=True):
    st.write(
        """     
        - Some blurb here about the app
        - Usage instructions
        """
    )

    st.markdown("")

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

c30, c31 = st.columns([.5, 1])
#col for Input by File
with c30:
    st.text("📤 Upload File to Summarise")
    label = ""
    st.file_uploader(label)

# Col for Input by text
with c31:
    st.header️("⌨️ Enter or Paste Text to Summarise")
    label = ""
    st.text_area(label, height=None, placeholder="Type or Paste the text you would like to summarise here...")

c30, c31 = st.columns([.25, 1])
#col for Start Button
with c30:
    label = "Generate Summary"
    st.button(label)
# Col for Prog Bars
with c31:
    st.text("⏳ Progress...")
    stat_val=0
    st.progress(stat_val)

st.subheader("📥 Summary Download")
st.header("")
label = "Summary Download"
data = "This is a temporary string"
st.download_button(label, data, file_name="summary.txt")

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
