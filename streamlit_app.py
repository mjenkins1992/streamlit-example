from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st

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
    page_icon="👩‍⚕️",
)

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

c30, c31, c32 = st.columns([3, 3, 3])

#col for parameters

#col for Upload
with c31:
    st.title("👆 Input Upload")
    label = ""
    st.file_uploader(label)

#col for Download
with c32:
    st.title("👇 Output Download")
    st.header("")
    label = "Summary Download"
    data = "This is a temporary string"
    st.download_button(label, data, file_name="summary.txt")

#with st.echo(code_location='below'):
total_points = st.slider("Number of points in spiral", 1, 5000, 2000)
num_turns = st.slider("Number of turns in spiral", 1, 100, 9)

Point = namedtuple('Point', 'x y')
data = []

points_per_turn = total_points / num_turns

for curr_point_num in range(total_points):
    curr_turn, i = divmod(curr_point_num, points_per_turn)
    angle = (curr_turn + 1) * 2 * math.pi * i / points_per_turn
    radius = curr_point_num / total_points
    x = radius * math.cos(angle)
    y = radius * math.sin(angle)
    data.append(Point(x, y))

st.altair_chart(alt.Chart(pd.DataFrame(data), height=500, width=500)
    .mark_circle(color='#0068c9', opacity=0.5)
    .encode(x='x:Q', y='y:Q'))
