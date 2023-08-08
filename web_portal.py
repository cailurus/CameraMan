import streamlit as st


st.title("Uber pickups in NYC")


picture = st.camera_input("Take a picture")

if picture:
    st.image(picture)
