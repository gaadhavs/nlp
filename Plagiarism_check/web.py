import streamlit as st
import checker

st.title("Plagarism Checker")
st.write("This is a simple web app to check for plagarism in a given text.")

glider = st.slider("Select a value", 0, 100)

text1 = st.text_area("Enter text 1")
text2 = st.text_area("Enter text 2")

if st.button("Check Plagarism"):
    out = checker.checkPlagarism(text1, text2, glider)
    if out[0]:
        st.error("The texts are plagarised.")
    else:
        st.success("The texts are not plagarised.")

    st.write("Similarity: ", out[1])
