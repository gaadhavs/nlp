import streamlit as st
import questionAnswer

st.title("Question Answer")
st.write("This is a simple web app to check find answers for questions from a given paragraph.")

para = st.text_area("", placeholder = "Enter the paragraph here.")
question = st.text_input("", placeholder = "Enter the question here.")

if st.button("Find Answer"):
    out = questionAnswer.question_answering(question, para)
    st.write("Answer: ", out)

