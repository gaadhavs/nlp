import streamlit as st
import mask_predictor

st.title("Masked Word Prediction with BERT")
st.subheader("Predict the masked word in the sentence")

with st.expander("How to use this website"):
    st.write("This is a web app to predict the masked word in a sentence. You need to enter the sentence with a masked word. The model will predict the masked word. Example: 'The quick brown [MASK] jumps over the lazy dog', the model will predict 'fox'.")
    
Masked_Sentence = st.text_area("Masked_sentence", placeholder="Type Here with [MASK] in place of masked word")

if st.button('Predict', key=1):
    st.write("The predicted tokens are:")
    out = ""
    for i in mask_predictor.predict(Masked_Sentence):
        out += i + ", "
    st.write(out[0:len(out)-2])