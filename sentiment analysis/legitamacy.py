import streamlit as st
#import pyautogui
import predictor


st.title("Legitamecy Analysis with BERT")
st.subheader("Sentiment Analysis of reviews using BERT")

with st.expander("How to use this website"):
    st.write("This is a web app to analyze the legitemacy of reviews. You need to enter the ratings given by the user and the review text. The model will predict the sentiment of the review and will tell you whether the review is legit or not.")
    




Rating = st.slider("Rating", 1, 5)
Review = st.text_area("Review", placeholder="Type Here")






if st.button('Predict', key=1):
    if predictor.compare(Rating, Review):
        st.success("The review is legit")
    else:
        st.error("The review is not legit")
    

    

# if st.button('Reset all parameters', key=14):
#             pyautogui.hotkey("ctrl","F5")

