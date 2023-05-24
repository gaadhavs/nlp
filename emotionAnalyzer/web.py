import streamlit as st
import pandas as pd
import analyzer

st.title('Emotion Analyzer')


text = st.text_area("", placeholder = "Enter text here")

if st.button("Analyze"):
    prediction = analyzer.emotionAnalyzer(text)
    
    def getVal(key):
        for i in prediction:
            if i['label'] == key:
                return i['score']
            
    chart_data = pd.DataFrame([[getVal("sadness"), getVal("joy"), getVal("love"), getVal("anger"), getVal("fear"), getVal("surprise")]], columns = ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"])

    st.bar_chart(chart_data)