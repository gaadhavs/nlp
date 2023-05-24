import models
import streamlit as st
import pandas as pd


# model = models.models()


tab7, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["|", "Emotion Analyzer", 
                            "Plagiarism checker", 
                            "Question Answering", 
                            "Masked Word", 
                            "Text Summarizer", 
                            "Legitimacy Checker"])

with tab1:
   st.header("Emotion Analyzer")

   text = st.text_area("Text", placeholder = "Enter text here")

   if st.button("Analyze"):
      prediction = models.getModel().emotionAnalyzer(text)
      
      def getVal(key):
         for i in prediction:
               if i['label'] == key:
                  return i['score']
               
      chart_data = pd.DataFrame([[getVal("sadness"), getVal("joy"), getVal("love"), getVal("anger"), getVal("fear"), getVal("surprise")]], columns = ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"])

      st.bar_chart(chart_data)
   

with tab2:
   st.header("Plagiarism checker")
   st.write("This is a simple web app to check for plagiarize in a given text.")
   
   glider = st.slider("Select a threshold", 0, 100)

   text1 = st.text_area("Paragraph 1", placeholder = "Enter the first paragraph here.")
   text2 = st.text_area("Paragraph 2", placeholder = "Enter the second paragraph here.")

   if st.button("Check Plagiarism"):
       out2 = models.getModel().checkPlagarism(text1, text2, glider)
       if out2[0]:
           st.error("The texts are plagiarized.")
       else:
           st.success("The texts are not plagiarized.")

       st.write("Similarity: ", out2[1])

   

with tab3:
   st.header("Question Answering")

   para = st.text_area("Paragraph", placeholder = "Enter the paragraph here.")
   question = st.text_input("Question", placeholder = "Enter the question here.")

   if st.button("Find Answer"):
      out3 = models.getModel().question_answering(question, para)
      st.write("Answer: ", out3)



with tab4:
   st.header("Masked Word Prediction")

   text_area5 = st.text_area("Masked Sentence", placeholder = "Put '_' inplace of the missing word...")


   if st.button("Predict"):
      text_area5 = text_area5.replace("_", "[MASK]")
      out5 = models.getModel().maskedWordPredictor(text_area5)

      data = []

      for i in range(len(out5[0])):
         data.append([out5[0][i], out5[1][i]])

      df = pd.DataFrame(data, columns = ("Prediction", "Probability"))

      st.table(df)
   

with tab5:
   st.header("Text Summarizer")

   slider = st.slider("Select the number of sentences", 4, 10) 

   text_area = st.text_area("Para Text", placeholder = "Enter the Paragraph here to summarize.", height = 450)

   if st.button("Summarize"):

      text_area = text_area.replace("\n", " ")
      lines = text_area.split(".")

      
      combinedList = []

      while len(lines) > slider:
         out = ""
         for i in range(slider):
            out += lines.pop(0)
            out += "."
         
         combinedList.append(models.getModel().textSummerier(out))

      out = ""
      for i in lines:
         out += i
         out += "."
      combinedList.append(models.getModel().textSummerier(out))

      finalOut = ""
      for i in combinedList:
         finalOut += i
         finalOut += ". "
      
      st.write("Summary: ")
      st.write("")
      st.write(finalOut)

      

with tab6:
   st.header("Legitimacy Checker")
   glider6 = st.slider("Select a rating", 1, 5)

   text_area = st.text_area("Review", placeholder = "Enter the review here.")

   if st.button("Validate"):
      out6 = models.getModel().compare(glider6, text_area)

      if out6:
         st.success("The review is legitimate.")
      else:
         st.error("The review is not legitimate.")
   


with tab7:
   st.header("About")
   st.write("This is a simple web app that performs various NLP tasks using different BERT models.")
   st.write("Select the task from the tabs above and enter the required inputs.")
   st.write("The app will then perform the task and display the results.")
   
























css = '''
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:0.8rem;
    }
</style>
'''

st.markdown(css, unsafe_allow_html=True)