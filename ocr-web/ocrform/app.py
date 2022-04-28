import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import ocr_form_mink

def app():

    fig = plt.figure()

    st.title("OCR Form to Text")



    file_uploaded = st.file_uploader("Choose PDF file", type=["pdf"])



    class_btn = st.button("Extract")

    if file_uploaded is not None:
        with open(os.path.join("form_test",file_uploaded.name),"wb") as f:
            f.write((file_uploaded).getbuffer())

        file_path=f'form_test/{file_uploaded.name}'
        #st.write(file_path)

    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload a PDF file")
        else:
            with st.spinner('Model working....'):


                pre_result = ocr_form_mink.getMain(file_path)
                st.markdown('<font size=6><b>Result :</b></font>', unsafe_allow_html=True)
                st.write(pre_result)

                os.remove(file_path)

app()


