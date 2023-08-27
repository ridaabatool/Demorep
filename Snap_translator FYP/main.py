from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
import easyocr
from googletrans import Translator
from autocorrect import Speller
import re
import streamlit as st
from streamlit_cropper import st_cropper
import cv2
from streamlit_image_select import image_select
from imutils.perspective import four_point_transform

def preprocess(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # threshold the image using Otsu's thresholding method
    _, thresh = cv2.threshold(gray, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    return thresh


def draw_boxes(image, bounds, color='yellow', width=2):
    draw = ImageDraw.Draw(image)
    for bound in bounds:
        p0, p1, p2, p3 = bound[0]
        draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)
    return image




st.title(" Snap Translator")


st.markdown('<p style="font-family:sans-serif; color:Green; font-size: 32px;">Choose Language</p>', unsafe_allow_html=True)
lang = st.radio('Language :',
('en', 'ur'))
if lang == 'ur':
    im = image_select("Urdu images", ["","img/sample_img9.jpg", "img/sample_img10.jpg", "img/sample_img_11.jpg"])
    st.write("img:", im)
elif lang =='en':
    im = image_select("English images", ["","img/sample_img_eng.jpg", "img/sample_img_eng4.jpg", "img/sample_img_eng3.jpg"])

if im == "":
    st.markdown('<p style="font-family:sans-serif; color:Green; font-size: 32px;">Choose Image</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Uploading image", type="jpg")
else : 
    uploaded_file = im
if uploaded_file is not None:

    if lang == 'ur':
        lang_target = 'en'
        im = Image.open(uploaded_file)
        st.markdown(f'<p style="font-family:sans-serif; color:Red; font-size: 20px;">Double click on the zone after finishing croping then click on the button:</p>', unsafe_allow_html=True)
        cropped_img = st_cropper(im, realtime_update=False, aspect_ratio=None)
        if st.button(' Done Crop'):
            st.image(cropped_img, caption = 'cropped img', use_column_width='auto')
            cropped_img.save("cropped_img.jpg")
            cropped_img = Image.open("cropped_img.jpg")
            opencvImage = cv2.cvtColor(np.array(cropped_img), cv2.COLOR_RGB2BGR)
            preproc_im = preprocess(opencvImage)
            st.image(preproc_im, caption = 'preprocessed img', use_column_width=True)
            st.write("")
            st.markdown('<p style="font-family:sans-serif; color:Red; font-size: 32px;">Preparing the reader...</p>', unsafe_allow_html=True)
            reader = easyocr.Reader(['ur'])
            bounds = reader.readtext(cropped_img, text_threshold = 0.5, paragraph = True)
            #draw_boxes(im, bounds)
            text = []
            for i in range(len(bounds)):
                text.append(bounds[i][-1])
                #print(bounds[i][-1])
            text = '\n'.join(text)
            st.markdown('<p style="font-family:sans-serif; color:Blue; font-size: 25px;"> Original text extracted from the picture :</p>', unsafe_allow_html=True)
            st.write(text)
            translator = Translator()
            trans_ur = translator.translate(text,src ='ur', dest = 'en').text
            st.write('')
            st.markdown(f'<p style="font-family:sans-serif; color:Blue; font-size: 25px;">Translated text from Urdu to English:</p>', unsafe_allow_html=True)
            st.write(trans_ur)
            with open('saved/text.txt', 'w',encoding='utf-8') as f:
                f.write('Original text : ')
                f.write( text)
                f.write('\n')
                f.write('Translated text : ')
                f.write(trans_ur)
                st.write('Saved !')

    elif lang == 'en': 
        lang_target = 'ur'
        im = Image.open(uploaded_file)        
        st.markdown(f'<p style="font-family:sans-serif; color:Red; font-size: 20px;">Double click on the zone after finishing croping then click on the button:</p>', unsafe_allow_html=True)
        cropped_img = st_cropper(im, realtime_update=False, aspect_ratio=(1,1))
        if st.button(' Done Crop'):
            st.image(cropped_img, caption = 'cropped img', use_column_width=True)
            cropped_img.save("cropped_img.jpg")
            cropped_img = Image.open("cropped_img.jpg")
            opencvImage = cv2.cvtColor(np.array(cropped_img), cv2.COLOR_RGB2BGR)
            preproc_im = preprocess(opencvImage)
            st.image(preproc_im, caption = 'preprocessed img', use_column_width=True)
            st.markdown('<p style="font-family:sans-serif; color:Red; font-size: 32px;">Preparing the reader...</p>', unsafe_allow_html=True)
            reader = easyocr.Reader(['en'])
            bounds = reader.readtext(cropped_img, text_threshold = 0.5, paragraph = True)
            #draw_boxes(im, bounds)
            text = []
            for i in range(len(bounds)):
                text.append(bounds[i][-1])
                #print(bounds[i][-1])
            text = '\n'.join(text)
            spell = Speller(lang='en')
            text = re.sub('[^A-Za-z0-9\.,]+', ' ', text)
            text_corrected = spell(text)
            st.markdown('<p style="font-family:sans-serif; color:Blue; font-size: 25px;"> Original text extracted from the picture :</p>', unsafe_allow_html=True)
            st.write(text_corrected)
            translator = Translator()
            trans_en = translator.translate(text_corrected,src =lang, dest = 'ur').text
            st.write('')
            st.markdown(f'<p style="font-family:sans-serif; color:Blue; font-size: 25px;">Translated text from English to Urdu:</p>', unsafe_allow_html=True)
            st.write(trans_en)
            with open('saved/text.txt', 'w', encoding='utf-8') as f:
                    f.write('Original text : ')
                    f.write( text_corrected)
                    f.write('\n')
                    f.write('Translated text : ')
                    f.write(trans_en)
                    st.write('Saved !')

