# import streamlit as st
# import numpy as np
# import nltk
# import re
# import string
# import joblib
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from gensim.models import Word2Vec
# from PIL import Image  # To handle images
# import time  # For potential timing adjustments

# # --- Page Configuration ---
# st.set_page_config(
#     page_title="BBC News Classifier",
#     page_icon="📰",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # --- Styling ---
# st.markdown(
#     """
#     <style>
#     .reportview-container .main .block-container{
#         max-width: 80%;
#         padding-top: 2rem;
#         padding-bottom: 2rem;
#     }
#     .stTextArea textarea {
#         height: 200px;
#     }
#     .big-font {
#         font-size:1.5rem !important;
#     }
#     .smaller-font {
#         font-size:0.8rem !important;
#         color: #888;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# # --- Sidebar ---
# with st.sidebar:
#     # Load and display the BBC logo (or a placeholder)
#     try:
#         bbc_logo = Image.open(r"BBC_World_News_2022.svg.png")  # Ensure you have this image
#         st.image(bbc_logo, width=200)
#     except FileNotFoundError:
#         st.image(np.zeros((100, 100, 3)), width=200) # Display blank if logo is missing
#         st.warning("BBC News Logo not found. Please add 'bbc_news_logo.png' to the directory.")

#     st.title("BBC News Classifier")
#     st.markdown("This app classifies news articles into the following categories:")
#     st.markdown("*Business, Entertainment, Politics, Sport, Tech*")
#     st.markdown("---")
#     st.info("Please enter the full text of a news article in the box. The classifier will then predict its category.")

# # --- NLTK Downloads ---
# try:
#     nltk.data.find('punkt')
#     nltk.data.find('stopwords')
#     nltk.data.find('wordnet')
# except LookupError:
#     nltk.download('punkt')
#     nltk.download('stopwords')
#     nltk.download('wordnet')

# # --- Load Models ---
# @st.cache_resource
# def load_models():
#     model = joblib.load("news_classifier_model.pkl")
#     w2v_model = Word2Vec.load("word2vec.model")
#     return model, w2v_model

# model, w2v_model = load_models()

# # --- Preprocessing and Vectorization ---
# def preprocess_text(text):
#     lemmatizer = WordNetLemmatizer()
#     text = re.sub(r'\W', ' ', text.lower())
#     tokens = nltk.word_tokenize(text)
#     tokens = [lemmatizer.lemmatize(word)
#               for word in tokens
#               if word not in stopwords.words('english') and word not in string.punctuation]
#     return tokens


# def vectorize_text(tokens):
#     vec = np.zeros(100)
#     count = 0
#     for word in tokens:
#         if word in w2v_model.wv:
#             vec += w2v_model.wv[word]
#             count += 1
#     return vec / count if count != 0 else vec

# # --- Main Section ---
# st.title("News Article Classification")

# user_input = st.text_area("Enter News Article Text Here:", "")

# if st.button("✨ Classify"):
#     if user_input.strip() == "":
#         st.warning("Please enter some text.")
#     else:
#         tokens = preprocess_text(user_input)
#         vector = vectorize_text(tokens).reshape(1, -1)
#         prediction = model.predict(vector)
#         st.markdown("<p class='big-font'><b>Predicted Category:</b></p>", unsafe_allow_html=True)
#         st.markdown(f"<p class='big-font'>{prediction[0]}</p>", unsafe_allow_html=True)

#         st.balloons()  # Launch the balloons!
#         # Optionally, you can add a short delay if you want the balloons to appear after the text
#         time.sleep(1)


# # --- Footer ---
# st.markdown("---")
# st.caption("Developed with Streamlit and Python")

import streamlit as st
import numpy as np
import nltk
import re
import string
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from PIL import Image
import time
import os
import sys

# --- NLTK Data Path Configuration ---
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")

# --- Forcefully Set NLTK Data Path ---
nltk.data.path = [nltk_data_path]  # Override the entire path!
st.warning(f"Setting NLTK data path to: {nltk_data_path}")

# --- Check and Create NLTK Data Directory ---
if not os.path.exists(nltk_data_path):
    try:
        os.makedirs(nltk_data_path)
        st.warning(f"Created NLTK data directory at: {nltk_data_path}")
    except Exception as e:
        st.error(f"Failed to create NLTK data directory: {e}")
        st.stop()
elif not os.access(nltk_data_path, os.W_OK):
    st.error(f"NLTK data directory is not writable: {nltk_data_path}")
    st.stop()
else:
    st.info(f"NLTK data directory exists and is writable: {nltk_data_path}")

# --- NLTK Downloads and Debugging ---
def download_nltk_data(resource_name, download_message):
    try:
        nltk.data.find(resource_name)
    except LookupError:
        st.warning(download_message)
        try:
            nltk.download(resource_name, download_dir=nltk_data_path)
        except Exception as e:
            st.error(f"Error during download of {resource_name}: {e}")
            st.stop()
        try:
            nltk.data.find(resource_name)
            st.success(f"{resource_name} downloaded successfully to: {nltk_data_path}")
        except LookupError:
            st.error(
                f"Failed to find {resource_name} after download.  NLTK data path: {nltk.data.path}, CWD: {os.getcwd()}, sys.path: {sys.path}")
            st.stop()

download_nltk_data('tokenizers/punkt', "Downloading Punkt tokenizer data...")
download_nltk_data('corpora/stopwords', "Downloading stopwords data...")
download_nltk_data('corpora/wordnet', "Downloading wordnet data...")
download_nltk_data('omw-1.4', "Downloading omw-1.4 data...")

# --- Debugging Output ---
st.sidebar.markdown("---")
st.sidebar.subheader("NLTK Data Path:")
st.sidebar.code(str(nltk.data.path))
st.sidebar.subheader("Current Working Directory:")
st.sidebar.code(os.getcwd())
st.sidebar.subheader("sys.path:")
st.sidebar.code(str(sys.path))

# --- Page Configuration ---
st.set_page_config(
    page_title="BBC News Classifier",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Styling ---
st.markdown(
    """
    <style>
    .reportview-container .main .block-container{
        max-width: 80%;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stTextArea textarea {
        height: 200px;
    }
    .big-font {
        font-size:1.5rem !important;
    }
    .smaller-font {
        font-size:0.8rem !important;
        color: #888;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar ---
with st.sidebar:
    # Load and display the BBC logo
    try:
        bbc_logo = Image.open("BBC_World_News_2022.svg.png")
        st.image(bbc_logo, width=200)
    except FileNotFoundError:
        st.image(np.zeros((100, 100, 3)), width=200)
        st.warning("BBC News Logo not found. Please add 'BBC_World_News_2022.svg.png' to the directory.")

    st.title("BBC News Classifier")
    st.markdown("This app classifies news articles into the following categories:")
    st.markdown("*Business, Entertainment, Politics, Sport, Tech*")
    st.markdown("---")
    st.info("Please enter the full text of a news article in the box. The classifier will then predict its category.")

# --- Load Models ---
@st.cache_resource
def load_models():
    model = joblib.load("news_classifier_model.pkl")
    w2v_model = Word2Vec.load("word2vec.model")
    return model, w2v_model

model, w2v_model = load_models()

# --- Preprocessing and Vectorization ---
def preprocess_text(text):
    lemmatizer = WordNetNetLemmatizer()
    text = re.sub(r'\W', ' ', text.lower())
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word)
              for word in tokens
              if word not in stopwords.words('english') and word not in string.punctuation]
    return tokens


def vectorize_text(tokens):
    vec = np.zeros(100)
    count = 0
    for word in tokens:
        if word in w2v_model.wv:
            vec += w2v_model.wv[word]
            count += 1
    return vec / count if count != 0 else vec

# --- Main Section ---
st.title("News Article Classification")

user_input = st.text_area("Enter News Article Text Here:", "")

if st.button("✨ Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        tokens = preprocess_text(user_input)
        vector = vectorize_text(tokens).reshape(1, -1)
        prediction = model.predict(vector)
        st.markdown("<p class='big-font'><b>Predicted Category:</b></p>", unsafe_allow_html=True)
        st.markdown(f"<p class='big-font'>{prediction[0]}</p>", unsafe_allow_html=True)
        st.balloons()
        time.sleep(1)

# --- Footer ---
st.markdown("---")
st.caption("Developed with Streamlit and Python")


