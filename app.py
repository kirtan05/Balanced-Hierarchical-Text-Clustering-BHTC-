import streamlit as st
import pandas as pd
from hac import hierarchical_clustering
from preprocess import generate_embeddings
st.markdown(
    """
    <style>
    body {
        background-color: #f5f5f5;
        color: #333333;
    }
    .main {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    }
    h1 {
        color: #2E86C1;
        text-align: center;
        font-size: 3em;
        font-family: 'Arial', sans-serif;
    }
    .stButton button {
        background-color: #2E86C1;
        color: white;
        border-radius: 8px;
        padding: 10px;
        font-size: 16px;
        width: 100%;
    }
    .stButton button:hover {
        background-color: #1B4F72;
    }
    .stFileUploader label {
        font-size: 16px;
        color: #333333;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Equal Size Hierarchical Textual Clustering ")
# Step 1: Upload CSV file
st.subheader("Step 1: Upload CSV File")
uploaded_file = st.file_uploader("", type=["csv"])
if uploaded_file:
    # Read the file into a pandas dataframe
    df = pd.read_csv(uploaded_file)
    st.write("Preview of Uploaded Data:")
    st.dataframe(df.head(10))

    # Step 2: Choose number of levels for clustering
    st.subheader("Step 2: Select the Number of Clustering Levels")
    num_levels = st.number_input("Number of levels:", min_value=1, step=1, value=3)

    # Step 3: Specify the number of items per cluster for each level
    st.subheader("Step 3: Specify the Number of Items per Cluster (Comma-separated)")
    num_items_per_level = st.text_input(
        "Items per cluster (e.g., 5,3,2)", value="5,3,2"
    )
    num_items_per_level = [int(i) for i in num_items_per_level.split(",")]

    # Step 4: Choose the primary key index (column name)
    st.subheader("Step 4: Choose the Primary Key Column")
    primary_key_column = st.selectbox("Primary Key Column", df.columns)
    st.subheader("Step 5: Choose Embedding Mode")
    option = st.selectbox(
    "Choose Embedding Mode (bert will take time on large data   )",
    ("tfidf", "bert"),
    )
    st.write("You selected:", option)
    st.subheader("Run the Clustering Algorithm")
    if st.button("Run Clustering"):
        embeddings_dict = generate_embeddings(df,primary_key_column, option)
        result = hierarchical_clustering(embeddings_dict, num_items_per_level) 
        st.write("Clustering Results:")
        st.json(result, expanded=2)
        st.download_button(
        label="Download JSON",
        data=result,
        file_name="clustering_result.json",
        mime="application/json"
        )
st.markdown(
    """
    <style>
    .footer-heart {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        text-align: center;
        font-family: 'Arial', sans-serif;
        padding: 10px 0;
        font-size: 16px;
        box-shadow: 0px -1px 5px rgba(0, 0, 0, 0.1);
    }
    .footer-heart a {
        border-bottom: 1px solid #ffffff; /* Original purple for the underline */
        color: #ffffff; 
        padding-bottom: .25em;
        text-decoration: none;
    }
    .footer-heart a:hover {
        color: #2196f3; 
        background-image: url("data:image/svg+xml;charset=utf8,%3Csvg id='squiggle-link' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink' xmlns:ev='http://www.w3.org/2001/xml-events' viewBox='0 0 20 4'%3E%3Cstyle type='text/css'%3E.squiggle{animation:shift .3s linear infinite;}@keyframes shift {from {transform:translateX(0);}to {transform:translateX(-20px);}}%3C/style%3E%3Cpath fill='none' stroke='%23453886' stroke-width='2' class='squiggle' d='M0,3.5 c 5,0,5,-3,10,-3 s 5,3,10,3 c 5,0,5,-3,10,-3 s 5,3,10,3'/%3E%3C/svg%3E");
        background-position: bottom;
        background-repeat: repeat-x;
        background-size: 20%;
        border-bottom: 0;
        padding-bottom: .3em;
    }
    .emoji {
        vertical-align: middle;
        height: 20px;
        width: 20px;
    }
    </style>
    <div class="footer-heart">
        Made with <img class="emoji" alt="heart" src="https://github.githubassets.com/images/icons/emoji/unicode/2764.png" /> 
        by <a href="https://www.linkedin.com/in/kirtan-jain-19ba55223/" target="_blank">Kirtan Jain</a> and 
        <a href="https://www.linkedin.com/in/parthagarwal04/" target="_blank">Parth Agarwal</a>
    </div>
    """,
    unsafe_allow_html=True
)
