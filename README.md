# Balanced Hierarchical Text Clustering (BHTC)

Clusters short pieces of text into groups of equal size, in a hierarchy, using embeddings and a primary key to keep related items together.

Ordinary clustering gives you groups of wildly different sizes. That is fine for exploring data and useless when each group has a fixed capacity, such as a hostel or a mentoring cohort. BHTC keeps the hierarchy but forces balanced groups.

## Where it was used

- Allocating hostels for about 800 first-year students at IIT Patna
- Forming mentorship groups from student preferences and attributes

## Try it

Live demo: https://balanced-clustering.streamlit.app

Run locally:

    pip install -r requirements.txt
    streamlit run app.py

## Files

- `preprocess.py` turns raw text into embeddings
- `hac.py` the hierarchical clustering with the balancing step
- `app.py` the Streamlit front end

Built with Parth Agarwal in August 2024.
