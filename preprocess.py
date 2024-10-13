import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
import torch

def generate_embeddings(df: pd.DataFrame, primary_key: str, mode: str = 'tfidf'):
    """
    Generate embeddings for each row in a DataFrame based on the chosen mode.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    primary_key (str): The column name of the primary key.
    mode (str): Embedding mode, either 'tfidf' or 'bert'. Defaults to 'tfidf'.
    
    Returns:
    dict: A dictionary where keys are primary key values, and values are row embeddings.
    """
    if primary_key not in df.columns:
        raise ValueError(f"Primary key column '{primary_key}' not found in DataFrame.")
    text_data = df.drop(columns=[primary_key])
    if mode == 'tfidf':
        text_rows = text_data.astype(str).agg(' '.join, axis=1)
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(text_rows)
        embeddings = tfidf_matrix.toarray()
    # Mode 2: BERT embeddings
    elif mode == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        embeddings = []
        for _, row in text_data.iterrows():
            row_text = ' '.join(row.astype(str))
            inputs = tokenizer(row_text, return_tensors='pt', max_length=512, truncation=True, padding=True)
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()
            embeddings.append(cls_embedding.flatten())
        embeddings = torch.tensor(embeddings)
    else:
        raise ValueError("Mode must be either 'tfidf' or 'bert'.")
    # Create dictionary with primary key and corresponding embeddings
    embeddings_dict = dict(zip(df[primary_key], embeddings))
    return embeddings_dict
