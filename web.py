from flask import Flask, render_template, request
import urllib.parse
import sys
from elasticsearch import Elasticsearch
from urllib.parse import quote
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")


app = Flask(__name__)

@app.route('/')
def home():
    baseUrl = "http://127.0.0.1:5601/app/dashboards#/view/f11ea6db-9917-466b-9b7a-a7aadbe01216?embed=true&_g=(refreshInterval%3A(pause%3A!t%2Cvalue%3A60000)%2Ctime%3A(from%3Anow-15m%2Cto%3Anow))&show-query-input=true&show-time-filter=true"
    return render_template('index.html', baseUrl=baseUrl)


es = Elasticsearch("https://localhost:9200", verify_certs=False, api_key="XzBDX1dwQUJxRFZrNzV4aU94bHg6NjJTMnhfYTVSUm1CU3VZaUJWV3MtUQ==")
index_name = "research_papers_relevence"

def recommend(search_value, top_n = 5): 

    query_body = {
        "query": {
            "multi_match": {
                "query": search_value,
                "fields": ["title", "DOI"]
            }
         },
        "_source": ["title", "abstract"],
        "size": 1,
        }

    search_results = es.search(index=index_name, body=query_body)
    title = search_results['hits']['hits'][0]['_source']['title']
    abstract = search_results['hits']['hits'][0]['_source']['abstract']
    text = title + " " + abstract * 2 
    tokens = preprocess_text(text)
    text = ' '.join(tokens)
    result_dict = {"title": title, "text": text}
    recommands = find_similar_articles(top_n, result_dict, get_all_documents())
    return recommands

def get_all_documents(): 
    query_body = {
        "query": {
            "match_all": {}
         },
        "_source": ["title", "abstract"],
        "size": 125,
        }
#    print(query_body)
    search_results = es.search(index=index_name, body=query_body)
    result_dicts = []
    for hit in search_results['hits']['hits']:
        title = hit['_source']['title']
        abstract = hit['_source']['abstract']
        combined_text = title + "  " + abstract * 2
        preprocessed_tokens = preprocess_text(combined_text)
        result_dicts.append({"title": title, "text": ' '.join(preprocessed_tokens)}) 
    return result_dicts

def preprocess_text(input_text):
    # Tokenization
    tokens = word_tokenize(input_text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token.lower() for token in tokens if token not in stop_words]

    # Stemming
    ps = PorterStemmer()
    stemmed_tokens = [ps.stem(token) for token in filtered_tokens]
    stemmed_filtered_tokens = [token.lower() for token in stemmed_tokens if token not in stop_words]

    # Normalize: removing punctuation and numbers
    normalized_tokens = [ token.translate(str.maketrans('', '', string.punctuation)) for token in stemmed_filtered_tokens ]
    normalized_tokens = [re.sub(r'\d+', '', token) for token in normalized_tokens]

    # Filter tokens length under 3
    final_tokens = [token for token in normalized_tokens if len(token) > 2]

    return final_tokens

def find_similar_articles(top_n, user_search, all_docs):
    tfidf_vectorizer = TfidfVectorizer()
    all_texts = [doc["text"] for doc in all_docs]
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_texts)
    search_vector = tfidf_vectorizer.transform([user_search['text']])
    similarities = cosine_similarity(search_vector, tfidf_matrix).flatten()
    top_n_indices = similarities.argsort()[-top_n:][::-1]
    top_n_titles = [all_docs[i]["title"] for i in top_n_indices]
    return top_n_titles


@app.route('/rs', methods=['GET', 'POST'])
def recommand():
    nltk.download('punkt')
    nltk.download('stopwords')

    baseUrl = "http://127.0.0.1:5601/app/dashboards#/view/f11ea6db-9917-466b-9b7a-a7aadbe01216?embed=true&_g=(refreshInterval%3A(pause%3A!t%2Cvalue%3A60000)%2Ctime%3A(from%3Anow-15m%2Cto%3Anow))&show-query-input=true&show-time-filter=true"
    if request.method == 'POST':
        search_term = request.form.get('searchTerm')
        num_recommendations = int(request.form.get('numRecommendations'))
        recommendations = recommend(search_term, num_recommendations + 1)
        if(len(recommendations)==0):
            return
        kqlQuery = ""
        for i, title in enumerate(recommendations):
            if not title.strip(): 
                continue
            # Escape the ':' character in the title
            escaped_title = title.replace(':', ' ')
            kqlQuery += f'title:"{escaped_title}"'
            if i != len(recommendations) - 1:
                kqlQuery += ' OR '


        encoded_query = quote(kqlQuery)
        querySegment = f"&_a=(query:(language:lucene,query:'({encoded_query})'))"
        print(querySegment)
        full_kibana_url = f"{baseUrl}{querySegment}"
        
        return render_template('rs.html', baseUrl=full_kibana_url, recommendations=recommendations)
    else:
        return render_template('rs.html', baseUrl=baseUrl)


if __name__ == '__main__':
    app.run(debug=True)
