import streamlit as st
import json
from sentence_transformers import SentenceTransformer, util
import torch
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import time
from huggingface_hub import InferenceClient


HUGGINGFACE_API_TOKEN = "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"

@st.cache_data
def load_data():
    with open('train-v1.1.json', 'r', encoding='utf-8') as f:
        squad_data = json.load(f)

    articles = []
    count = 0
    max_articles = 50  # Установите желаемое количество статей

    for article_data in squad_data['data']:
        title = article_data.get('title', 'No Title')
        for paragraph in article_data['paragraphs']:
            if count >= max_articles:
                break
            context = paragraph['context']
            qas = paragraph['qas']
            articles.append({
                'title': title,
                'content': context,
                'qas': qas
            })
            count += 1
        if count >= max_articles:
            break
    return articles

@st.cache_resource
def load_models():
    # модель для преобразования текста в векторы
    retriever_model = SentenceTransformer('all-MiniLM-L6-v2')
    # модель для поиска документов (она преобразует тексты в числовые векторы)

    # KeyBERT для извлечения ключевых слов
    kw_model = KeyBERT()

    client = InferenceClient(token=HUGGINGFACE_API_TOKEN)

    return retriever_model, kw_model, client

@st.cache_data
def prepare_model_data(articles):
    # Получаем retriever_model из кэшированной функции
    # мы получаем три значения, но нужно только первое поэтому ставим _
    retriever_model, _, _ = load_models()

    # articles это список, где каждый элемент — это словарь(структура данных), представляющий статью с ключами 'title', 'content', и 'qas'.
    # Для каждой статьи в articles возьми её содержимое 'content' и добавь в список documents
    documents = [article['content'] for article in articles]

    # convert_to_tensor=True означает, что мы хотим получить результат в виде тензора
    document_embeddings = retriever_model.encode(documents, convert_to_tensor=True)

    # TF-IDF Векторизатор
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

    return documents, document_embeddings, tfidf_vectorizer, tfidf_matrix


# поиск похожих статей по запросу
def retrieve(query, retriever_model, document_embeddings, articles, top_k=5):
    query_embedding = retriever_model.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, document_embeddings)[0]

    # Значения k наибольших элементов из cos_scores.
    # Индексы этих элементов (то есть, какие именно документы соответствуют этим значениям)
    top_results = torch.topk(cos_scores, k=top_k)

    results = []
    for score, idx in zip(top_results[0], top_results[1]):
        idx = idx.item()  # Преобразуем тензор в число
        results.append({
            'title': articles[idx]['title'],
            'content': articles[idx]['content'],
            'score': score.item()
        })
    return results

# поиск статей по tf-idf
def retrieve_tfidf(query, tfidf_vectorizer, tfidf_matrix, articles, top_k=5):
    query_vec = tfidf_vectorizer.transform([query])

    # насколько каждый документ похож на запрос пользователя, используя метод TF-IDF
    # dot - матричное умножение между матрицей документов и вектором запроса
    # .flatten() превращает многомерный массив в одномерный список
    # т.е типа умножая матрицу документов на вектор запроса, мы получаем общее число совпадений для каждого документа
    scores = np.dot(tfidf_matrix, query_vec.T).toarray().flatten()

    # argsort - возвращает индексы элемента массива scores, отсортированных по возрастанию
    # [-top_k:] - последние k индексов из отсортированного списка
    # [::-1] разворачивает список индексов в обратном порядке, чтобы получить их по убыванию
    top_indices = scores.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            'title': articles[idx]['title'],
            'content': articles[idx]['content'],
            'score': scores[idx]
        })
    return results

# извлечения ключевых слов из запроса
def extract_keywords(query, kw_model):
    keywords = kw_model.extract_keywords(query, top_n=5)
    return [kw[0] for kw in keywords]

def generate_answer(query, retrieved_docs, client):
    context = ' '.join([doc['content'] for doc in retrieved_docs])

    max_context_length = 2000
    if len(context) > max_context_length:
        context = context[:max_context_length]


    prompt = f"Вопрос: {query}\nКонтекст: {context}\nОтвет:"

    try:
        response = client.text_generation(
            model="meta-llama/Llama-3.2-1B",
            prompt=prompt,
            max_new_tokens=500,
            # temperature=0.7 — параметр, который контролирует разнообразие генерируемого текста
            # Значения ближе к 0 делают текст более предсказуемым и однообразным, а ближе к 1 — более разнообразным и творческим
            temperature=0.7,
        )
        if isinstance(response, dict) and 'generated_text' in response:
            answer = response['generated_text']
        else:
            answer = response
    except Exception as e:
        st.error(f"Ошибка при обращении к модели: {e}")
        answer = "Извините, произошла ошибка при генерации ответа."

    return answer

def compute_precision_at_k(articles, retriever_model, document_embeddings, k=10):
    total_precision = 0
    total_questions = 0

    for article in articles:
        qas = article['qas']
        for qa in qas:
            question = qa['question']
            answers = [answer['text'] for answer in qa['answers']]

            retrieved_docs = retrieve(question, retriever_model, document_embeddings, articles, top_k=k)

            relevant_docs = 0
            for doc in retrieved_docs:
                for answer in answers:
                    if answer.lower() in doc['content'].lower():
                        relevant_docs += 1
                        break

            precision = relevant_docs / k
            total_precision += precision
            total_questions += 1

    precision_at_k = total_precision / total_questions
    return precision_at_k




# -------------------------------------------- MAIN ---------------------------------------------------



def main():
    st.title('Retrieval-Augmented Generation (RAG) с использованием Llama 2 через Hugging Face Inference API')

    with st.spinner('Загрузка данных...'):
        articles = load_data()

    with st.spinner('Загрузка моделей...'):
        retriever_model, kw_model, client = load_models()

    with st.spinner('Подготовка данных...'):
        documents, document_embeddings, tfidf_vectorizer, tfidf_matrix = prepare_model_data(articles)

    query = st.text_input('Введите ваш вопрос:')

    if query:
        start_time = time.time()

        with st.spinner('Извлечение документов...'):
            retrieval_method = st.selectbox('Выберите метод извлечения:', ['Семантический поиск', 'TF-IDF'])
            top_k = st.slider('Количество документов для извлечения:', min_value=1, max_value=20, value=5)

            if retrieval_method == 'Семантический поиск':
                retrieved_docs = retrieve(query, retriever_model, document_embeddings, articles, top_k=top_k)
            else:
                retrieved_docs = retrieve_tfidf(query, tfidf_vectorizer, tfidf_matrix, articles, top_k=top_k)

        with st.spinner('Генерация ответа...'):
            answer = generate_answer(query, retrieved_docs, client)

        end_time = time.time()
        response_time = end_time - start_time

        st.subheader('Ответ:')
        st.write(answer)

        st.subheader('Извлеченные документы:')
        for doc in retrieved_docs:
            st.markdown(f"**{doc['title']}**")
            st.write(doc['content'][:200] + '...')

        st.subheader('Время ответа:')
        st.write(f"{response_time:.2f} секунд")

        if 'response_times' not in st.session_state:
            st.session_state['response_times'] = []

        st.session_state['response_times'].append(response_time)

        if st.button('Вычислить квантили времени ответа'):
            response_times = st.session_state['response_times']
            q50 = np.quantile(response_times, 0.50)
            q90 = np.quantile(response_times, 0.90)
            q99 = np.quantile(response_times, 0.99)

            st.subheader('Квантили времени ответа:')
            st.write(f"Q50: {q50:.2f} секунд")
            st.write(f"Q90: {q90:.2f} секунд")
            st.write(f"Q99: {q99:.2f} секунд")

        # Кнопка для вычисления Precision@10
        if st.button('Вычислить Precision@10'):
            with st.spinner('Вычисление Precision@10...'):
                precision_at_10 = compute_precision_at_k(articles, retriever_model, document_embeddings, k=10)
                st.subheader('Precision@10:')
                st.write(f"{precision_at_10:.4f}")

if __name__ == '__main__':
    main()
