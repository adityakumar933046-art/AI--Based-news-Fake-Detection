# Fake News Detection System



A Django-based application combining Machine Learning and Google's Fact Check Tools API.



\## Features

\- Fake and real news classification

\- TF-IDF text vectorization

\- Logistic Regression model

\- Google Fact Check API integration

\- Existing fact-check review links

\- Prediction history

\- Django Admin dashboard



\## Technologies

\- Python

\- Django

\- Pandas and NumPy

\- Scikit-learn

\- NLTK

\- TF-IDF

\- Logistic Regression

\- Google Fact Check Tools API

\- HTML and CSS



\## Installation



\### 1. Clone the repository

```bash

git clone https://github.com/Alok018-vk/AI--Based-news-Fake-Detection.git

cd AI--Based-news-Fake-Detection

```



\### 2. Install dependencies

```bash

pip install -r requirements.txt

```



\### 3. Configure API key



Create a `.env` file in the project root:



```env

GOOGLE\_FACT\_CHECK\_API\_KEY=your\_api\_key\_here

```



\### 4. Run migrations and start server

```bash

python manage.py migrate

python manage.py runserver

```



\## Dataset

The WELFake dataset is used for model training. Dataset CSV files are not included in this repository.



\## Disclaimer

The ML model predicts based on learned text patterns. Google Fact Check searches previously published fact-check reviews. Neither result independently guarantees that an entire news article is true or false.
