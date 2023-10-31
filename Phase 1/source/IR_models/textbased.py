from fileinput import filename
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import string
import nltk
from nltk.corpus import stopwords

from sklearn.metrics.pairwise import cosine_similarity
import time
import pickle

IMAGE_DIR = './source/data/img'
CAPTION_DIR = './source/data/txt'

vectorizer = TfidfVectorizer()

nltk.download('punkt')
nltk.download('stopwords')

def remove_punctuation(text):
    return text.translate(str.maketrans('','', string.punctuation))

def remove_whitespace(text):
    return " ".join(text.split())

def remove_stopwords(doc):
    words_to_remove = stopwords.words('english')
    cleaned_doc = []
    for word in doc:
        if word not in words_to_remove:
            cleaned_doc.append(word)
    return cleaned_doc

def get_tokenized_list(doc):
    tokens = nltk.word_tokenize(doc)
    return tokens

def word_stemmer(token_list):
    ps = nltk.stem.PorterStemmer()
    stemmed = []
    for words in token_list:
        stemmed.append(ps.stem(words))
    return stemmed

def prepocessing_text(caption_path):
    id_corpus = []
    corpus = []

    for file_name in sorted(os.listdir(caption_path)):
        lines = open(os.path.join(caption_path, file_name), 'r', encoding='cp1252')
        temp_str = ''
        try:
          for line in lines:
              if line[-1] == '\n':
                  temp_str += line[:-1] + ' '
              else:
                  temp_str += line
          temp_str = temp_str.lower()
          temp_str = remove_whitespace(temp_str)
          temp_str = remove_punctuation(temp_str)
          corpus.append(temp_str)
          id_corpus.append(str(file_name[:-4]))
          lines.close()
        except:
          pass
    cleaned_corpus = []
    for d in corpus:
        tokens = get_tokenized_list(d)
        doc = remove_stopwords(tokens)
        doc = word_stemmer(doc)
        doc = ' '.join(doc)
        cleaned_corpus.append(doc)
    
    return (id_corpus, cleaned_corpus)

def preprocessing_query(query):
    query = query.lower()
    query = remove_punctuation(query)
    query = remove_whitespace(query)
    query = get_tokenized_list(query)
    query = remove_stopwords(query)
    q = []
    for w in word_stemmer(query):
        q.append(w)
    q = ' '.join(q)
    vector_query = vectorizer.transform([q])
    return vector_query

def show_img_retrieved(related_docs_indices, id_corpus):
    fig = plt.figure(figsize=(15,6))
    for idx, id in enumerate(related_docs_indices):
        img_name = str(id_corpus[id]) + '.png'
        img = mpimg.imread(os.path.join(IMAGE_DIR, img_name))
        fig.add_subplot(len(related_docs_indices)//5 + 1, 5, idx+1)
        plt.title('#{}'.format(idx+1))
        plt.axis('off')
        plt.imshow(img)
    plt.show()

if __name__ == "__main__":
    id_corpus, corpus = prepocessing_text(CAPTION_DIR)

    filename = 'source/corpus'
    outfile = open(filename, 'wb')
    pickle.dump(prepocessing_text(CAPTION_DIR), outfile)
    outfile.close()

    vector_doc = vectorizer.fit_transform(corpus)

    query = input('Your query: ')
    number_of_img = int(input('Number of images retrieved: '))
    start = time.time()
    vector_query = preprocessing_query(query)
    similar = cosine_similarity(vector_doc, vector_query).flatten()
    related_docs_indices = similar.argsort()[:-(number_of_img+1):-1]
    stop = time.time()
    running_time = stop - start
    print('{} images retrieved in {}s'.format(number_of_img, running_time))
    show_img_retrieved(related_docs_indices, id_corpus)
