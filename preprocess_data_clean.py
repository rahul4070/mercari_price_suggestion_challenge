# import modules

# %matplotlib inline
import warnings
warnings.filterwarnings("ignore")

import datetime
import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import seaborn as sns
import math
import nltk
from collections import Counter
import os
import shutil
from tqdm.notebook import tqdm
tqdm.pandas()
import re
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm
tqdm.pandas()
import pickle
import nltk
nltk.download('stopwords')

nltk.download('vader_lexicon')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler
import csv
from scipy import sparse

class DataCleaning_Preprocessing:

    def __init__(self):
        self.all_unique_brands = []


    def preprocess(self, data):

        print('\n')

        print('*'*20 + 'preprocessing the data...' + '*' * 20)
        print()

        start_time = datetime.datetime.now()

        def category_split(string):
            '''
                function to split the category column into three subcategories

                input: category string
                returns: three subcategory strings 
            '''
            try:
                # split the string with '/'
                t = string.split('/')
                return t[0], t[1], t[2]
            except:
                return 'unk_cat', 'unk_cat', 'unk_cat'


        print('filling nan category_name values with "unk_cat/unk_subcat1/unk_subcat2"...')
        data.category_name.fillna('unk_cat/unk_subcat1/unk_subcat2',inplace=True)
        print('filling nan category_name complete!')
        print('-' * 100)
        print()

        print('creating subcategory columns...')
        data['main_category'], data['sub_category1'], data['sub_category2'] = zip(*data.category_name.apply(category_split))
        print('subcategory columns creation completed!')
        print('-' * 100)
        print()

        print('filling nan of item_description...')
        data['item_description'].fillna('unk_desc',inplace=True)
        print('item_description fillna complete !')
        print('-' * 100)
        print()


        print('filling nan brand values with "unk_brand" and converting to lower case...')
        data['brand_name'] = data['brand_name'].str.lower()
        data.brand_name.fillna('unk_brand',inplace=True)
        print('filled nan brand_name and converted to lower case!')
        print('-' * 100)
        print()

        print('time taken to execute the cell : ', datetime.datetime.now()- start_time)
        print()

        print('*'*20 + 'Preprocessing of the data done!' + '*' * 20)

        return data

    def calculate_unique_brands(self, data):
        self.all_unique_brands = data.brand_name.unique()


    def feature_engineering(self, data):
        '''
            function to perform the feature engineering 
            input - dataframe
            output - dataframe
        '''


        def decontracted(phrase):

            # reference - https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python

            '''  
                this function helps in expanding the given phrases.

                input: phrase/ word
                returns: expanded string 
            '''

            # specific
            phrase = re.sub(r"won\'t", "will not", phrase)
            phrase = re.sub(r"can\'t", "can not", phrase)

            # general
            phrase = re.sub(r"n\'t", " not", phrase)
            phrase = re.sub(r"\'re", " are", phrase)
            phrase = re.sub(r"\'s", " is", phrase)
            phrase = re.sub(r"\'d", " would", phrase)
            phrase = re.sub(r"\'ll", " will", phrase)
            phrase = re.sub(r"\'t", " not", phrase)
            phrase = re.sub(r"\'ve", " have", phrase)
            phrase = re.sub(r"\'m", " am", phrase)
            return phrase


        stopwords_ = stopwords.words('english')

        def preprocess_text(text, col):
            """
                Function to clean the strings containing special characters and converts them to lowercase characters.

                input: string
                output: string which contains number and lower character.
            """

            try:
                # convert the string to lowercase
                text = text.lower()
                # decontraction - expanding the words like : i'll -> i will, he'd -> he would
                text = decontracted(text)
                # replace & and - character with _ . 
                text = re.sub('[&-]', '_', text)    #  Example : t-shirt -> t_shirt, horse&sweater -> horse_sweater
                # replace special characters except _
                text = re.sub('[^0-9a-z_]',' ',text)
                text = re.sub('\s_\s', ' ', text)   #  replace strings like  ' _ ' with ' ' (string with a space)
                text = re.sub('\s+', ' ', text).strip()  # replace more than one_space_character to single_space_character
                if col != 'name':
                    # removing the stopwords
                    text = ' '.join(i for i in text.split(' ') if not i in stopwords_)
                else:
                    text = ' '.join(i for i in text.split(' '))
            except:
                text = np.nan
            return text


        def generate_ngrams(s, n):
            # reference - https://albertauyeung.github.io/2018/06/03/generating-ngrams.html


            # Break sentence in the token, remove empty tokens
            tokens = [token for token in s.split(" ") if token != ""]
            
            # Use the zip function to help us generate n-grams
            # Concatentate the tokens into ngrams and return
            ngrams = zip(*[tokens[i:] for i in range(n)])
            # print(list(ngrams))
            return [" ".join(ngram) for ngram in ngrams]


        def fill_missing_brands(df, all_unique_brands):
            name, brand_name, item_description = df[0], df[1], df[2]
            name = str(name) + ' ' + str(item_description)
            ngram_ = [4,3,2,1]
            if brand_name != 'unk_brand':
                return brand_name
            else:
                try:
                    brand_names = []
                    for i in ngram_:
                        for grams in generate_ngrams(name, i):
                            brand = ' '.join(grams)
                            if brand in all_unique_brands:
                                brand_names.append(brand)
                    if len(brand_names) > 0:
                        return brand_names[0]
                    else:
                        return 'unk_brand'
                except :
                    return 'unk_brand'








        print('\n')
        print('*'*20 + 'Feature engineering...' + '*' * 20)
        print()
        start_time = datetime.datetime.now()



        print('preprocessing name...')
        data['name'] = data['name'].progress_apply(lambda x: preprocess_text(x, 'name'))
        print('preprocessing of name complete!')
        print('-' * 80)
        print()

        print('preprocessing item_description...')
        data['item_description'] = data['item_description'].progress_apply(lambda x: preprocess_text(x, 'item_description'))
        print('preprocessing of item_description complete!')
        print('-' * 100)
        print()



        print('computing word count of name feature...')
        data['len_name'] = data['name'].progress_apply(lambda x: len(str(x).split(' ')))
        print('name_feature word count computation done!')
        print('-' * 100)
        print()

        print('computing word count of item description...')
        data['len_item_description'] = data['item_description'].progress_apply(lambda x: len(str(x).split(' ')))
        print('item description word count computation done!')
        print('-' * 100)
        print()

        print('combining name with item descripiton with word count 10 ...')
        data['name_desc'] = data['name'] + ' ' + data['item_description'].progress_apply(lambda x: ' '.join(str(x).split(' ')[:10]))
        print('combining feature name and item_description done!')
        print('-' * 100)
        print()

        print('combining name, brand_name, subcategories together...')
        data['name_brand_cat'] = 'name_' + data['name'] + ' ' + 'brand' + data['brand_name'] + ' ' + 'main category' + data['main_category'] + \
                                ' ' + 'sub category' + data['sub_category1'] + ' ' + 'sub category' + data['sub_category2']
        print('combining feature name, brand_name, subcategories together done!')
        print('-' * 100)
        print()

        print('assigning the branded_products with value 1 and unknown_branded products with 0...')
        data['brand_value'] = data['brand_name'].progress_apply(lambda x: 1 if x != 'unk_brand' else 0)
        print('assigning the branded products with 1 and non branded products with 0 done!')
        print('-' * 100)
        print()


        no_brand_name_before = data[data.brand_name == 'unk_brand'].shape[0]

        print('filling missing brand_name with help of "name" and "item_descripiton" feature...')
    #    all_unique_brands = data.brand_name.unique()
        data['brand_name'] = data[['name','brand_name','item_description']].progress_apply(lambda x: fill_missing_brands(x, self.all_unique_brands), axis=1)
        print('fill missing brand_name complete!')
        print()

        print('number of unk_brand filled- {}'.format(no_brand_name_before - data[data.brand_name == 'unk_brand'].shape[0]))
        print()

        print('-' * 100)

        sid = SentimentIntensityAnalyzer()

        def sentiment_analysis(sentence, sentiment):
            ss = sid.polarity_scores(sentence)
            senti_ = ss[sentiment]

            return senti_

        print('doing sentiment_analysis for positive sentences...')
        data['pos'] = data.item_description.astype(str).progress_apply(lambda x: sentiment_analysis(x, 'pos'))
        print('sentiment analysis of positive sentences done!')
        print('-' * 100)
        print()

        print('doing sentiment_analysis for negative sentences...')
        data['neg'] = data.item_description.astype(str).progress_apply(lambda x: sentiment_analysis(x, 'neg'))
        print('sentiment analysis of negative sentences done!')
        print('-' * 100)
        print()

        print('doing sentiment_analysis for neutral sentences...')
        data['neu'] = data.item_description.astype(str).progress_apply(lambda x: sentiment_analysis(x, 'neu'))
        print('sentiment analysis of neutral sentences done!')
        print('-' * 100)
        print()



        print('time taken to execute the cell : ', datetime.datetime.now()- start_time)

        print('*'*30 + 'Feature engineering done!' + '*' * 30)

        return data


    def vectorization(self, data):

        '''
            function to convert the data into its respective vector form
            input - dataframe
            output - csr_matrix with horizontal stacking of features
        '''
        print('\n')
        start_time = datetime.datetime.now()

        print('*'*20 + 'Converting the features into vectors...' + '*' * 20)
        print()

        print(data.columns)

        import os.path
        if not os.path.isfile('glove.6B.100d.txt'):
            print('please download glove vector first!')
            return  

        print('loading the glove vector...')
        words_dict = dict()
        f = open('glove.6B.100d.txt')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            words_dict[word] = coefs
        f.close()
        print('glove vector loaded!')


        def sentence_word2vec(sentence):
            '''
                function to convert text_feature to respective vector form using glove_vector
            '''
            vector = np.zeros(100)
            for word in sentence.split():
                if word in words_dict:
                    vector += words_dict[word]
            
            return vector


        print('converting the concatenation of name_description column to its respective vector form...')
        name_desc_vector = data.name_desc.astype(str).progress_apply(lambda x: sentence_word2vec(x))
        print('\nconversion of name_desc to vector completed!')
        print('-' * 100)
        print()

        print('converting the concatenation of name, brand, sub_categories column to its respective vector form...')
        name_brand_cat_vector = data.name_brand_cat.astype(str).progress_apply(lambda x: sentence_word2vec(x))
        print('\nconversion of name_brand_cat to vector completed!')
        print('-' * 100)
        print()

        data.drop(columns=['name_desc', 'name_brand_cat'], inplace=True)

        print('converting text_feature to vector form...')
        name_desc_vector = sparse.csc_matrix(name_desc_vector.values.tolist())
        name_brand_cat_vector = sparse.csc_matrix(name_brand_cat_vector.values.tolist())
        print('converted text features to vector form')
        print('-'*100)
        print()

        cheap_brand = pickle.load(open('cheap_brand_set', "rb"))
        affordable_brand = pickle.load(open('affordable_brand_set', "rb"))
        expensive_brand = pickle.load(open('expensive_brand_set', "rb"))

        def fill_brand_category(brand_name):
            try:
                if brand_name in cheap_brand:
                    return 'cheap'
                elif brand_name in affordable_brand:
                    return 'affordable'
                elif brand_name in expensive_brand:
                    return 'expensive'
                else:
                    return 'affordable'
            except:
                return 'affordable'


        data['categorise_brand'] = data['brand_name'].apply(lambda x: fill_brand_category(x))


        def ordinal_encoder_load(feature_name, file_name):
            encoder = pickle.load(open('encoder/' + file_name + '_ordinal_encoder.pkl', 'rb'))
            encoder_values = encoder.transform(data[feature_name].astype(str).values.reshape(-1,1))

            imputer = pickle.load(open('imputer/' + file_name + '_imputer.pkl', 'rb'))
            imputer_values = imputer.transform(encoder_values)
            return imputer_values

        print('-'*100)
        print('applying label encoder and scaling the features...')


        train_brand_name = ordinal_encoder_load('brand_name', 'brand_name')
        train_category_brand = ordinal_encoder_load('categorise_brand', 'categorise_brand')
        train_category = ordinal_encoder_load('category_name', 'category_name')
        train_main_category = ordinal_encoder_load('main_category', 'main_category')
        train_sub_category1 = ordinal_encoder_load('brand_name', 'sub_category1')
        train_sub_category2 = ordinal_encoder_load('brand_name', 'sub_category2')

        # scaling the label encoded features
        brand_scaler = pickle.load(open('scaler/brand_scaler.pkl', 'rb'))
        train_brand_name =  brand_scaler.transform(train_brand_name)

        category_brand_scaler = pickle.load(open('scaler/category_brand_scaler.pkl', "rb"))
        train_category_brand  = category_brand_scaler.transform(train_category_brand)

        category_scaler = pickle.load(open('scaler/category_scaler.pkl', 'rb'))
        train_category =  category_scaler.transform(train_category)

        main_category_scaler = pickle.load(open('scaler/main_category_scaler.pkl', 'rb'))
        train_main_category =  main_category_scaler.transform(train_main_category)

        main_sub_category1_scaler = pickle.load(open('scaler/main_sub_category1_scaler.pkl', 'rb'))
        train_sub_category1 =  main_sub_category1_scaler.transform(train_sub_category1)

        main_sub_category2_scaler = pickle.load(open('scaler/main_sub_category2_scaler.pkl', 'rb'))
        train_sub_category2 =  main_sub_category2_scaler.transform(train_sub_category2)
        
        train_brand_name = sparse.csr_matrix(train_brand_name)
        train_category_brand = sparse.csr_matrix(train_category_brand)
        train_category = sparse.csr_matrix(train_category)
        train_main_category = sparse.csr_matrix(train_main_category)
        train_sub_category1 = sparse.csr_matrix(train_sub_category1)
        train_sub_category2 = sparse.csr_matrix(train_sub_category2)

        print('label encoding and scaling of features done!')
        print('-'*100)

        data.drop(columns=['brand_name','category_name', 'main_category', 'sub_category1', 'sub_category2'], inplace=True)

        data['item_condition_id'] = data['item_condition_id'] / 5.0

        len_name_scaler = pickle.load(open('scaler/len_name_scaler.pkl', 'rb'))
        data['len_name'] =  len_name_scaler.transform(data['len_name'].values.reshape(-1,1))

        len_item_description_scaler = pickle.load(open('scaler/len_item_description_scaler.pkl', 'rb'))
        data['len_item_description'] =  len_item_description_scaler.transform(data['len_item_description'].values.reshape(-1,1))

        print('feature scaling done!')



        data_hstack = sparse.hstack((data['item_condition_id'].values.reshape(-1,1),data['shipping'].values.reshape(-1,1),\
                                    train_brand_name, train_category_brand, train_category, train_main_category,\
                                    train_sub_category1, train_sub_category2, data['len_name'].values.reshape(-1,1),\
                                    data['len_item_description'].values.reshape(-1,1), data['brand_value'].values.reshape(-1,1),\
                                    data['pos'].values.reshape(-1,1),\
                                    data['neg'].values.reshape(-1,1), data['neu'].values.reshape(-1,1), name_desc_vector, \
                                    name_brand_cat_vector)).tocsr()

        print('time taken to execute the cell : ', datetime.datetime.now()- start_time)
        print()


        print('*'*30 + 'feature vectorization is done!' + '*' * 30)

        return data_hstack

