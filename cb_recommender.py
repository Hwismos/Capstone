# Load Models
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# read dataset
df_data = pd.read_csv('movies_metadata.csv', low_memory=False)

# 
df_data = df_data[df_data['vote_count'].notna()]
plt.figure(figsize=(20,5))
sns.distplot(df_data['vote_count'])
plt.title("Histogram of vote counts")

min_votes = np.percentile(df_data['vote_count'].values, 85)             # minimum vote 결정
df = df_data.copy(deep=True).loc[df_data['vote_count'] > min_votes]     # minimum number of votes의 movie 제외

# ============================================================================================================

# overview feature를 사용한 추천 → similar movie plot 추천
# removing rows with missing overview
df = df[df['overview'].notna()]
df.reset_index(inplace=True)

# processing of overviews
def process_text(text):
    # replace multiple spaces with one
    text = ' '.join(text.split())
    # lowercase
    text = text.lower()
    return text
df['overview'] = df.apply(lambda x: process_text(x.overview),axis=1)


# tf-idf representation 생성
tf_idf = TfidfVectorizer(stop_words='english')
tf_idf_matrix = tf_idf.fit_transform(df['overview']);

# 코사인 유사도 계산
cosine_similarity_matrix = cosine_similarity(tf_idf_matrix, tf_idf_matrix)

# title과 일치하는 인덱스를 찾는 메소드
def index_from_title(df,title):
    return df[df['original_title']==title].index.values[0]


# function that returns the title of the movie from its index
def title_from_index(df,index):
    return df[df.index==index].original_title.values[0]

# generating recommendations for given title → title을 인자로 전달했을 때 영화 추천
def recommendations(original_title, 
                    df, 
                    cosine_similarity_matrix, 
                    number_of_recommendations):
    index = index_from_title(df, original_title)
    similarity_scores = list(enumerate(cosine_similarity_matrix[index]))
    similarity_scores_sorted = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    recommendations_indices = [t[0] for t in similarity_scores_sorted[1:(number_of_recommendations+1)]]
    return df['original_title'].iloc[recommendations_indices]

# run
print(recommendations('Batman', df, cosine_similarity_matrix, 10))

'''
# * result
3693    Batman Beyond: Return of the Joker    
5962    The Dark Knight Rises                 
7379    Batman vs Dracula                     
5476    Batman: Under the Red Hood            
6654    Batman: Mystery of the Batwoman       
3911    Batman Begins                         
6334    Batman: The Dark Knight Returns, Part
1770     Batman & Robin                        
4725    The Dark Knight                       
709     Batman Returns   
'''