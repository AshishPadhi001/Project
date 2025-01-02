# -*- coding: utf-8 -*-
"""AdvancedMRS.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/178lNyMY6U8eu9PLDYd0YASDzEPxMTbqN
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies=pd.read_csv('bollywood_full.csv')

movies.head(5)

#Shape of the dataset
movies.shape

#Information about the dataset
movies.info()

#Columns to be used
#imdb_id
#original_title
#is_adult
#year_of_release
#runtime
#genres
#imdb_rating
#imdb_voting
#actors

df=movies[['imdb_id','original_title','actors','is_adult','year_of_release','runtime','genres','imdb_rating','imdb_votes']]

print(df.shape)

df.head(5)

#Column 1: imdb_id
print("Data of imdb_id",df['imdb_id'][0])
print("Data Type of imdb_id",df['imdb_id'].dtype)

#Checking for null and duplicated values as any row having duplicate id can affect the data
print("Null Values in imdb_id",df['imdb_id'].isnull().sum())
print("Duplicated values in imdb_id:",df['imdb_id'].duplicated().sum())

#printing the duplicated dats rows
duplicate_rows = df[df['imdb_id'].duplicated(keep=False)]
print("Data With Dupliacte rows",duplicate_rows)

#Removing the duplicated one and keeping the first one

# Using .loc to modify the column properly and avoid the warning
df= df.drop_duplicates(subset='imdb_id', keep='first')


print("Duplicated values after imputation:",df['imdb_id'].duplicated().sum())

#Column 2: original_title
print("Data of original_title",df['original_title'][0])
print("Data Type of original_title",df['original_title'].dtype)

#Checking for null and duplicated values as any row having duplicate id can affect the data
print("Null Values in original_title",df['original_title'].isnull().sum())
print("Duplicated values in original_title:",df['original_title'].duplicated().sum())

#printing the duplicated dats rows
duplicate_rows = df[df['original_title'].duplicated(keep=False)]
print("Data With Dupliacte rows",duplicate_rows)

#Keep the first one
#Removing the duplicated one and keeping the first one

# Using .loc to modify the column properly and avoid the warning
df= df.drop_duplicates(subset='imdb_id', keep='first')


print("Duplicated values after imputation:",df['imdb_id'].duplicated().sum())

#Column 3: Actors
print("Data of actors",df['actors'][0])
print("Data Type of actors",df['actors'].dtype)

#Checking for null and duplicated values as any row having duplicate id can affect the data
print("Null Values in actors",df['actors'].isnull().sum())
print("Duplicated values in actors:",df['actors'].duplicated().sum())

#Replace the null values into 'Unknown'
df['actors']=df['actors'].fillna('Unknown')

#Converting the dtaya type into string
df['actors']=df['actors'].astype(str)

#Replacing the '|' with ","
df['actors']=df['actors'].str.replace('|',',')

df['actors'][0]

#Column 4: is_adult
print("Data of is_adult",df['is_adult'][0])
print("Data Type of is_adult",df['is_adult'].dtype)

#Checking for null and duplicated values as any row having duplicate id can affect the data
print("Null Values in is_adult",df['is_adult'].isnull().sum())
print("Duplicated values in is_adult:",df['is_adult'].duplicated().sum())

 #Replace the '0' with Not adult
df['is_adult']=df['is_adult'].replace(0,'Not Adult')

#Replace the '1' with Adult
df['is_adult']=df['is_adult'].replace(1,'Adult')

df['is_adult'][0]

#Column 5: year_of_release
print("Data of year_of_release",df['year_of_release'][0])
print("Data Type of year_of_release",df['year_of_release'].dtype)

#Checking for null and duplicated values as any row having duplicate id can affect the data
print("Null Values in year_of_release",df['year_of_release'].isnull().sum())
print("Duplicated values in year_of_release:",df['year_of_release'].duplicated().sum())

#Column 6: runtime
print("Data of runtime",df['runtime'][0])
print("Data Type of runtime",df['runtime'].dtype)

#Checking for null and duplicated values as any row having duplicate id can affect the data
print("Null Values in runtime",df['runtime'].isnull().sum())
print("Duplicated values in runtime:",df['runtime'].duplicated().sum())
# Replace '\\N' with NaN
df['runtime'] = df['runtime'].replace('\\N', pd.NA)

# Convert the column to numeric (invalid entries will become NaN)
df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce')

# Calculate the mean of the valid runtime values
mean_runtime = df['runtime'].mean()

# Fill NaN values with the mean runtime
df['runtime'] = df['runtime'].fillna(mean_runtime)

# Convert the column to integers if needed
df['runtime'] = df['runtime'].astype(int)

# Check the updated DataFrame
print(df['runtime'])

#Categorize the column runtime
def categorize_runtime(runtime):
    if runtime < 90:
        return "Short Movie"
    elif 90 <= runtime <= 150:
        return "Regular Movie"
    else:
        return "Long Movie"

# Apply the categorization
df['runtime'] = df['runtime'].apply(categorize_runtime)

df['runtime'][0]

#Plotting runtime vs rating showing how a movie runtime affects its rating
sns.barplot(data=df,x='runtime',y='imdb_rating')
plt.title('Runtime vs Rating')
plt.xlabel('Runtime')
plt.ylabel('Rating')
plt.show()

#Plotting runtime vs votes showing how a user voting get affected by runtime
sns.barplot(data=df,x='runtime',y='imdb_votes')
plt.title('Runtime vs Votes')
plt.xlabel('Runtime')
plt.ylabel('Votes')
plt.show()

#Plotting runtime vs year_of_release it shoes how with time runtime gets affected
sns.boxplot(data=df,x='runtime',y='year_of_release')
plt.title('Runtime vs year of release')
plt.xlabel('Runtime')
plt.ylabel('year of release')
plt.show()

#Column 7: genres
print("Data of genres",df['genres'][0])
print("Data Type of genres",df['genres'].dtype)

#Checking for null and duplicated values as any row having duplicate id can affect the data
print("Null Values in genres",df['genres'].isnull().sum())
print("Duplicated values in genres:",df['genres'].duplicated().sum())

#Converting it into string adata type
df['genres']=df['genres'].astype(str)

#Replacing the '|' with ","
df['genres']=df['genres'].str.replace('|',',')

df['genres'][0]

df['genres'].value_counts()

#Column 8: imdb_rating
print("Data of imdb_rating",df['imdb_rating'][0])
print("Data Type of imdb_rating",df['imdb_rating'].dtype)

#Checking for null and duplicated values as any row having duplicate id can affect the data
print("Null Values in imdb_rating",df['imdb_rating'].isnull().sum())
print("Duplicated values in imdb_rating:",df['imdb_rating'].duplicated().sum())

#Categorize the rating column
def categorize_runtime(runtime):
    if runtime < 5:
        return "Poor Movie"
    elif 5 <= runtime <= 7.5:
        return "Average Movie"
    elif 7.5 <= runtime <= 9:
        return "Hit Movie"
    else:
        return "Blockbuster Movie"

df['imdb_rating'] = df['imdb_rating'].apply(categorize_runtime)

df['imdb_rating'][0]

df['imdb_rating'].value_counts()

#plotting rating vs year of release showing how the movies are rating goes with time
sns.boxplot(data=df,y='imdb_rating',x='year_of_release')
plt.title('Rating vs year of release')
plt.xlabel('Rating')
plt.ylabel('year of release')
plt.show()

#Plotting rating vs votes showing how it affects the rating of movie
sns.barplot(data=df,x='imdb_rating',y='imdb_votes')
plt.title('Rating vs Votes')
plt.xlabel('Rating')
plt.ylabel('Votes')
plt.show()

#Column 9: imdb_votes
print("Data of imdb_votes",df['imdb_votes'][0])
print("Data Type of imdb_votes",df['imdb_votes'].dtype)

#Checking for null and duplicated values as any row having duplicate id can affect the data
print("Null Values in imdb_votes",df['imdb_votes'].isnull().sum())
print("Duplicated values in imdb_votes:",df['imdb_votes'].duplicated().sum())

# Maximum value in df['imdb_votes']
max_votes = df['imdb_votes'].max()
print("Maximum votes:", max_votes)

# Minimum value in df['imdb_votes']
min_votes = df['imdb_votes'].min()
print("Minimum votes:", min_votes)

# Function to categorize the votes with adjusted intervals
def categorize_votes(votes):
    if votes <= 1000:
        return "Low Votes"
    elif votes <= 100000:
        return "Moderate Votes"
    elif votes <= 200000:
        return "High Votes"
    else:
        return "Blockbuster Votes"

# Apply the categorization function to the 'imdb_votes' column
df['imdb_votes'] = df['imdb_votes'].apply(categorize_votes)

df['imdb_votes'][0]

df['imdb_votes'].value_counts()

#Plotting votes vs year of release showing bhow people rated movies of latest ones
sns.boxplot(data=df,y='imdb_votes',x='year_of_release')
plt.title('Votes vs year of release')
plt.xlabel('Votes')
plt.ylabel('year of release')
plt.show()

# Combining both votes and rating so that reliability increase

# Define the function to assign refined review categories
def assign_rank(rating, votes):
    if rating in ["Hit Movie", "Blockbuster Movie"] and votes in ["Moderate Votes", "High Votes", "Blockbuster Votes"]:
        return 1  # Most reliable
    elif rating in ["Hit Movie", "Blockbuster Movie"] and votes == "Low Votes":
        return 2  # Reliable but fewer votes
    elif rating == "Average Movie" and votes in ["Moderate Votes", "High Votes"]:
        return 3  # Moderately reliable
    elif rating == "Average Movie" and votes == "Low Votes":
        return 4  # Less reliable
    elif rating == "Poor Movie":
        return 5  # Least reliable
# Apply the function to the DataFrame
df['review'] = df.apply(lambda row: assign_rank(row['imdb_rating'], row['imdb_votes']), axis=1)

# Display the results
print(df[['imdb_rating', 'imdb_votes', 'review']])

df['review'].value_counts()

#Removing the not required columns
df=df.drop(['imdb_rating','imdb_votes'],axis=1)

#Plotting review vs year of relase
sns.boxplot(data=df,x='review',y='year_of_release')
plt.title('Review vs year of release')
plt.xlabel('Review')
plt.ylabel('year of release')
plt.show()

# Add 'sr' column as the first column
df.insert(0, 'Sr.no', range(1, len(df) + 1))

df.to_csv("movies_refined.csv", index=False)

new_df=df.copy()

new_df.sample(5)

new_df.info()

new_df.columns

new_df['tags'] = (
    new_df['original_title'] + "  " +
    new_df['actors'] + " " +
    new_df['year_of_release'].astype(str) + " " +
    new_df['runtime'] + " " +
    new_df['genres'] + " " +
    new_df['is_adult'] + " " +
    new_df['review'].astype(str)
)

new_df['tags'][0]

#Dropping the columns
new_df=new_df.drop(['original_title','actors','year_of_release','runtime','genres','is_adult','review'],axis=1)

#printing the new df
print("Shape of new data :",new_df.shape)
new_df.head(5)

# Removing punctuation from the column 'Merged_Columns'
import string

# Remove punctuation using the string.punctuation constant
new_df['tags'] = new_df['tags'].apply(lambda x: ''.join([char for char in x if char not in string.punctuation]))


new_df['tags'][0]

#Tokenize the columns
import nltk
from nltk import word_tokenize
nltk.download('punkt_tab')

new_df['tokenized_tags']=new_df['tags'].apply(word_tokenize)

new_df['tokenized_tags'][0]

#Stop word removal
from nltk.corpus import stopwords

nltk.download('stopwords')
stopwords=stopwords.words('english')

new_df['filtered_tags']=new_df['tokenized_tags'].apply(lambda x: [word for word in x if word not in stopwords])

new_df['filtered_tags'][0]

#Lemmatization of word
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
lemmatizer=WordNetLemmatizer()

new_df['lemmatized_tags']=new_df['filtered_tags'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

new_df['lemmatized_tags'][0]

#Dropping the columns
new_df=new_df.drop(['tokenized_tags','filtered_tags'],axis=1)

#Converting into string
new_df['lemmatized_tags_str']=new_df['lemmatized_tags'].apply(lambda x:' '.join(x)).astype(str)
print(new_df['lemmatized_tags_str'].dtype)
#Converting into lower case
new_df['lemmatized_tags_str']=new_df['lemmatized_tags_str'].str.lower()

new_df['lemmatized_tags_str'][0]

#Vectorize the column
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(input='content', encoding='utf-8', decode_error='replace', use_idf=True, smooth_idf=True)
#Input content will specify that we have string type data
#Will use encoding on the file input
#decode error replace will replace problematic statements with a unicode character
#use idf will apply idf also otherwise it wil perform same as bow
#Smooth idf will prevent the terms to be divided by zero

#Applying vectorization on the data
tfidf_matrix= vectorizer.fit_transform(new_df['lemmatized_tags_str'])

print("Sparse Matrix",tfidf_matrix)

#Converting the sparse array into dense
tfidf_dense= tfidf_matrix.toarray()

print("Dense Array:",tfidf_dense)

# Convert the dense array to a DataFrame (with proper column names)
vect_df = pd.DataFrame(tfidf_dense, columns=vectorizer.get_feature_names_out())

# Print the new DataFrame with the TF-IDF features

print(vect_df.shape)
vect_df.head()

# now combining the dataframes
# Combine merged_df and tfidf_df
# id is a column in merged_df that uniquely identifies each row
merged_df_combined = new_df.copy()  # Make a copy to avoid modifying the original
merged_df_combined['Vectorized_Tokens'] = list(vect_df.values)  # Add the vectorized tokens as a new column

merged_df_combined.sample(10)

merged_df_combined.to_csv("Vectorized_df.csv",index=False)

# Define valid options for genres, ratings, and review values
valid_genres = ['action', 'comedy', 'drama', 'thriller', 'sci-fi']
valid_ratings = ['poor', 'average', 'hit', 'blockbuster']
valid_reviews = ['1', '2', '3', '4', '5']

#Creating a function for selecting filters
def get_filters():
    print("Available genres: Action, Comedy, Drama, Thriller, Sci-Fi")  # Genres we got
    print("Available ratings: Poor, Average, Hit, Blockbuster")#Ratings we have
    print("Available review values: 1, 2, 3, 4, 5")#Review values we have

    selected_genres = input("Select the genres you are interested in (comma-separated, or 'All' for no filter): ").split(',')#Take input of genres
    selected_ratings = input("Select the ratings (comma-separated, or 'All' for no filter): ").split(',')#Take input of ratings
    selected_reviews = input("Select the review values (comma-separated, or 'All' for no filter): ").split(',')#Take input of reviews

    # Clean and validate the inputs
    selected_genres = [genre.strip().lower() for genre in selected_genres] #Cleaning the input genres by removing the whitesapce and convert into lower case
    selected_ratings = [rating.strip().lower() for rating in selected_ratings]#Cleaning the input rating by removing the whitesapce and convert into lower case
    selected_reviews = [review.strip() for review in selected_reviews]#Cleaning the input reviews by removing the whitesapce and convert into lower case

    # Validate input
    #It will get terminate if the condition becomes true till it is false it will get excuted
    while not all(genre == 'all' or genre in valid_genres for genre in selected_genres):
        print("Incorrect option selected please select from the options")
        #Asking user for enter inout correctly
        selected_genres = input("Enter the genres you are interested in (comma-separated, or 'All' for no filter): ").split(',')
        #Cleaning the input genres by removing the whitesapce and convert into lower case
        selected_genres = [genre.strip().lower() for genre in selected_genres]

    # Validate input
    #It will get terminate if the condition becomes true till it is false it will get excuted
    while not all(rating == 'all' or rating in valid_ratings for rating in selected_ratings):
        print("Incorrect option selected please select from the options")
        #Asking user for enter inout correctly
        selected_ratings = input("Select the ratings (comma-separated, or 'All' for no filter): ").split(',')
        #Cleaning the input ratings by removing the whitesapce and convert into lower case
        selected_ratings = [rating.strip().lower() for rating in selected_ratings]

    # Validate input
    #It will get terminate if the condition becomes true till it is false it will get excuted
    while not all(review == 'all' or review in valid_reviews for review in selected_reviews):
        print("Incorrect option selected please select from the options")
        #Asking user for enter inout correctly
        selected_reviews = input("Select the review values (comma-separated, or 'All' for no filter): ").split(',')
        #Cleaning the input reviews by removing the whitesapce and convert into lower case
        selected_reviews = [review.strip() for review in selected_reviews]

    #Display the Selection by user
    print(f"You selected genres: {', '.join(selected_genres)}")
    print(f"You selected ratings: {', '.join(selected_ratings)}")
    print(f"You selected review values: {', '.join(selected_reviews)}")

    return selected_genres, selected_ratings, selected_reviews

def apply_function(df, selected_genres, selected_ratings, selected_reviews):
    # Copy the DataFrame to avoid modifying the original
    filtered_df = df.copy()

    # Apply genre filters
    if selected_genres and selected_genres != ['all']:
        filtered_df = filtered_df[filtered_df['lemmatized_tags_str'].str.contains('|'.join(selected_genres), case=False)]

    # Apply rating filters
    if selected_ratings and selected_ratings != ['all']:
        filtered_df = filtered_df[filtered_df['lemmatized_tags_str'].str.contains('|'.join(selected_ratings), case=False)]

    # Apply review filters
    if selected_reviews and selected_reviews != ['all']:
        filtered_df = filtered_df[filtered_df['lemmatized_tags_str'].str.contains('|'.join(selected_reviews), case=False)]

    return filtered_df

# Step 1: Ask if the user wants to apply a filter
use_filter = input("Do you want to use filters (genres, ratings, reviews)? (yes/no): ").strip().lower()

if use_filter == 'yes':
    # Apply filters based on user input
    selected_genres, selected_ratings, selected_reviews = get_filters() #Applying the filter Function on here

    # Filter the DataFrame
    filtered_df = apply_function(df=merged_df_combined, selected_genres=selected_genres, selected_ratings=selected_ratings, selected_reviews=selected_reviews)
else:
    # If no filter, use the full DataFrame
    print("You selected no filter so original dataset will be used ")
    filtered_df = merged_df_combined

#It will take input from user about the query
user_choice = input("Do you want to enter a search query (actor, movie name, etc.)? (yes/no): ").strip().lower()

#Handles the query
if user_choice == 'yes':
    #Then will ask for the choices of user
    user_query = input("Enter your query (actor name, movie name, etc.): ").strip()
    # The variable n will display number of recommendations the user wants
    n = int(input("Enter the number of recommendations you want: "))

elif user_choice == 'no' and use_filter == 'yes':
    #It states that if there is no user chice but user filterr then filter will be used as query
    user_query = ' '.join(selected_genres + selected_ratings + selected_reviews)
    print(f"No query entered. Using selected filters as the query: {user_query}")
    # The variable n will display number of recommendations the user wants
    n = int(input("Enter the number of recommendations you want: "))

else:
    # If both query and filters are not provided
    user_query = None
    print("No query or filters provided. Restarting search process.")

# Step 4: Proceed if there is a valid query
if user_query:
  # Vectorize the user query
  user_query_vector = vectorizer.transform([user_query])
  user_query_vector_dense = user_query_vector.toarray()

  # Compute cosine similarity with the filtered dataset
  #this will convert the 1d vectors into 2d array
  stored_vectors = np.vstack(merged_df_combined['Vectorized_Tokens'])
  similarities = cosine_similarity(user_query_vector_dense, stored_vectors)

  # Check for matches
  if np.all(similarities == 0):
    print(f"No match found for your query: {user_query}")
  else:
    # Get top matches based on similarity scores
    top_indexes = np.argsort(similarities[0])[::-1][:n]
    top_matches = movies.iloc[top_indexes]

    # Display top movie titles
    print(f"\nTop matches for your query: {user_query}")
    for id, row in enumerate(top_matches.itertuples(), start=1):
      print(f"{id}: {row._asdict()['title_x']}")  # Use _asdict() to access column names correctly

    # Select the movie number whose details you want
    movie_choice = int(input(f"Enter the number of the movie you want details for (1-{n}): ")) - 1

    # Ensure the choice is valid
    if 0 <= movie_choice < n:
      selected_movie = top_matches.iloc[movie_choice] #Storing the index of selected movie
      movie_id = selected_movie['imdb_id']  # getting the imdb_id of selected mpvie

      # Now we will use this id and check in whole Data frame of movies where the id will match we will extract the output
      movie_details = movies[movies['imdb_id'] == movie_id].iloc[0]

      # Displaying detailed information
      print("\nMovie Details:")
      print(f"Original Title: {movie_details['original_title']}")
      print(f"Actors: {movie_details['actors']}")
      print(f"Year of Release: {movie_details['year_of_release']}")
      print(f"Genres: {movie_details['genres']}")
      print(f"Runtime: {movie_details['runtime']}")
      print(f"Summary: {movie_details['summary']}")
    else:
      print("Invalid selection. Please restart the process.")
