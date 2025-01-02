import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the CSV files
vectorized_df = pd.read_csv("MRS/vectorized_df.csv")
bollywood_full = pd.read_csv("MRS/bollywood_full.csv")
movies_refined = pd.read_csv("MRS/movies_refined.csv")

def main():
    st.markdown("## 🎥 Welcome to **Movie Lens** 🎬")  
    st.markdown("Discover movies tailored to your taste. Apply filters, enter a query, and let us find the perfect flick for you!")

    # Predefined lists for filters
    valid_genres = ['action', 'comedy', 'drama', 'thriller', 'sci-fi']
    valid_ratings = ['poor', 'average', 'hit', 'blockbuster']
    valid_reviews = ['1', '2', '3', '4', '5']

    # Ask user if they want to use a filter
    use_filter = st.radio("Do you want to use a filter?", ("Yes", "No"))

    filters = {}
    if use_filter == "Yes":
        # Allow user to select filters
        genre = st.multiselect("Select Genres:", ["None", "All"] + valid_genres)
        rating = st.multiselect("Select Ratings:", ["None", "All"] + valid_ratings)
        review = st.multiselect("Select Reviews:", ["None", "All"] + valid_reviews)

        # Store selected filters
        if "All" in genre:
            filters['genre'] = valid_genres
        elif "None" not in genre:
            filters['genre'] = genre

        if "All" in rating:
            filters['rating'] = valid_ratings
        elif "None" not in rating:
            filters['rating'] = rating

        if "All" in review:
            filters['review'] = valid_reviews
        elif "None" not in review:
            filters['review'] = review

        if filters:
            st.write("Filters applied:")
            if 'genre' in filters:
                st.write(f"Genres: {', '.join(filters['genre'])}")
            if 'rating' in filters:
                st.write(f"Ratings: {', '.join(filters['rating'])}")
            if 'review' in filters:
                st.write(f"Reviews: {', '.join(filters['review'])}")
        else:
            st.write("No valid filters selected.")

    if use_filter == "No" or not filters:
        st.write("No filters have been selected.")

    # Ask user if they want to provide a query
    provide_query = st.radio("Do you want to provide a query?", ("Yes", "No"))

    query = ""
    if provide_query == "Yes":
        query = st.text_input("Enter your query:")
        if query:
            st.write(f"Generating recommendations based on query: '{query}' and selected filters.")
        else:
            st.write("No query provided. Please enter a query to proceed.")

    if provide_query == "No" and filters:
        query = " ".join(filters.get('genre', []) + filters.get('rating', []) + filters.get('review', []))
        st.write(f"Using filters as query: {query}")

    # If neither query nor filters are provided
    if not query.strip() and not filters:
        st.write("Please enter a filter or query to proceed.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # Apply filters if available
    filtered_df = vectorized_df.copy()

    if filters:
        if 'genre' in filters:
            filtered_df = filtered_df[filtered_df['lemmatized_tags_str'].apply(lambda x: any(genre in x for genre in filters['genre']))]
        if 'rating' in filters:
            if 'imdb_rating' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['imdb_rating'].apply(lambda x: any(rating in str(x) for rating in filters['rating']))]
        if 'review' in filters:
            if 'imdb_votes' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['imdb_votes'].apply(lambda x: any(review in str(x) for review in filters['review']))]

    # If the user enters a query, vectorize and find recommendations
    if query.strip():
        vectorizer = TfidfVectorizer(stop_words='english')
        query_vec = vectorizer.fit_transform([query])
        movie_vecs = vectorizer.transform(filtered_df['lemmatized_tags_str'])
        cosine_sim = cosine_similarity(query_vec, movie_vecs)

        if cosine_sim.max() == 0:
            st.write("No movies found matching the query and filters.")
            st.markdown('</div>', unsafe_allow_html=True)
            return

        num_recommendations = st.slider("Select number of recommendations", 1, 10, 5)
        recommended_indices = cosine_sim[0].argsort()[-num_recommendations:][::-1]

        st.subheader("Top Movie Recommendations:")
        for idx in recommended_indices:
            movie = filtered_df.iloc[idx]
            movie_details = bollywood_full[bollywood_full['imdb_id'] == movie['imdb_id']].iloc[0]
            movie_title = movie_details['title_x']

            if st.button(f"Show details for {movie_title}"):
                st.write(f"**{movie_details['title_x']}**")
                st.write(f"**Actors**: {movie_details['actors']}")
                st.write(f"**Genres**: {movie_details['genres']}")
                st.write(f"**IMDb Rating**: {movie_details['imdb_rating']}")
                st.write(f"**Summary**: {movie_details['summary']}")
                st.write(f"**Release Date**: {movie_details['release_date']}")
                st.write(f"[More Info on IMDB](http://www.imdb.com/title/{movie_details['imdb_id']})")

    elif not query.strip() and filters:
        st.write("Here are some movies based on your selected filters:")
        for idx, row in filtered_df.iterrows():
            movie_details = bollywood_full[bollywood_full['imdb_id'] == row['imdb_id']].iloc[0]
            movie_title = movie_details['title_x']

            if st.button(f"Show details for {movie_title}"):
                st.write(f"**{movie_details['title_x']}**")
                st.write(f"**Actors**: {movie_details['actors']}")
                st.write(f"**Genres**: {movie_details['genres']}")
                st.write(f"**IMDb Rating**: {movie_details['imdb_rating']}")
                st.write(f"**Summary**: {movie_details['summary']}")
                st.write(f"**Release Date**: {movie_details['release_date']}")
                st.write(f"[More Info on IMDB](http://www.imdb.com/title/{movie_details['imdb_id']})")
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
