import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import matplotlib.pyplot as plt




# Define the FashionRecommender class
class FashionRecommender:
    def __init__(self, item_features, fashion_data):
        self.item_features = item_features
        self.fashion_data = fashion_data
        self.vectorizer = TfidfVectorizer()
        self.item_features = self.vectorizer.fit_transform(self.fashion_data['features'])

    def recommend_items(self, input_categories, input_description, n=10):
        # Combine input categories and description into a single query
        query = ' '.join(input_categories) + ' ' + input_description
        # Transform the query using the same vectorizer
        query_features = self.vectorizer.transform([query])
        # Calculate cosine similarity between the query and item features
        similarity_scores = cosine_similarity(query_features, self.item_features)
        # Find indices of the most similar items
        similar_items_indices = similarity_scores.argsort()[0][-n-1:-1][::-1]
        return self.fashion_data.iloc[similar_items_indices]

# Function to find and display the image based on the given image ID
def find_and_display_image(image_id, image_df):
    try:
        # Find the image path based on the provided ID
        image_path = image_df[image_df['id'] == image_id]['file_path'].values[0]
        # Open and display the image
        img = Image.open(image_path)
        st.image(img, caption=f"Budget: {image_id / 10}")
    except IndexError:
        st.write(f"Image path not found for ID {image_id}.")

# Load fashion item dataset and image CSV file
df = pd.read_csv('styles.csv')
image_df = pd.read_csv('image_data.csv')

# Drop the column
column_to_drop = 'Unnamed: 11'
df.drop(columns=[column_to_drop], inplace=True)  # Specify inplace=True to drop the column in-place

# Save the DataFrame back to a CSV file
df.to_csv('df.csv', index=False)
column_to_drop = 'year'
df.drop(columns=[column_to_drop], inplace=True)  # Specify inplace=True to drop the column in-place

# Save the DataFrame back to a CSV file
df.to_csv('df1.csv', index=False)

df['usage'].fillna(value='Regular', inplace=True)
df['gender'].fillna(value='Unisex', inplace=True)
df['baseColour'].fillna(value='Multi Colour', inplace=True)
column_to_drop = 'season'
df.drop(columns=[column_to_drop], inplace=True)  # Specify inplace=True to drop the column in-place

# Save the DataFrame back to a CSV file
df.to_csv('df1.csv', index=False)
df['productDisplayName'].fillna(value='Product', inplace=True)
df.merge(image_df, on='id')
# Feature Engineering
df['features'] = df['subCategory'] + ' ' + df['baseColour'] + ' ' + df['articleType'] + ' ' + df['masterCategory'] + ' ' + df['usage'] + ' ' + df['gender'] + ' ' + df['productDisplayName']

# Instantiate the recommender system
recommender = FashionRecommender(df['features'], df)

# Streamlit App
st.title("Fashion Recommender App")

# Input categories for recommendation
input_categories = st.text_input("Enter categories (comma-separated):")
input_description = st.text_input("Enter outfit description:")

if st.button("Recommend Outfits"):
    if input_categories:
        input_categories = [category.strip() for category in input_categories.split(',')]
        recommendations = recommender.recommend_items(input_categories, input_description)

        # Display recommended items and their images
        for index, row in recommendations.iterrows():
            image_id = row['id']  # Assuming 'id' is the column containing unique image IDs
            find_and_display_image(image_id, image_df)