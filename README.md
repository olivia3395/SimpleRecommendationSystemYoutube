

### README: YouTube Video Recommendation System

---

## Project Overview

This project aims to develop a **content-based recommendation system** for YouTube videos based on their **titles** and **descriptions**. Our approach focuses on utilizing video metadata (textual content) to recommend similar videos, even in the absence of user interaction data (such as likes, views, or ratings). The recommendation system is designed to suggest thematically related videos by analyzing the content using **natural language processing** techniques.

---

## Features

- **Content-Based Filtering**: The system recommends videos based on the **similarity of their content** (title and description).
- **TF-IDF Vectorization**: Text data is converted into numerical vectors using the **Term Frequency-Inverse Document Frequency (TF-IDF)** method.
- **Cosine Similarity**: Videos are compared based on **cosine similarity** between their TF-IDF vectors.
- **Visualizations**: Various visualizations, including **cosine similarity heatmaps** and **dimensionality reduction plots** (t-SNE, PCA), provide insights into the relationships between videos.

---

## Data Overview

The dataset used in this project contains the following fields for each YouTube video:
- **Video ID**: Unique identifier for each video.
- **Title**: The title of the video, providing a brief idea of its content.
- **Description**: A detailed summary of the video’s content.
- **Category**: The category to which the video belongs (e.g., Education).

---

## Exploratory Data Analysis (EDA)

We began by conducting a comprehensive **Exploratory Data Analysis (EDA)** to better understand the dataset and inform the recommendation process. Here are the key EDA techniques applied:

1. **Title Length Distribution**: We analyzed the length of video titles to observe trends in how creators structure titles to attract viewers.
   
2. **Description Length Distribution**: Similarly, we examined the length of video descriptions to understand how detailed they are and whether longer descriptions might correlate with certain types of content.

3. **Word Clouds for Titles and Descriptions**: Word clouds were generated to highlight the most common words across video titles and descriptions. This helped identify key themes and frequently discussed topics.

4. **Category Distribution**: We analyzed the distribution of video categories to better understand the variety of content available and how it might impact recommendations.

5. **Temporal Analysis**: Though not directly used in the recommendation system, we analyzed trends in video publication dates (such as months and days of the week) to explore potential relationships between content and time of publication.

6. **Sentiment Analysis**: We performed a basic sentiment analysis on video descriptions to see if certain types of content are generally more positive, negative, or neutral in tone.

7. **Topic Modeling (LDA)**: Latent Dirichlet Allocation (LDA) was used to uncover hidden topics within the video descriptions, helping to group videos based on common themes.

---

## Approach

### 1. Data Preprocessing
- **Text Cleaning**: We cleaned the video titles and descriptions by removing HTML entities, special characters, and non-alphanumeric symbols. This ensured that the text was well-formatted and ready for further analysis.
- **Tokenization**: We tokenized the cleaned text into words, lowercasing them for consistency.

### 2. TF-IDF Vectorization
To represent the textual data numerically, we applied **TF-IDF Vectorization** on the combined video titles and descriptions. This technique captures the importance of each word relative to the entire dataset, allowing for meaningful comparisons between videos.

### 3. Cosine Similarity
Once the text was vectorized, we calculated **cosine similarity** between all video pairs. Cosine similarity measures the angle between two vectors, providing a score between 0 and 1 that indicates how similar two videos are based on their text content.

### 4. Content-Based Recommendation System
Using the cosine similarity scores, we built a **content-based recommendation system**. For each video, the system retrieves the most similar videos by ranking the cosine similarity scores.

---

## Visualizations

To enhance our understanding of the data and the recommendations, we created several visualizations:

1. **Cosine Similarity Heatmap**: This heatmap shows the cosine similarity between all videos in the dataset, making it easy to see clusters of highly similar content.
   
2. **t-SNE/PCA for Dimensionality Reduction**: Using **t-SNE** and **PCA**, we reduced the high-dimensional TF-IDF vectors into 2D space. These visualizations help display how videos are grouped based on content similarity.

3. **Bar Plot for Similarity Scores**: For a given video, we plotted the **similarity scores** of its top 5 recommended videos, showing how closely each recommendation matches the original content.

---

## How It Works

### Running the Recommendation System:
1. **Preprocessing**: Clean the text data by removing unwanted characters.
2. **Vectorization**: Apply TF-IDF to transform the cleaned text into numerical vectors.
3. **Similarity Calculation**: Calculate cosine similarity between the TF-IDF vectors.
4. **Get Recommendations**: Given a video title, the system returns a list of the most similar videos based on cosine similarity.



## Conclusion

Our content-based recommendation system effectively suggests YouTube videos based on their textual content (titles and descriptions). By utilizing **TF-IDF vectorization** and **cosine similarity**, the system generates thematically relevant recommendations without the need for user interaction data. The project includes comprehensive EDA, insightful visualizations, and a straightforward recommendation pipeline, making it a scalable solution for video recommendation tasks.

Future improvements could involve incorporating **user behavior data** (e.g., views, likes) to build a **hybrid recommendation system** that combines content-based filtering with **collaborative filtering** for more personalized results.

