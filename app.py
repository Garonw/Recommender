import os

from flask import Flask, render_template, request, url_for
import pandas as pd
import matplotlib.pyplot as plt
import gc

app = Flask(__name__)


def load_data():
    book_data = pd.read_csv('data/Books.csv', low_memory=False)
    ratings_data = pd.read_csv('data/Ratings.csv')
    return book_data, ratings_data


# Cleans and removes unnecessary book data columns, handles missing values, and filters out invalid ratings.
def preprocess_data(book_data, ratings_data):
    book_data.drop(['Year-Of-Publication', 'Publisher'], axis=1, inplace=True)
    book_data['Book-Title'] = book_data['Book-Title'].str.replace(r'\(.*\)|:.*', '', regex=True)
    book_data['Book-Title'] = book_data['Book-Title'].str.replace(r'amp;', '', regex=True)
    book_data['ISBN'] = book_data['ISBN'].astype('str')
    book_data["ISBN"] = book_data["ISBN"].str.replace(r'\(.*?\)', '', regex=True)
    book_data.dropna()

    ratings_data["ISBN"] = ratings_data["ISBN"].str.replace(r'[^A-Za-z0-9]', '', regex=True)
    ratings_data["ISBN"] = ratings_data["ISBN"].str.replace(r'ISBN', '', regex=True)
    filtered_ratings = ratings_data[ratings_data['Book-Rating'] >= 1]
    filtered_ratings = filtered_ratings.dropna()
    return book_data, filtered_ratings


# Gets the number of ratings and average ratings for each book and merges with book data.
# Filters out books with low rating counts(for better representation of ratings data) and removes duplicates.
def process_books_and_ratings(books_data, ratings_data, min_ratings=50):
    ratings_count_per_book = ratings_data.groupby('ISBN')['Book-Rating'].count().reset_index()
    ratings_count_per_book.rename(columns={'Book-Rating': 'num_ratings'}, inplace=True)
    average_ratings_per_book = ratings_data.groupby('ISBN')['Book-Rating'].mean().reset_index()
    average_ratings_per_book.rename(columns={'Book-Rating': 'average_rating'}, inplace=True)

    rats_books = pd.merge(ratings_count_per_book, average_ratings_per_book, on='ISBN')
    ratings_and_books = pd.merge(rats_books, books_data, how='inner', on=['ISBN'])

    ratings_and_books_filtered = ratings_and_books[ratings_and_books['num_ratings'] >= min_ratings]
    ratings_and_books_filtered = ratings_and_books_filtered.drop_duplicates('Book-Title')

    gc.collect()

    return ratings_and_books_filtered


# A user-based collaborative filtering method to determine similar likes between the user and other users.
# Returns up to four book recommendations based on the amount of relative data for the input book.
def recommend(input_book_isbn, ratings_df, books_df, n_recommendations=4):
    # Find users who rated the input book over 5
    users_who_liked_input_book = ratings_df[(ratings_df['ISBN'] == input_book_isbn) & (ratings_df['Book-Rating'] > 5)][
        'User-ID'].unique()

    # Find other books those users rated over 5
    other_books_rated_by_users = ratings_df[(ratings_df['User-ID'].isin(users_who_liked_input_book)) &
                                            (ratings_df['Book-Rating'] > 5) &
                                            (ratings_df['ISBN'] != input_book_isbn)]

    # Group by ISBN and count how many users liked each book
    book_recommendations = other_books_rated_by_users.groupby('ISBN').size().sort_values(ascending=False).head(
        n_recommendations).index

    # Merge the recommendations with book details
    recommended_books = books_df[books_df['ISBN'].isin(book_recommendations)]

    gc.collect()

    return recommended_books[['Book-Title', 'Book-Author', 'Image-URL-L', 'ISBN']]


# Determines popular books by number of ratings and average rating for each book.
# Returns a dataframe of four random book from the popular books dataframe.
def get_popular_books(rats_per_book):
    pop_books = rats_per_book[(rats_per_book['num_ratings'] > 100) &
                              (rats_per_book['average_rating'] > 4)]
    rand_pop_books = []
    sample_books = pop_books.sample(4)
    for _, row in sample_books.iterrows():
        temp_df = pop_books[pop_books['Book-Title'] == row['Book-Title']]
        rand_pop_books.append([
            temp_df['Book-Title'].values[0],
            temp_df['Book-Author'].values[0],
            temp_df['Image-URL-L'].values[0],
            temp_df['ISBN'].values[0],
        ])
    gc.collect()

    return rand_pop_books


# Creates a bar chart of the top ten rated books
def plot_top_ten_books(books_with_ratings):
    books_with_ratings = books_with_ratings[books_with_ratings['num_ratings'] >= 20]
    top_ten_books = books_with_ratings.nlargest(10, 'average_rating')

    plt.figure(figsize=(12, 5))
    plt.subplots_adjust(left=0.4, bottom=0.25)
    plt.barh(top_ten_books['Book-Title'], top_ten_books['average_rating'], color='skyblue')
    plt.xlabel('Average Rating', fontsize=14)
    plt.yticks(fontsize=12)  # Enable wrapping of labels to show the full title
    plt.title('Top 10 Highest Rated Books', fontsize=16)
    plt.xlim(0, 10)
    plt.gca().invert_yaxis()  # To display the highest rating book on top
    plt.savefig('static/top_ten_books_chart.png')  # Save the figure in the specified path
    plt.show()

    gc.collect()


# Creates a bar chart of the top five rated authors
def plot_top_five_authors(books_with_ratings):
    books_with_ratings = books_with_ratings[books_with_ratings['num_ratings'] >= 100]
    top_five_authors = books_with_ratings.groupby('Book-Author').agg({'average_rating': 'mean'}).rename(
        columns={'average_rating': 'avg_rating'}).sort_values(by='avg_rating', ascending=False).head(5).reset_index()

    plt.figure(figsize=(12, 5))
    plt.subplots_adjust(left=0.4, bottom=0.25)
    plt.barh(top_five_authors['Book-Author'], top_five_authors['avg_rating'], color='skyblue')
    plt.xlabel('Combined Average Rating', fontsize=14)
    plt.yticks(fontsize=12)
    plt.title('Top 5 Highest Rated Authors', fontsize=16)
    plt.xlim(0, 10)
    plt.gca().invert_yaxis()
    plt.savefig('static/top_five_authors_chart.png')
    plt.show()

    gc.collect()


# Load and prep data
books, ratings = load_data()
books, ratings = preprocess_data(books, ratings)
filtered_books = process_books_and_ratings(books, ratings)
plot_top_ten_books(filtered_books)
plot_top_five_authors(filtered_books)


@app.route('/', methods=['GET', 'POST'])
def index():
    popular_books = get_popular_books(filtered_books)
    top_ten_chart = url_for('static', filename='top_ten_books_chart.png')
    top_five_chart = url_for('static', filename='top_five_authors_chart.png')

    if request.method == 'POST':
        book_isbn = request.form['book_isbn']
        book_title = request.form['book_title']
        recs = recommend(book_isbn, ratings, books)
        return render_template('index.html', recs=recs.to_dict(orient='records'),
                               popular_books=popular_books, selected_book=book_isbn, book_title=book_title)
    return render_template('index.html', popular_books=popular_books,
                           top_ten_chart=top_ten_chart, top_five_chart=top_five_chart)


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)