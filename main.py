# ============================================================
# MOVIE INFORMATION RETRIEVAL SYSTEM
# WITH MOVIE RANKING / RATINGS SUPPORT
# ============================================================

# =========================
# IMPORT LIBRARIES
# =========================

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import nltk
import re
import time
import ast

from difflib import get_close_matches

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# DOWNLOAD NLTK DATA
# =========================

nltk.download('stopwords')

# =========================
# LOAD DATASETS
# =========================

# Main movie dataset
df = pd.read_csv(
    "movies_metadata.csv",
    low_memory=False
)

# Links dataset
links_df = pd.read_csv(
    "links.csv"
)

# Ratings dataset
ratings_df = pd.read_csv(
    "ratings.csv"
)

# =========================
# HANDLE MISSING VALUES
# =========================

df['overview'] = df['overview'].fillna('')
df['title'] = df['title'].fillna('Unknown')
df['genres'] = df['genres'].fillna('[]')

# =========================
# FIX ID TYPES
# =========================

# Convert ids safely

df['id'] = pd.to_numeric(
    df['id'],
    errors='coerce'
)

links_df['tmdbId'] = pd.to_numeric(
    links_df['tmdbId'],
    errors='coerce'
)

links_df['movieId'] = pd.to_numeric(
    links_df['movieId'],
    errors='coerce'
)

ratings_df['movieId'] = pd.to_numeric(
    ratings_df['movieId'],
    errors='coerce'
)

# Remove invalid rows
df = df.dropna(subset=['id'])

# =========================
# EXTRACT GENRES
# =========================

def extract_genres(genres_text):

    try:

        genres_list = ast.literal_eval(genres_text)

        names = [
            genre['name']
            for genre in genres_list
            if 'name' in genre
        ]

        return " ".join(names)

    except:
        return ""

df['genres_text'] = df['genres'].apply(
    extract_genres
)

# =========================
# ADD MOVIE RATINGS / RANKS
# =========================

# Merge links with ratings
ratings_merged = pd.merge(
    links_df,
    ratings_df,
    on='movieId'
)

# Calculate rating statistics
movie_ratings = ratings_merged.groupby(
    'tmdbId'
).agg({

    'rating': ['mean', 'count']

}).reset_index()

# Rename columns
movie_ratings.columns = [
    'tmdbId',
    'avg_rating',
    'rating_count'
]

# Merge ratings into main dataframe
df = pd.merge(
    df,
    movie_ratings,
    left_on='id',
    right_on='tmdbId',
    how='left'
)

# Fill missing values
df['avg_rating'] = df['avg_rating'].fillna(0)
df['rating_count'] = df['rating_count'].fillna(0)

# =========================
# COMBINE SEARCH FEATURES
# =========================

df['combined_text'] = (
    (df['title'] + " ") * 5 +
    df['overview'] + " " +
    (df['genres_text'] + " ") * 2
)

documents = df['combined_text']

# =========================
# TEXT PREPROCESSING
# =========================

stemmer = PorterStemmer()

stop_words = set(
    stopwords.words('english')
)

def preprocess(text):

    text = str(text).lower()

    text = re.sub(
        r'[^a-zA-Z\s]',
        '',
        text
    )

    tokens = text.split()

    tokens = [

        stemmer.stem(word)

        for word in tokens

        if word not in stop_words

    ]

    return " ".join(tokens)

# =========================
# CLEAN DOCUMENTS
# =========================

cleaned_docs = documents.apply(
    preprocess
)

# =========================
# TF-IDF MODEL
# =========================

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2)
)

tfidf_matrix = vectorizer.fit_transform(
    cleaned_docs
)

# =========================
# MOVIE TITLES
# =========================

movie_titles = (
    df['title']
    .dropna()
    .unique()
    .tolist()
)

movie_titles_lower = [
    title.lower()
    for title in movie_titles
]

# =========================
# SPELL CORRECTION
# =========================

def correct_spelling(query):

    matches = get_close_matches(
        query.lower(),
        movie_titles_lower,
        n=1,
        cutoff=0.6
    )

    if matches:

        idx = movie_titles_lower.index(
            matches[0]
        )

        return movie_titles[idx]

    return query

# =========================
# SEARCH FUNCTION
# =========================

def search_movies(event=None):

    start_time = time.time()

    query = search_entry.get().strip()

    if query == "":

        messagebox.showwarning(
            "Warning",
            "Please enter movie keyword"
        )

        return

    # =========================
    # SPELL CORRECTION
    # =========================

    corrected_query = correct_spelling(
        query
    )

    # =========================
    # PROCESS QUERY
    # =========================

    processed_query = preprocess(
        corrected_query
    )

    query_vector = vectorizer.transform(
        [processed_query]
    )

    # =========================
    # COSINE SIMILARITY
    # =========================

    similarity = cosine_similarity(
        query_vector,
        tfidf_matrix
    )

    scores = similarity.flatten()

    # =========================
    # TITLE BOOSTING
    # =========================

    query_lower = query.lower()

    exact_matches = df[
        df['title']
        .str.lower()
        == query_lower
    ]

    partial_matches = df[
        df['title']
        .str.lower()
        .str.contains(query_lower, na=False)
    ]

    # Exact match boost
    for idx in exact_matches.index:
        scores[idx] += 1.0

    # Partial match boost
    for idx in partial_matches.index:
        scores[idx] += 0.3

    # =========================
    # ADD RATING BOOST
    # =========================
    # IMPORTANT:
    # THIS DOES NOT REPLACE SEARCH SCORE
    # It only slightly improves ranking
    # for highly rated movies

    rating_boost = (
        (df['avg_rating'] / 10) * 0.15
    )

    popularity_boost = (
        (df['rating_count'] / (
            df['rating_count'].max() + 1
        )) * 0.10
    )

    final_scores = (
        scores +
        rating_boost +
        popularity_boost
    )

    # =========================
    # TOP RESULTS
    # =========================

    top_indices = (
        final_scores.argsort()[-10:][::-1]
    )

    # =========================
    # CLEAR RESULTS
    # =========================

    result_box.config(state=tk.NORMAL)

    result_box.delete(
        1.0,
        tk.END
    )

    # =========================
    # SEARCH INFO
    # =========================

    end_time = time.time()

    duration = end_time - start_time

    result_box.insert(
        tk.END,
        "============================================================\n",
        "header"
    )

    result_box.insert(
        tk.END,
        f"Total Movies : {len(df)}\n",
        "info"
    )

    result_box.insert(
        tk.END,
        f"Search Time : {duration:.4f} seconds\n",
        "info"
    )

    result_box.insert(
        tk.END,
        "Top 10 Ranked Results\n",
        "info"
    )

    result_box.insert(
        tk.END,
        "============================================================\n\n",
        "header"
    )

    # =========================
    # SPELL CORRECTION MESSAGE
    # =========================

    if corrected_query.lower() != query.lower():

        result_box.insert(
            tk.END,
            f"Showing results for: {corrected_query}\n\n",
            "correction"
        )

    # =========================
    # DISPLAY RESULTS
    # =========================

    rank = 1

    for i in top_indices:

        title = df['title'].iloc[i]

        overview = df['overview'].iloc[i]

        genres = df['genres_text'].iloc[i]

        search_score = scores[i]

        avg_rating = df['avg_rating'].iloc[i]

        rating_count = int(
            df['rating_count'].iloc[i]
        )

        percentage = min(
            search_score * 100,
            100
        )

        result_box.insert(
            tk.END,
            f"#{rank}  {title}\n",
            "title"
        )

        result_box.insert(
            tk.END,
            f"{percentage:.2f}% Match\n",
            "score"
        )

        # MOVIE RATING
        result_box.insert(
            tk.END,
            f"Movie Rating: {avg_rating:.1f}/5.0\n",
            "score"
        )

        result_box.insert(
            tk.END,
            f"Total Ratings: {rating_count}\n",
            "info"
        )

        result_box.insert(
            tk.END,
            f"Genres: {genres}\n",
            "genres"
        )

        result_box.insert(
            tk.END,
            f"\nOverview:\n{overview[:500]}\n",
            "overview"
        )

        result_box.insert(
            tk.END,
            "\n------------------------------------------------------------\n\n",
            "separator"
        )

        rank += 1

    result_box.config(state=tk.DISABLED)

    suggestion_box.pack_forget()

# =========================
# CLEAR FUNCTION
# =========================

def clear_results():

    search_entry.delete(0, tk.END)

    result_box.config(state=tk.NORMAL)

    result_box.delete(
        1.0,
        tk.END
    )

    result_box.config(state=tk.DISABLED)

    suggestion_box.pack_forget()

# =========================
# AUTOCOMPLETE
# =========================

def update_suggestions(event=None):

    typed = (
        search_entry.get()
        .strip()
        .lower()
    )

    suggestion_box.delete(0, tk.END)

    if typed == "":

        suggestion_box.pack_forget()

        return

    matches = []

    for idx, lower_title in enumerate(movie_titles_lower):

        if lower_title.startswith(typed):

            matches.append(
                movie_titles[idx]
            )

        if len(matches) >= 8:
            break

    if not matches:

        for idx, lower_title in enumerate(movie_titles_lower):

            if typed in lower_title:

                matches.append(
                    movie_titles[idx]
                )

            if len(matches) >= 8:
                break

    if matches:

        suggestion_box.delete(
            0,
            tk.END
        )

        for item in matches:

            suggestion_box.insert(
                tk.END,
                item
            )

        suggestion_box.pack(
            pady=(0, 10)
        )

    else:

        suggestion_box.pack_forget()

# =========================
# SELECT SUGGESTION
# =========================

def select_suggestion(event=None):

    try:

        selected = suggestion_box.get(
            suggestion_box.curselection()
        )

        search_entry.delete(
            0,
            tk.END
        )

        search_entry.insert(
            0,
            selected
        )

        suggestion_box.pack_forget()

        search_entry.focus_set()

    except:
        pass

# =========================
# KEYBOARD NAVIGATION
# =========================

def move_down(event):

    if suggestion_box.size() > 0:

        suggestion_box.focus_set()

        suggestion_box.selection_clear(
            0,
            tk.END
        )

        suggestion_box.selection_set(0)

        suggestion_box.activate(0)

def move_up_entry(event):

    search_entry.focus_set()

# =========================
# ROOT WINDOW
# =========================

root = tk.Tk()

root.title(
    "Movie Information Retrieval System"
)

root.geometry("1200x800")

root.config(bg="#121212")

# =========================
# COLORS
# =========================

BG_COLOR = "#121212"
CARD_COLOR = "#1E1E1E"
ENTRY_COLOR = "#252525"
ACCENT_COLOR = "#00BFA6"

# =========================
# SCROLLBAR STYLE
# =========================

style = ttk.Style()

style.theme_use("clam")

style.configure(
    "Vertical.TScrollbar",
    background=ACCENT_COLOR,
    troughcolor=BG_COLOR,
    bordercolor=BG_COLOR,
    arrowcolor="white"
)

# =========================
# TITLE
# =========================

title_label = tk.Label(
    root,
    text="MOVIE INFORMATION RETRIEVAL SYSTEM",
    font=("Segoe UI", 28, "bold"),
    bg=BG_COLOR,
    fg=ACCENT_COLOR
)

title_label.pack(
    pady=25
)

# =========================
# SEARCH FRAME
# =========================

search_frame = tk.Frame(
    root,
    bg=BG_COLOR
)

search_frame.pack(
    pady=5
)

# =========================
# SEARCH ENTRY
# =========================

search_entry = tk.Entry(
    search_frame,
    width=40,
    font=("Segoe UI", 14),
    bg=ENTRY_COLOR,
    fg="white",
    insertbackground="white",
    relief=tk.FLAT,
    bd=0
)

search_entry.grid(
    row=0,
    column=0,
    padx=10,
    ipady=12,
    ipadx=10
)

search_entry.bind(
    "<KeyRelease>",
    update_suggestions
)

search_entry.bind(
    "<Down>",
    move_down
)

search_entry.bind(
    "<Return>",
    search_movies
)

# =========================
# SEARCH BUTTON
# =========================

search_button = tk.Button(
    search_frame,
    text="Search",
    font=("Segoe UI", 12, "bold"),
    bg=ACCENT_COLOR,
    fg="white",
    relief=tk.FLAT,
    padx=25,
    pady=10,
    cursor="hand2",
    command=search_movies
)

search_button.grid(
    row=0,
    column=1,
    padx=5
)

# =========================
# CLEAR BUTTON
# =========================

clear_button = tk.Button(
    search_frame,
    text="Clear",
    font=("Segoe UI", 12, "bold"),
    bg="#E53935",
    fg="white",
    relief=tk.FLAT,
    padx=25,
    pady=10,
    cursor="hand2",
    command=clear_results
)

clear_button.grid(
    row=0,
    column=2,
    padx=5
)

# =========================
# AUTOCOMPLETE FRAME
# =========================

suggestion_frame = tk.Frame(
    root,
    bg=BG_COLOR
)

suggestion_frame.pack()

# =========================
# AUTOCOMPLETE LISTBOX
# =========================

suggestion_box = tk.Listbox(
    suggestion_frame,
    font=("Segoe UI", 11),
    bg=CARD_COLOR,
    fg="white",
    selectbackground=ACCENT_COLOR,
    relief=tk.FLAT,
    height=6,
    width=48,
    bd=0,
    highlightthickness=1,
    highlightbackground="#333333"
)

suggestion_box.pack_forget()

# =========================
# BINDINGS
# =========================

suggestion_box.bind(
    "<<ListboxSelect>>",
    select_suggestion
)

suggestion_box.bind(
    "<Return>",
    select_suggestion
)

suggestion_box.bind(
    "<Double-Button-1>",
    select_suggestion
)

suggestion_box.bind(
    "<Up>",
    move_up_entry
)

# =========================
# RESULT FRAME
# =========================

result_frame = tk.Frame(
    root,
    bg=BG_COLOR
)

result_frame.pack(
    fill=tk.BOTH,
    expand=True,
    padx=20,
    pady=15
)

# =========================
# RESULT BOX
# =========================

result_box = tk.Text(
    result_frame,
    wrap=tk.WORD,
    font=("Segoe UI", 12),
    bg=CARD_COLOR,
    fg="white",
    relief=tk.FLAT,
    padx=20,
    pady=20
)

result_box.pack(
    side=tk.LEFT,
    fill=tk.BOTH,
    expand=True
)

# =========================
# TEXT TAG STYLES
# =========================

result_box.tag_config(
    "title",
    font=("Segoe UI", 16, "bold"),
    foreground="#00E5FF"
)

result_box.tag_config(
    "score",
    font=("Segoe UI", 12, "bold"),
    foreground="#00FF99"
)

result_box.tag_config(
    "genres",
    font=("Segoe UI", 11, "italic"),
    foreground="#FFC107"
)

result_box.tag_config(
    "overview",
    font=("Segoe UI", 11),
    foreground="#DDDDDD"
)

result_box.tag_config(
    "header",
    foreground="#00BFA6",
    font=("Segoe UI", 11, "bold")
)

result_box.tag_config(
    "info",
    foreground="#BBBBBB",
    font=("Segoe UI", 10)
)

result_box.tag_config(
    "correction",
    foreground="#FF9800",
    font=("Segoe UI", 11, "bold")
)

# =========================
# SCROLLBAR
# =========================

scrollbar = ttk.Scrollbar(
    result_frame,
    orient=tk.VERTICAL,
    command=result_box.yview
)

scrollbar.pack(
    side=tk.RIGHT,
    fill=tk.Y
)

result_box.config(
    yscrollcommand=scrollbar.set
)

# =========================
# START APP
# =========================

root.mainloop()