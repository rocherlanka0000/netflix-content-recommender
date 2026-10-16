# Netflix Content Recommendation System

This repository contains the code and results for a content-based recommendation system built on Netflix catalogue metadata. The goal is to suggest titles similar to a given Netflix movie or TV show, using genres, country, director, rating and type. The project was completed as Task 1 of the Machine Learning Internship at Auspify Technologies.

## 1. Project Overview

### Objective

- Recommend similar Netflix titles from content attributes only, without user history.
- Convert text metadata into numerical vectors with TF-IDF.
- Measure similarity between titles with cosine similarity.
- Evaluate the quality of the recommendations against a random baseline.

### Why content-based?

The dataset contains no user ratings or viewing history, so collaborative filtering is not possible. A content-based approach only needs the description of each title, which makes it suitable here.

## 2. Data & Preprocessing

- Dataset: Netflix Movies and TV Shows metadata (`Dataset.csv`), 8,790 rows and 10 columns: `show_id`, `type`, `title`, `director`, `country`, `date_added`, `release_year`, `rating`, `duration`, `listed_in`.
- The dataset has no synopsis column, so recommendations rely on metadata only.
- Basic steps:
  - Remove duplicated titles (3 found), leaving 8,787 titles.
  - Treat the director value `Not Given` (2,588 rows, about 29%) as missing, so that unknown directors do not look similar to each other.
  - Fill missing countries with an empty value.
  - Turn each genre, country, director, rating and type into one token (for example `international_movies`, `rating_tv_ma`).
  - Combine all tokens into one text field per title. Genres are repeated twice to give them more weight.

## 3. Method

1. **Feature preparation:** build the combined text field for each title.
2. **TF-IDF:** convert the text into a matrix of 8,787 titles by 5,131 tokens. TF-IDF gives more weight to rare tokens, such as a niche genre or a specific director, than to very common ones.
3. **Cosine similarity:** since TF-IDF vectors are normalized, the dot product equals the cosine similarity. It is computed on demand for one title instead of building the full 8,787 x 8,787 matrix, which would take more than 600 MB.
4. **Recommendation:** the function `recommend(title, n)` returns the `n` titles with the highest similarity score, excluding the title itself.

## 4. Example Recommendations

| Starting title | Top recommendations |
|---|---|
| Stranger Things | Chilling Adventures of Sabrina, Helix, Nightflyers |
| The Irishman | Gangs of New York, The Departed, Hugo |
| Narcos | Marvel's Jessica Jones, Queen of the South, Narcos: Mexico |

Several titles reach a similarity of 1.000. This happens when two titles have identical metadata (same genres, country, rating and type), and it shows a limit of using metadata only.

## 5. Evaluation

There is no user history, so a classical precision measure cannot be computed. Two consistency measures are used, compared with a random recommendation baseline. They are averaged over 500 randomly chosen titles, with 10 recommendations each.

| | Genre overlap (Jaccard) | Same type (movie / TV show) |
|---|---|---|
| Model | 0.905 | 1.000 |
| Random | 0.095 | 0.588 |

The system recommends titles of the same type with very close genres, and does much better than a random choice. Because genres are part of the model inputs, the genre-overlap score favours the model. It shows consistency, not real user satisfaction.

## 6. Repository Structure

```
.
├── README.md
├── requirements.txt
├── Dataset.csv
├── task1_recommendation_en.ipynb
└── task1_recommendation_en.pdf       # Executed notebook exported as PDF
```

## 7. Installation & Local Run

### 7.1. Clone the repository

```
git clone https://github.com/<your-username>/netflix-content-recommender.git
cd netflix-content-recommender
```

### 7.2. Create and activate a virtual environment (recommended)

```
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 7.3. Install dependencies

```
pip install -r requirements.txt
```

### 7.4. Run the notebook

Make sure `Dataset.csv` is in the same folder as the notebook, then run:

```
jupyter notebook task1_recommendation_en.ipynb
```

Use "Restart & Run All" to reproduce all results.

## 8. Limitations & Next Steps

- Without a synopsis, similarity relies on metadata. Two titles with the same genre but different themes will be matched.
- The evaluation measures consistency, not real user satisfaction, which would require viewing data.
- Potential extensions:
  - Add a synopsis and apply TF-IDF on real text.
  - Weight genres, country and director differently.
  - Accept several liked titles and recommend from their average profile.
  - Combine with collaborative filtering if user data becomes available.

## 9. Credits

Developed by Rocher Lanka as Task 1 of the Machine Learning Internship at Auspify Technologies, October 2026.
