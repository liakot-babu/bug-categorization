import json
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Load GitHub issues dataset
with open("ApolloAuto_apollo_issues.json", "r") as file:
    data = json.load(file)

# Preprocess the text
def preprocess_text(text):
    text = re.sub(r'[\n\t\r]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower().strip()

# Extract issue descriptions
issues = []
for issue in data:
    title = issue.get("title", "")
    body = issue.get("body", "")
    comments = issue.get("comments", []) if isinstance(issue.get("comments", []), list) else []

    combined_text = f"{title} {body} "
    for comment in comments:
        combined_text += comment.get("body", "") + " "

    cleaned_text = preprocess_text(combined_text)
    issues.append(cleaned_text)

# Create a DataFrame
df = pd.DataFrame(issues, columns=["description"])

# Vectorize the text
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
doc_term_matrix = vectorizer.fit_transform(df['description'])

# Fit LDA model
lda = LatentDirichletAllocation(n_components=12, random_state=42)
lda.fit(doc_term_matrix)

# Get topic-word distribution
def get_topics(lda_model, vectorizer, top_n=10):
    topics = {}
    for idx, topic in enumerate(lda_model.components_):
        top_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-top_n - 1:-1]]
        topics[idx] = top_words
    return topics

topics = get_topics(lda, vectorizer)

# Assign topics to documents
doc_topics = lda.transform(doc_term_matrix)
df['topic'] = doc_topics.argmax(axis=1)

# Summarize topic counts and keywords
def create_summary_table(lda_model, topics, doc_topics):
    rows = []
    for topic_id, keywords in topics.items():
        count = (doc_topics.argmax(axis=1) == topic_id).sum()
        rows.append({
            "Topic": topic_id,
            "Count": count,
            "Name": "_".join(keywords[:3]),
            "Representation": ", ".join(keywords)
        })
    return pd.DataFrame(rows)

summary_table = create_summary_table(lda, topics, doc_topics)
summary_table = summary_table.sort_values(by="Count", ascending=False).reset_index(drop=True)
print(summary_table)

# Save summary table
summary_table.to_csv("lda_summary_table_apollo.csv", index=False)
