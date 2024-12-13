import pandas as pd
import numpy as np
import json
import re
import traceback
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

class ApolloBugCategorizer:
    def __init__(self, json_data):
        """
        Initialize the bug categorization pipeline with JSON data
        
        Args:
            json_data (list or dict): GitHub issues data
        """
        self.json_data = json_data if isinstance(json_data, list) else [json_data]
        
    @classmethod
    def from_file(cls, file_path):
        """
        Class method to create an instance from a JSON file
        
        Args:
            file_path (str): Path to the JSON file
        
        Returns:
            ApolloBugCategorizer: Initialized instance
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                # Try to load as JSON list or single object
                file_content = file.read()
                try:
                    data = json.loads(file_content)
                except json.JSONDecodeError:
                    # If not valid JSON, try to parse as a list
                    try:
                        data = eval(file_content)
                    except Exception as e:
                        print(f"Error parsing file content: {e}")
                        raise
                
                return cls(data)
        except Exception as e:
            print(f"Error reading file: {e}")
            traceback.print_exc()
            raise
        
    def load_issues(self):
        """
        Process issues from loaded JSON data
        
        Returns:
            pandas.DataFrame: DataFrame with issue data
        """
        processed_issues = []
        for issue in self.json_data:
            try:
                processed_issues.append({
                    'title': str(issue.get('title', '')),
                    'body': str(issue.get('body', '')),
                    'number': issue.get('number', 0),
                    'state': issue.get('state', ''),
                    'labels': [str(label.get('name', '')) for label in issue.get('labels', [])]
                })
            except Exception as e:
                print(f"Error processing issue: {e}")
                print(f"Problematic issue: {issue}")
        
        df = pd.DataFrame(processed_issues)
        
        # Debug: Print DataFrame info
        print("DataFrame Info:")
        print(df.info())
        print("\nColumn names:", df.columns.tolist())
        
        return df
    
    def preprocess_text(self, text):
        """
        Preprocess text for topic modeling
        
        Args:
            text (str): Input text
        
        Returns:
            str: Cleaned text
        """
        # Handle None or non-string inputs
        if not isinstance(text, str):
            text = str(text) if text is not None else ''
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove code blocks and markdown formatting
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`[^`]+`', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespaces
        text = ' '.join(text.split())
        
        return text
    
    def create_topic_model(self, df, num_topics=10):
        """
        Create BERTopic model
        
        Args:
            df (pandas.DataFrame): DataFrame with issues
            num_topics (int): Number of topics to extract
        
        Returns:
            tuple: (BERTopic model, topic assignments)
        """
        # Combine title and body
        df['full_text'] = df['title'] + ' ' + df['body']
        
        # Preprocess text
        df['processed_text'] = df['full_text'].apply(self.preprocess_text)
        
        # Remove very short documents
        df = df[df['processed_text'].str.len() > 50]
        
        # Debug: Print processed text
        print("\nProcessed Text Samples:")
        print(df['processed_text'].head())
        
        # Initialize embedding model
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Custom vectorizer to control topic diversity
        vectorizer_model = CountVectorizer(
            stop_words='english', 
            min_df=1, 
            max_df=0.95
        )
        
        # Create BERTopic model
        topic_model = BERTopic(
            embedding_model=embedding_model,
            vectorizer_model=vectorizer_model,
            nr_topics=num_topics,
            calculate_probabilities=True
        )
        
        # Fit the model
        topics, probs = topic_model.fit_transform(df['processed_text'])
        
        return topic_model, topics, df
    
    def analyze_topics(self, topic_model, df, topics):
  
        try:
            # Get topic names and their top words
            topic_info = topic_model.get_topic_info()
            
            # Debug: Print topic info columns
            print("\nTopic Info Columns:")
            print(topic_info.columns)
            
            # Prepare results DataFrame
            topic_results = []
            for topic_id in set(topics):
                if topic_id == -1:  # Skip noise topic
                    continue
                
                # Defensive column access with fallback
                topic_name = topic_info.loc[topic_info['Topic'] == topic_id, 'Name'].values
                topic_name = topic_name[0] if len(topic_name) > 0 else f"Topic {topic_id}"
                
                # Get top words, with fallback
                try:
                    top_words = topic_model.get_topic(topic_id)
                    top_words = [word for word, _ in top_words][:5]
                except Exception as e:
                    print(f"Could not retrieve top words for topic {topic_id}: {e}")
                    top_words = ['N/A']
                
                # Count issues in this topic
                topic_issues = df[topics == topic_id]
                
                topic_results.append({
                    'topic_id': topic_id,
                    'topic_name': topic_name,
                    'top_words': ', '.join(top_words),
                    'issue_count': len(topic_issues),
                    'sample_issues': topic_issues['title'].head(3).tolist()
                })
            
            return pd.DataFrame(topic_results)
        
        except Exception as e:
            print(f"Error in analyze_topics: {e}")
            traceback.print_exc()
            return pd.DataFrame()  # Return empty DataFrame if analysis fails
    
        except Exception as e:
            print(f"Error in analyze_topics: {e}")
            traceback.print_exc()
            return pd.DataFrame()  # Return empty DataFrame if analysis fails
   
    def run_analysis(self, num_topics=10):
        """
        Run complete bug categorization pipeline
        
        Args:
            num_topics (int): Number of topics to extract
        
        Returns:
            tuple: (issues DataFrame, topic model, topic results)
        """
        # Load issues 
        issues_df = self.load_issues()
        
        # Create topic model
        topic_model, topics, processed_df = self.create_topic_model(issues_df, num_topics)
        
        # Analyze topics
        topic_results = self.analyze_topics(topic_model, processed_df, topics)
        
        return issues_df, topic_model, topic_results

def main():
    # Path to your JSON file
    JSON_FILE_PATH = 'ApolloAuto_apollo_issues.json'  # Update this with your actual file path
    
    try:
        # Initialize categorizer from file
        categorizer = ApolloBugCategorizer.from_file(JSON_FILE_PATH)
        
        # Run analysis
        issues_df, topic_model, topic_results = categorizer.run_analysis(num_topics=5)
        
        # Print topic results
        print("Topic Analysis Results:")
        print(topic_results.to_string(index=False))
        
        # Optional: Visualize topics
        try:
            topic_model.visualize_topics().write_html("apollo_bug_topics.html")
            print("\nTopic visualization saved to 'apollo_bug_topics.html'")
        except Exception as e:
            print(f"Could not generate topic visualization: {e}")
    
    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()