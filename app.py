import streamlit as st
import praw
import os
from dotenv import load_dotenv
from langchain_openai import OpenAI, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import time
import random
from typing import Dict, List
import requests
from bs4 import BeautifulSoup
from googlesearch import search

# Load environment variables
load_dotenv()

class BlogScraper:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.google.com/",
        }
        self.session = requests.Session()

    def search_blogs(self, topic: str, num_results: int = 10) -> List[str]:
        query = f"{topic} related blogs Medium"
        urls = list(search(query, num_results=num_results))
        return urls

    def scrape_content(self, url: str) -> str:
        try:
            response = self.session.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text content
            text = soup.get_text(separator=" ", strip=True)

            # Remove extra whitespace
            text = " ".join(text.split())

            return text
        except requests.exceptions.RequestException as e:
            print(f"Error scraping {url}: {str(e)}")
            return ""

    def scrape_blogs(self, topic: str, urls_per_topic: int = 5) -> List[Dict[str, str]]:
        urls = self.search_blogs(topic, urls_per_topic)
        topic_results = []

        for url in urls:
            content = self.scrape_content(url)
            if content:
                topic_results.append({"url": url, "content": content})
            time.sleep(random.uniform(1, 3))  # Randomized sleep time

        return topic_results

def get_reddit_posts(keyword, limit):
    # Reddit app credentials
    client_id = "2IhTJfbQfPiwvDSoou7lZw"  # os.getenv('REDDIT_CLIENT_ID')
    client_secret = "nsU99R0sm5Wc_8LJXnT0mhh0seNK0g"  # os.getenv('REDDIT_CLIENT_SECRET')
    user_agent = "test-backend"  # os.getenv('REDDIT_USER_AGENT')
    
    # Initialize Reddit instance
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent
    )
    
    combined_text = ""
    references = []
    
    # Search for subreddits related to the keyword
    subreddits = reddit.subreddits.search(keyword, limit=limit)
    
    for subreddit in subreddits:
        try:
            # Search within each subreddit found
            search_results = reddit.subreddit(subreddit.display_name).search(keyword, limit=limit)
            
            for post in search_results:
                combined_text += f"Subreddit: {subreddit.display_name}\n"
                combined_text += f"Title: {post.title}\n"
                combined_text += f"Text: {post.selftext}\n\n"
                references.append({
                    "subreddit": subreddit.display_name,
                    "title": post.title,
                    "url": f"https://www.reddit.com{post.permalink}"
                })
        except Exception as ex:
            continue
    
    return combined_text, references

def generate_blog_post(prompt, keyword, references):
    llm = ChatOpenAI(model_name="gpt-4o-mini")
    
    references_text = "\n".join([f"{i+1}. [{ref['title']}]({ref['url']})" for i, ref in enumerate(references)])
    
    prompt_template = PromptTemplate(
        input_variables=["content", "keyword", "references"],
        template="""
        Craft an engaging and informative blog post on {keyword} using the following content from Reddit and Google search results:

        {content}

        Structure the blog post as follows:

        1. Title: Create an attention-grabbing, SEO-friendly title that accurately represents the main topic.

        2. Introduction (100-150 words):
        - Open with a compelling hook
        - Provide context for {keyword}
        - Outline the key points the post will cover

        3. Main Body (3-5 sections, 200-300 words each):
        - Organize information into coherent themes or subtopics
        - For each section:
            * Use descriptive subheadings
            * Present key ideas and insights
            * Include relevant examples or anecdotes
            * Ensure smooth transitions between sections

        4. Expert Analysis (150-200 words):
        - Offer in-depth analysis of the topic
        - Highlight trends, patterns, or conflicting viewpoints

        5. Practical Applications (100-150 words):
        - Provide actionable advice or insights for readers

        6. Conclusion (100-150 words):
        - Summarize key points
        - End with a thought-provoking statement or call-to-action

        Writing Guidelines:
        - Maintain a conversational yet informative tone
        - Ensure logical flow and readability
        - Use transitional phrases between paragraphs and sections
        - Accurately represent main ideas without direct quotes or naming users
        - Aim for a total word count of 1200-1500 words

        7. References:
        Include a "References" section at the end, listing sources as follows:
        {references}

        Ensure the final blog post is coherent, engaging, and valuable to readers interested in {keyword}.
        """
    )
    
    chain = LLMChain(llm=llm, prompt=prompt_template)
    
    return chain.run(content=prompt, keyword=keyword, references=references_text)

def main():
    st.title("Reddit and Google Blog Post Generator")
    
    # User inputs
    keyword = st.text_input("Enter the search keyword:")
    reddit_limit = st.slider("Number of Reddit posts to fetch:", min_value=1, max_value=50, value=4)
    google_limit = st.slider("Number of Google search results to fetch:", min_value=1, max_value=20, value=5)
    
    if st.button("Generate Blog Post"):
        if not keyword:
            st.error("Please fill in all fields.")
        else:
            with st.spinner(f"Fetching {reddit_limit} Reddit posts and {google_limit} Google search results..."):
                reddit_text, reddit_references = get_reddit_posts(keyword, reddit_limit)
                
                scraper = BlogScraper()
                google_results = scraper.scrape_blogs(keyword, google_limit)
                
                google_text = "\n\n".join([f"URL: {result['url']}\nContent: {result['content']}" for result in google_results])
                google_references = [{"title": f"Google Search Result {i+1}", "url": result['url']} for i, result in enumerate(google_results)]
                
                combined_text = f"Reddit Content:\n{reddit_text}\n\nGoogle Search Content:\n{google_text}"
                combined_references = reddit_references + google_references
            
            st.subheader("Most Relevant Text from Reddit and Google:")
            st.text_area("Combined Content", combined_text, height=200)
            
            with st.spinner("Generating blog post..."):
                blog_post = generate_blog_post(combined_text, keyword, combined_references)
            
            st.subheader("Generated Blog Post:")
            st.markdown(blog_post)

if __name__ == "__main__":
    main()