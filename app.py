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
        Craft a compelling, SEO-optimized blog post on {keyword} using insights from Reddit discussions and authoritative web sources:

        {content}

        Blog Post Structure:

        1. Title: Create an attention-grabbing, keyword-rich title (60-70 characters).

        2. Introduction (100-150 words):
        - Hook: Start with a surprising fact, question, or statistic
        - Context: Briefly explain the importance of {keyword}
        - Thesis: Clearly state the post's main argument or purpose
        - Roadmap: Outline 3-5 key points to be covered

        3. Main Body (3-5 sections, 200-300 words each):
        - Use H2 headers for main sections, H3 for subsections
        - For each section:
            * Begin with a clear, informative subheading
            * Present a key idea or argument
            * Support with evidence from the provided content
            * Include a relevant example, anecdote, or case study
            * Conclude with a transition to the next section

        4. Expert Insights (150-200 words):
        - Analyze trends, patterns, or debates within the {keyword} topic
        - Offer a unique perspective or synthesis of ideas
        - Address any counterarguments or limitations

        5. Practical Applications (100-150 words):
        - Provide 3-5 actionable tips or strategies for readers
        - Explain how to implement these ideas in real-world scenarios

        6. Conclusion (100-150 words):
        - Recap the main points without introducing new information
        - Emphasize the key takeaway or main argument
        - End with a thought-provoking question or call-to-action

        Content Guidelines:
        - Total word count: 1200-1500 words
        - Tone: Conversational yet authoritative
        - Use bullet points or numbered lists for easy readability
        - Include 2-3 relevant statistics or data points
        - Incorporate 1-2 analogies or metaphors to explain complex ideas
        - Ensure proper keyword density (use {keyword} naturally throughout)
        - Add internal links to 2-3 related topics (placeholder URLs are fine)

        SEO Optimization:
        - Include {keyword} in the title, first paragraph, and at least one H2 header
        - Use related long-tail keywords throughout the content
        - Optimize meta description (150-160 characters)

        7. References:
        Include a "Sources" section at the end, formatted as follows:
        {references}

        Final Checks:
        - Ensure coherence, logical flow, and engaging content
        - Verify that the post provides unique value to readers interested in {keyword}
        - Confirm all information is accurate and up-to-date
        - Proofread for grammar, spelling, and punctuation

        Generate the complete blog post following these guidelines, balancing depth of information with readability and SEO best practices.
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