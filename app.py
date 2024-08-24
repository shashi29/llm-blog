import streamlit as st
import praw
import os
from dotenv import load_dotenv
from langchain_openai import OpenAI, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load environment variables
load_dotenv()

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
        input_variables=["reddit_content", "keyword", "references"],
        template="""
        Create a comprehensive and engaging blog post based on the following combined text from Reddit search results about {keyword}:

        {reddit_content}

        Follow these instructions to create the blog post:

        1. Title: Generate an attention-grabbing title that accurately reflects the main topic.

        2. Introduction (1-2 paragraphs):
           - Begin with a hook to capture the reader's interest.
           - Provide context about the topic ({keyword}).
           - Briefly outline what the blog post will cover.

        3. Information (3-5 sections):
           - Organize the information from Reddit into coherent themes or subtopics.
           - For each section:
             * Use a clear subheading.
             * Present the main ideas and insights from the Reddit posts.
             * Include relevant examples or anecdotes from the Reddit content.
             * Ensure smooth transitions between sections.

        4. Expert Insights or Analysis (1-2 paragraphs):
           - Offer a deeper analysis of the topic based on the Reddit discussions.
           - Identify trends, patterns, or conflicting viewpoints if present.

        5. Practical Applications or Takeaways (1 paragraph):
           - Provide actionable advice or insights readers can apply.

        6. Conclusion (1 paragraph):
           - Summarize the key points discussed in the blog post.
           - End with a thought-provoking statement or call-to-action.

        7. Writing Style:
           - Use a conversational yet informative tone.
           - Ensure the content is well-structured and easy to read.
           - Include transitional phrases to improve flow between paragraphs and sections.

        8. Final Check:
           - Ensure the blog post is coherent, engaging, and provides value to the reader.
           - Make sure it accurately represents the main ideas from the Reddit content without directly quoting or naming specific users.

        9. References:
           - At the end of the blog post, include a "References" section.
           - List the references in the following format:
             {references}

        Generate the complete blog post following these instructions, including the references section at the end.
        """
    )
    
    chain = LLMChain(llm=llm, prompt=prompt_template)
    
    return chain.run(reddit_content=prompt, keyword=keyword, references=references_text)

def main():
    st.title("Reddit-based Blog Post Generator")
    
    # User inputs
    keyword = st.text_input("Enter the search keyword:")
    limit = st.slider("Number of Reddit posts to fetch:", min_value=1, max_value=50, value=3)
    
    if st.button("Generate Blog Post"):
        if not keyword:
            st.error("Please fill in all fields.")
        else:
            with st.spinner(f"Fetching {limit} Reddit posts..."):
                combined_text, references = get_reddit_posts(keyword, limit)
            
            st.subheader("Most Relevant Text from Reddit:")
            st.text_area("Reddit Content", combined_text, height=200)
            
            with st.spinner("Generating blog post..."):
                blog_post = generate_blog_post(combined_text, keyword, references)
            
            st.subheader("Generated Blog Post:")
            st.markdown(blog_post)

if __name__ == "__main__":
    main()