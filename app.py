import streamlit as st
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
from langchain_community.chat_models import ChatCohere
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import requests

load_dotenv()

prompt_template = """
You will be providing summary and the details of YouTube videos. Use the transcript below text to give a professional 
appealing summary of at most 300 words and details of a Youtube video of at least 800 words. Follow this format:
Summary
Put the summaries here
Detail
Put the details here
 
{transcript_text}
"""

def extract_transcript(YT_video_url):
    try:
        video_id = get_video_id(YT_video_url)
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = ''
        for i in transcript_text:
            transcript += ' ' + i['text']
        return transcript
    except Exception as e:
        raise e

def generate_summary(transcript_text, prompt_template):
    class CustomChatCohere(ChatCohere):
        def _get_generation_info(self, response):
            generation_info = {}
            if hasattr(response, 'token_count'):
                generation_info["token_count"] = response.token_count
            return generation_info

        def get_num_tokens(self, text: str) -> int:
            response = self.client.tokenize(text=text, model=self.model)
            return len(response.tokens)

    llm = CustomChatCohere()
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["transcript_text"])
    
    chain = prompt | llm
    
    response = chain.invoke(transcript_text)
    return response.content

def get_video_id(youtube_url):
    parsed_url = urlparse(youtube_url)
    video_id = parse_qs(parsed_url.query).get('v')
    if video_id:
        return video_id[0]
    return None

def get_thumbnail_url(video_id):
    resolutions = ['maxresdefault', 'sddefault', 'hqdefault', 'mqdefault', 'default']
    for res in resolutions:
        url = f"http://img.youtube.com/vi/{video_id}/{res}.jpg"
        response = requests.get(url)
        if response.status_code == 200:
            return url
    return None  # Return None if no valid thumbnail is found

st.set_page_config(page_title='Youtube Transcribe', page_icon='🤖')
st.title('Youtube Transcript to Note Converter')
youtube_link = st.text_input('Paste Youtube video link')

if youtube_link:
    try:
        video_id = get_video_id(youtube_link)
        if video_id:
            thumbnail_url = get_thumbnail_url(video_id)
            if thumbnail_url:
                st.image(thumbnail_url, use_column_width=True)
            else:
                st.error("Thumbnail not available.")
        else:
            st.error("Invalid YouTube link format. Please check the URL.")
    except IndexError:
        st.error("Invalid YouTube link format. Please check the URL.")

if st.button("Get Video Summary"):
    transcript_text = extract_transcript(youtube_link)

    if transcript_text:
        summary = generate_summary(transcript_text, prompt_template)
        st.write(summary)
