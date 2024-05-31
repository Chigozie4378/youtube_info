import streamlit as st
from key import cohere_api_key
# Cohere Model
from langchain_community.chat_models import ChatCohere
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

prompt_template = """
You will be providing summary and the details of YouTube videos. Use the transcript below text to give a professional 
appealing summary of at most 300 words and details of a Youtube video of at least 800 words. Follow this format:
Summary
Put the summaries here
Detail
Put the details here
 
{transcript_text}
"""

# Getting transcript from the Youtube video
def extract_transcript(YT_video_url):
    try:
        video_id = YT_video_url.split("=")[1]
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = ''
        for i in transcript_text:
            transcript += ' ' + i['text']
        return transcript
    
    except Exception as e:
        raise e

# Getting the summary
def generate_summary(transcript_text, prompt_template):
    class CustomChatCohere(ChatCohere):
        def _get_generation_info(self, response):
            # Custom handling of generation info
            generation_info = {}
            if hasattr(response, 'token_count'):
                generation_info["token_count"] = response.token_count
            # Add other attributes if needed
            return generation_info

        def get_num_tokens(self, text: str) -> int:
            # Specify the model explicitly
            response = self.client.tokenize(text=text, model=self.model)
            return len(response.tokens)

    llm = CustomChatCohere(cohere_api_key=cohere_api_key)
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["transcript_text"])
    chain = LLMChain(prompt=prompt, llm=llm)
    
    response = chain.run(transcript_text)
    return response

st.set_page_config(page_title='Youtube Transcribe', page_icon='🤖')
st.title('Youtube Transcript to Note Converter')
youtube_link = st.text_input('Paste Youtube video link')

if youtube_link:
    video_id = youtube_link.split("=")[1]
    st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)

if st.button("Get Video Summary"):
    transcript_text = extract_transcript(youtube_link)

    if transcript_text:
        summary = generate_summary(transcript_text, prompt_template)
        st.write(summary)
