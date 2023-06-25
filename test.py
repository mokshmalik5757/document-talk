from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import openai
import os
from sentence_transformers import SentenceTransformer

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
import pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st
from streamlit_chat import message
# from utils import *
import tiktoken
from tempfile import NamedTemporaryFile, gettempdir
import pyautogui
from tqdm.auto import tqdm
from uuid import uuid4


load_dotenv()

api_key = st.secrets["openai"]["api_key"]

st.set_page_config(page_title="Chat with your Documents",
                   page_icon=":file_folder:")
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            
            div[class*="stTextInput"] label p {
            font-size: 20px;
            font-weight: bold;
            }
            
            div[class*="stCheckbox"] label p {
            font-size: 17px;
            font-weight: bold;
            }
            
            div[class*="stSlider"] label p {
            font-size: 14px;
            font-weight: bold;
            margin-top: 20px
            }
            
            div[class*="stTitle"] body span {
            text-align: center;
            }
            
            .css-zt5igj {
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            }
            </style>
            
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.title("Document Talk" + ":file_folder:")
st.divider()

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3,return_messages=True)

if "documents" not in st.session_state:
    st.session_state.documents = []

if "process_button" not in st.session_state:
    st.session_state.process_button = False

if "end_session" not in st.session_state:
    st.session_state.end_session = False

if "memory" not in st.session_state:
    st.session_state.memory = {}

# declaring LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key)

system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context only, 
and if the answer is not contained within the text below, just say 'I don't know'""")


human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)


response_container = st.container()

model = SentenceTransformer('all-MiniLM-L6-v2')

batch_limit = 100

embed = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# initialize pinecone
pinecone_api_key = st.secrets["pinecone"]["api_key"]
env = 'asia-southeast1-gcp-free'
index_name = 'cqa-chatbot'

pinecone.init(api_key=pinecone_api_key, environment=env)

index = pinecone.Index(index_name)

tokenizer = tiktoken.get_encoding('cl100k_base')

def find_match(input, k):
    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=k, includeMetadata=True, namespace=namespace_name)
    st.write(namespace_name)
    matches = result['matches']
    text_results = ""
    for match in matches:
        text_results += match['metadata']['text'] + "\n"
    return text_results
    # return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']


def query_refiner(conversation, query):

    response = openai.Completion.create(api_key=api_key,
    model="text-davinci-003",
    prompt=f"Given the following user query and conversation log, formulate a gramtically correct question that would be the most relevant to provide the user with an answer from a knowledge base. Only do it for queries that have a pronoun for them and a relevant noun from a conversation log that appropriately formulates the question that matches the conversation log and makes sense."
           f"Only and Only do it for those queries which have a relevant conversation log with them. Don't you ever do it for queries that do NOT have any conversation log."
           f"\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
    temperature=0.15,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response['choices'][0]['text']

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):

        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string

def get_file_text(file):
    loader = UnstructuredFileLoader(file)
    return loader.load()



# create the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

def get_vectorstore(text_chunks, namespace_name = "collection"):
    if namespace_name == None:
        namespace_name = ""

    texts = []
    metadatas = []
    # text_splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=tiktoken_len,
        separators=["\n"]
    )

    for i, record in enumerate(tqdm(text_chunks)):
        # first get metadata fields for this record
        metadata = {
            'source': record.metadata.get("source").split('\\')[-1].split('.')[0],
        }
        # now we create chunks from the record text
        record_texts = text_splitter.split_text(record.page_content)
        # create individual metadata dicts for each chunk
        record_metadatas = [{
            "chunk": j, "text": text, **metadata
        } for j, text in enumerate(record_texts)]
        # append these to current batches
        texts.extend(record_texts)
        metadatas.extend(record_metadatas)
        # if we have reached the batch_limit we can add texts
        if len(texts) >= batch_limit:
            ids = [str(uuid4()) for _ in range(len(texts))]
            embeds = embed.embed_documents(texts)
            index.upsert(vectors=zip(ids, embeds, metadatas), namespace=namespace_name)
            texts = []
            metadatas = []

    if len(texts) > 0:
        ids = [str(uuid4()) for _ in range(len(texts))]
        embeds = embed.embed_documents(texts)
        index.upsert(vectors=zip(ids, embeds, metadatas), namespace=namespace_name)
    return index


def delete_session_state():
    del st.session_state['documents']


def reset_chat():
    del st.session_state.responses

    del st.session_state.memory

    del st.session_state.requests

    del st.session_state.buffer_memory

def price_tokens(input_string: str or list, output_string: str, model_name: str) -> float:
    # Returns number of tokens in a string
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens_input = len(encoding.encode(input_string))
    num_tokens_output = len(encoding.encode(output_string))
    if model_name == "text-davinci-003":
        return ((num_tokens_input + num_tokens_output) * (0.02 / 1000))
    elif model_name == "gpt-3.5-turbo":
        return ((num_tokens_input * (0.0015 / 1000)) + (num_tokens_output * (0.002 / 1000)))

namespace_name = None

def main():
    css = r'''
        <style>
            [data-testid="stForm"] {border: 0px}
            
        </style>
    '''

    st.markdown(css, unsafe_allow_html=True)
    hide_label = """
    <style>
        .css-7oyrr6 eex3ynb0{
            display: none
    </style>
    """
    st.markdown(hide_label, unsafe_allow_html=True)
    raw_text = ""
    with st.sidebar:

        hide_browse_files_label = """
        <style>
            
        .css-7oyrr6.eex3ynb0{
            display: none;
        }
        .css-w770g5{
            display: flex;
            align-items:center
        }
        
        .css-nahz7x p {
        word-break: break-word;
        display: flex;
        text-align: center;
        align-items-center;
        justify-content:center;
        }
        
        .css-w770g5.e1ewe7hr5{
        margin-bottom: 120px;
        margin-top: 20px;
        }
        
        .css-w770g5.e1ewe7hr10{
        margin-left: auto;
        }
        
        .css-fhd5bg.eqdbnj015 {  
        max-width: 300px; /* Set the maximum width for the element */
        width: 100%; /* Allow the element to grow within the available space */
        display: flex;
        justify-content: center;
        align-items: center;
        }
        streamlit-expanderHeader st-ae st-bx st-ag st-ah st-ai st-aj st-by st-bz st-c0 st-c1 st-c2 st-c3 st-c4 st-ar st-as st-b6 st-b5 st-b3 st-c5 st-c6 st-c7 st-b4 st-c8 st-c9 st-ca st-cb st-cc {
        max-width: 100px; /* Set the maximum width for the element */
        width: 100%; /* Allow the element to grow within the available space */
        }
        .css-q8sbsg p{
        font-size:15px;
        font-weight:bold;
        }
        </style>
        """
        st.markdown(hide_browse_files_label, unsafe_allow_html=True)
        colT1, colT2, colT3 = st.columns([1, 8, 44])
        with colT2:
            st.sidebar.write("**Upload your own documents and chat!**" + " " + "💬", unsafe_allow_html=True)
        if "docs" not in st.session_state:
            st.session_state.docs = False
        colT1, colT2, colT3 = st.columns([1, 8, 64])
        with st.form(clear_on_submit=False, key="file-upload-form"):
            with colT3:
                st.session_state.docs = st.file_uploader("", accept_multiple_files=True)
                st.text("")
            colT1, colT2, colT3 = st.columns([1, 8, 24])
            with colT3:
                with st.expander("Collection name"):
                    global namespace_name
                    namespace_name = st.text_input("", placeholder="Name of your collection", help="Give your collection a name and only find documents saved in that collection")

                submitted = st.form_submit_button("Process" + " " + "\u23F3")

            if (st.session_state.docs is not None) and (submitted is not False):
                with st.spinner("Processing"):
                    for uploaded_file in st.session_state.docs:
                        file_extension = uploaded_file.name.split(".")[-1]

                        with NamedTemporaryFile(dir=gettempdir(), suffix='.' + file_extension, delete=False) as f:
                            f.write(uploaded_file.getbuffer())

                            raw_text = get_file_text(f.name)

                            vectorstore = get_vectorstore(raw_text, namespace_name)

        # reloading session state
        colT1, colT2, colT3 = st.columns([1, 8, 20])
        with colT3:
            st.text("")
            st.text("")
            end_session = st.button("Reload" + " " + "🔄" , key="reload")
        if end_session:
            with st.spinner("Reloading..."):
                delete_session_state()
                st.write("Session state expired. Please, upload your documents again. \U0001F614")

    form = st.form(key = "form_1")
    with form:
        query = st.text_input("", key="input", placeholder="Write your query")
        submit_button = st.form_submit_button(label = "Submit")
        with st.container():
            slider = st.slider("Source of answer", min_value=1, max_value=5, value=1, help="This slider lets you decide how many relevant pieces of information should the model use for output")
            st.divider()
            checkbox_col_1, checkbox_col_2 = st.columns(2)
            with checkbox_col_1:
                source = st.checkbox("Source", key = "source_checkbox", help="Source of documents")
            with checkbox_col_2:
                refined_query_checkbox = st.checkbox(" Refined Query", key = "refined_query", help = "Refined query of your original query")
                st.divider()

        col_1, col_2 = st.columns(2)
        with col_2:
            price_checkbox = st.checkbox("Price", key = "price_checkbox", help="Price of query")
        if (query != "") and (source is not False):
            with st.spinner("Typing..."):
                conversation_string = get_conversation_string()
                # st.code(conversation_string)
                refined_query = query_refiner(conversation_string, query)
                price_1 = price_tokens(input_string=conversation_string, output_string=refined_query, model_name="text-davinci-003")
                with st.container():
                    with checkbox_col_2:
                        if refined_query_checkbox:
                            st.write("<b> <font color = 'grey'> Refined Query  </font> </b> \n\n ", refined_query, unsafe_allow_html=True)
                            st.divider()
                    context = find_match(refined_query, k = slider)
                    with checkbox_col_1:
                        st.divider()
                        st.write("<b> --- Source documents --- </b>\n\n",unsafe_allow_html=True)
                        st.divider()
                        st.write(context)
                        st.divider()
                response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
                with st.container():
                    with checkbox_col_2:
                        # price_checkbox = st.checkbox("Price of query", key="price_checkbox")
                        if price_checkbox is not False:
                            with col_2:
                                price_1 = price_tokens(input_string=conversation_string, output_string=refined_query,
                                                       model_name="text-davinci-003")
                                price_2 = price_tokens(input_string=context + query, output_string=response,
                                                       model_name="gpt-3.5-turbo")
                                st.write("Price:💲" + str(round(price_1 + price_2, 4)))
            st.session_state.requests.append(query)
            st.session_state.responses.append(response)
        elif query != "":
            with st.spinner("Typing..."):
                conversation_string = get_conversation_string()
                refined_query = query_refiner(conversation_string, query)
                with st.container():
                    with checkbox_col_2:
                        if refined_query_checkbox:
                            st.write("<b> <font color = 'grey'> Refined Query  </font> </b> \n\n ", refined_query, unsafe_allow_html=True)
                            st.divider()
                context = find_match(refined_query, k=slider)
                response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
                with st.container():
                    with col_2:
                        if price_checkbox is not False:
                            with col_2:
                                price_1 = price_tokens(input_string=conversation_string, output_string=refined_query,model_name="text-davinci-003")
                                price_2 = price_tokens(input_string=context+query, output_string=response, model_name="gpt-3.5-turbo")
                                st.write("Price:💲"+ str(round(price_1 + price_2,4)))
            st.session_state.requests.append(query)
            st.session_state.responses.append(response)
        elif query == "":
            st.error("Can't send an empty query" + " "+"🚨")


    with response_container:
        if st.session_state['responses']:
            for i in range(len(st.session_state['responses'])):
                message(st.session_state['responses'][i],key=str(i))
                if i < len(st.session_state['requests']):
                    message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')

    column1, column3 = st.columns(2)
    with column1:
        if (st.button("Reset Chat")) and (query != ""):
            pyautogui.hotkey("ctrl", "F5")

    with column3:
        if (st.button("Delete Collection")):
            if namespace_name != "":
                index.delete(deleteAll='true', namespace=namespace_name)


if __name__ == '__main__':
    main()

