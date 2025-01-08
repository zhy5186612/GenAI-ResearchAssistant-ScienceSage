# This file contains utility function for scienceSage.
# Yong Zhang, 6/20/2024

## load the packages
# langchain
from langchain_community.chat_models import AzureChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import requests
from bs4 import BeautifulSoup
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.retrievers import ArxivRetriever

# # llama_index
from llama_index.embeddings import LangchainEmbedding
from langchain.embeddings import HuggingFaceEmbeddings, AzureOpenAIEmbeddings, OpenAIEmbeddings
from llama_index import QueryBundle
from llama_index import ServiceContext, StorageContext
from llama_index.schema import NodeWithScore
from llama_index.retrievers import BaseRetriever, VectorIndexRetriever, KGTableRetriever

# packages to use GPT4V
from moviepy.editor import VideoFileClip
import speech_recognition as sr
from llama_index.multi_modal_llms.azure_openai import AzureOpenAIMultiModal
from pathlib import Path
from llama_index.vector_stores import LanceDBVectorStore
from io import BytesIO
from PIL import Image
import glob

#get credentials
from azure.identity import EnvironmentCredential
from nebula3.data.DataObject import Node, PathWrapper, Relationship
from nebula3.Config import SessionPoolConfig
from nebula3.gclient.net.SessionPool import SessionPool
#
# modules to create a new graph space
import time
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
from nebula3.common import *
#
import weaviate

from pyvis.network import Network
import json
import os
import re
from ast import literal_eval
from typing import List, Dict
import base64
import pandas as pd
import streamlit.components.v1 as components
import streamlit as st

# save and downlaod a report 
import markdown2
import pdfkit
import datetime
import random
import string
# import openai
from dotenv import load_dotenv
load_dotenv()
#
# P&G Azure OAI credentials 
GENAI_PROXY = os.getenv('GENAI_PROXY')
CONGNITIVE_SERVICES = os.getenv('CONGNITIVE_SERVICES')
OPEN_API_VERSION = os.getenv('OPEN_API_VERSION')
LLM_DEPLOYMENT_NAME = os.getenv('DEPLOYMENT_NAME')
EMBEDDING_DEPLOYMENT_NAME = os.getenv('EMBEDDING_DEPLOYMENT_NAME')
# GPT4 vision 
DEPLOYMENT_NAME_GPT4V=os.getenv('DEPLOYMENT_NAME_GPT4V')
GPT4V_VERSION=os.getenv('GPT4V_VERSION')
#
env_credential = EnvironmentCredential()
# Replace userid with a real username (a person actually providing the prompt)
HEADERS = {
    "userid": "your-usr-id",
    "project-name": "your-project-name",
}
#
# monitor the app
from galileo_observe import GalileoObserveCallback

# # @st.cache_data(show_spinner=False)
# def getHandler():
#     import promptquality as pq
#     pq.login(console_url=os.getenv('GALILEO_CONSOLE_URL'))
#     galileo_handler = pq.GalileoPromptCallback(
#         project_name=os.getenv('AIS_PROJECT_NAME'), scorers=[pq.Scorers.latency, pq.Scorers.toxicity]
#         )
#     return galileo_handler
#
# scienceSage: generate a report by using a research question and internet
class scienceSage:
    def __init__(self, dataSource, searchMethod, llm_tag, prompt, question, numQuery, numRecords):
        self.dataSource = dataSource
        self.searchMethod = searchMethod
        self.prompt = prompt
        self.llm_tag = llm_tag
        self.question = question
        self.numQuery = numQuery
        self.numRecords = numRecords
        ### Using Azure ChatGPT
        if  self.llm_tag == 'AzureChatGPT':
            # Azure OpenAI The context window of gpt-4 is 32k tokens
            token = env_credential.get_token(CONGNITIVE_SERVICES).token
            # print(token)
            self.llm = AzureChatOpenAI(
                azure_endpoint=GENAI_PROXY,
                azure_deployment=LLM_DEPLOYMENT_NAME,
                api_version=OPEN_API_VERSION,
                api_key=token,
                temperature=0.02,
                default_headers=HEADERS,
                max_tokens = 7000)
        ### Using Mistral_7B model
        elif self.llm_tag == 'Mixtral_8X7B':
            # 8K context window for Mistral_7B
            self.llm = ChatOpenAI( temperature=0.02, max_tokens = 7000, 
                                   request_timeout=600, max_retries=10,
                                   openai_api_key=os.getenv('MISTRAL_7B_API_KEY'),
                                   openai_api_base=os.getenv('MISTRAL_7B_API_BASE'),
                                )     
        else:
            raise ValueError('---llam_tag need to be either AzureChatGPT or Mixtral_8X7B---')
    
    # function to save report and enable download
    def save_download_report(self, report, sageName):
        ## save to SageName and enable pdf file download
        date = datetime.datetime.now()
        # fileName = 'scienceSage_report_' + date.strftime("%Y%m%d") + '_' + ''.join(random.choices(string.ascii_letters + string.digits, k=6)) + '.pdf'
        fileName = 'scienceSage_report_' + date.strftime("%Y%m%d") + '_' + ''.join(random.choices(string.ascii_letters + string.digits, k=6)) + '.txt'

        ## save report to the chatDocText folder under the same sageName
        # filePath = './ssData/' + sageName + '/chatDocText/' + fileName[:-4] + '.txt'
        filePath = './ssData/' + sageName + '/chatDocText/' + fileName
        with open(filePath, "w") as fileI:
            fileI.write(report)
        fileI.close()

        # ## enable download button
        # # need to install wkhtmltopdf: https://github.com/JazzCore/python-pdfkit/wiki/Installing-wkhtmltopdf
        # options = {
        #     'page-size': 'Letter',
        #     'margin-top': '1.2in',
        #     'margin-right': '0.9in',
        #     'margin-bottom': '0.9in',
        #     'margin-left': '0.9in',
        #     'encoding': "UTF-8",
        #     'header-center': '---Generared by ScienceSage on ' + date.strftime("%x") +'---',
        #     # 'custom-header' : [
        #     #     ('Accept-Encoding', 'gzip')
        #     # ],            
        #     'no-outline': None
        #     }
        # html = markdown2.markdown(report)
        # fileContents = pdfkit.from_string(html, options=options)
        # st.download_button("Download Your Report", fileContents, fileName)
        st.download_button("Download Your Report", report, fileName)

    # function to put retrieved contents in a list for each of the 3 queries generated by LLM 
    def collapse_list_of_lists(self, list_of_lists):
        content = []
        for l in list_of_lists:
            content.append("\n\n".join(l))
        return "\n\n".join(content)

    # conduct websearch
    def web_search(self, query: str):
        # RESULTS_PER_QUESTION = 3
        ddg_search = DuckDuckGoSearchAPIWrapper()
        results = ddg_search.results(query, self.numRecords)
        return [r["link"] for r in results] 
    
    # 
    def extract_search_queries(self, input_text):
        # Define the regex pattern to match the search queries
        pattern = r'"(.*?)"'
        # Find all matches using the regex pattern
        matches = re.findall(pattern, input_text)
        return matches


    # scrape the webpage
    def scrape_text(self, url: str):
        # Send a GET request to the webpage
        try:
            response = requests.get(url)

            # Check if the request was successful
            if response.status_code == 200:
                # Parse the content of the request with BeautifulSoup
                soup = BeautifulSoup(response.text, "html.parser")

                # Extract all text from the webpage
                page_text = soup.get_text(separator=" ", strip=True)

                # Print the extracted text
                return page_text
            else:
                return f"Failed to retrieve the webpage: Status code {response.status_code}"
        except Exception as e:
            print(e)
            return f"Failed to retrieve the webpage: {e}"
          
    # Generate a report by using a Archive database
    def generateReportArchive(self):
        #
        retriever = ArxivRetriever(top_k_results=self.numRecords)
        # from langchain.retrievers import PubMedRetriever
        # retriever = PubMedRetriever()
        ###
        SUMMARY_TEMPLATE = '"""' + self.prompt['summaryPrompt'] + '"""'
        SUMMARY_PROMPT = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)
        #
        scrape_and_summarize_chain = RunnablePassthrough.assign(
                summary =  SUMMARY_PROMPT | self.llm | StrOutputParser()
        ) | (lambda x: f"Title: {x['text'].metadata['Title']}\n\nSUMMARY: {x['summary']}")
        #
        # archive retriever
        web_search_chain = RunnablePassthrough.assign(
            texts = lambda x: retriever.get_summaries_as_docs(x["question"])
        )| (lambda x: [{"question": x["question"], "text": u} for u in x["texts"]]) | scrape_and_summarize_chain.map()

        ### it is tricky to have the search prompt formated. It has to follow some tricky format
        queryList = ["query " + str(i+1) for i in range(self.numQuery)]
        #
        SEARCH_PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "user",
                    'Write ' + str(self.numQuery) + ' google search queries to search online that form an '
                    'objective opinion from the following: {question}\n'
                    'You must respond with a list of strings in the following format: ' +  json.dumps(queryList) + '.',
                ),
            ]
        )
       
        search_question_chain = SEARCH_PROMPT | self.llm | StrOutputParser() | self.extract_search_queries

        full_research_chain = search_question_chain | (lambda x: [{"question": q} for q in x]) | web_search_chain.map()
        #
        WRITER_SYSTEM_PROMPT = '"' + self.prompt['rolePrompt'] + '"'
        # 
        RESEARCH_REPORT_TEMPLATE = '"""' + self.prompt['reportPrompt'] + '"""'
        # 
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", WRITER_SYSTEM_PROMPT),
                ("user", RESEARCH_REPORT_TEMPLATE),
            ]
        )
        #
        chain = RunnablePassthrough.assign(
            research_summary= full_research_chain | self.collapse_list_of_lists
        ) | prompt | self.llm | StrOutputParser()
        #
        # use galileo to monitor the app
        try:
            monitor_handler=GalileoObserveCallback(project_name=os.getenv("SS_PROJECT_NAME"))
            report = chain.invoke({'question':self.question}, config=(dict(callbacks=[monitor_handler])))
        except:
            report = chain.invoke({'question':self.question})

        # # use galileo to monitor the app
        # try:
        #     galileo_handler = getHandler()
        #     report = chain.invoke({'question':self.question}, config=(dict(callbacks=[galileo_handler])))
        #     galileo_handler.finish()
        # except:
        #     report = chain.invoke({'question':self.question})       
        # # 
              
        return report
    
    #
    def generateReportInternet(self):
        ### Using Internet Search
        SUMMARY_TEMPLATE = '"""' + self.prompt['summaryPrompt'] + '"""'
        SUMMARY_PROMPT = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)
        
        ###
        scrape_and_summarize_chain = RunnablePassthrough.assign(
            summary = RunnablePassthrough.assign(
            text=lambda x: self.scrape_text(x["url"])[:10000]
        ) | SUMMARY_PROMPT | self.llm | StrOutputParser()
        ) | (lambda x: f"URL: {x['url']}\n\nSUMMARY: {x['summary']}")

        web_search_chain = RunnablePassthrough.assign(
            urls = lambda x: self.web_search(x["question"])
        ) | (lambda x: [{"question": x["question"], "url": u} for u in x["urls"]]) | scrape_and_summarize_chain.map()

        ### it is tricky to have the search prompt formated. It has to follow some tricky format
        queryList = ["query " + str(i+1) for i in range(self.numQuery)]
        
        SEARCH_PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "user",
                    'Write ' + str(self.numQuery) + ' google search queries to search online that form an '
                    'objective opinion from the following: {question}\n'
                    'You must respond with a list of strings in the following format: ' +  json.dumps(queryList) + '.',
                ),
            ]
        )

        search_question_chain = SEARCH_PROMPT | self.llm | StrOutputParser() | self.extract_search_queries

        full_research_chain = search_question_chain | (lambda x: [{"question": q} for q in x]) | web_search_chain.map()

        WRITER_SYSTEM_PROMPT = '"' + self.prompt['rolePrompt'] + '"'
        #
        # Report prompts from https://github.com/assafelovic/gpt-researcher/blob/master/gpt_researcher/master/prompts.py
        RESEARCH_REPORT_TEMPLATE = '"""' + self.prompt['reportPrompt'] + '"""'
        #
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", WRITER_SYSTEM_PROMPT),
                ("user", RESEARCH_REPORT_TEMPLATE),
            ]
        )
        #
        chain = RunnablePassthrough.assign(
            research_summary= full_research_chain | self.collapse_list_of_lists
        ) | prompt | self.llm | StrOutputParser()
        #
        # use galileo to monitor the app
        try:
            monitor_handler=GalileoObserveCallback(project_name=os.getenv("SS_PROJECT_NAME"))
            report = chain.invoke({'question':self.question}, config=(dict(callbacks=[monitor_handler])))
        except:
            report = chain.invoke({'question':self.question})
        return report

    # generate a report
    def generateReport(self):
        if self.dataSource=='internet': 
            if self.searchMethod=='searchInternet':
                res = self.generateReportInternet()
            elif self.searchMethod=='searchArxiv':
                res = self.generateReportArchive()
            else:
                raise ValueError('---searchMethod need to be either searchInternet or searchArchive---')
        elif self.dataSource=='localFile':
            res = '--- This module is still under development, please come back and check again in future!---'
        else:
            raise ValueError('---dataSource need to be either internet or localFile---')
        #
        return res

# customized retriever to generate answers from both vector index and KG index
class CustomRetriever(BaseRetriever):
    """Custom retriever that performs both Vector search and Knowledge Graph search"""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        kg_retriever: KGTableRetriever,
        mode: str = "OR",
    ) -> None:
        """Init params."""

        self._vector_retriever = vector_retriever
        self._kg_retriever = kg_retriever
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        kg_nodes = self._kg_retriever.retrieve(query_bundle)

        vector_ids = {n.node.node_id for n in vector_nodes}
        kg_ids = {n.node.node_id for n in kg_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in kg_nodes})

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(kg_ids)
        else:
            retrieve_ids = vector_ids.union(kg_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes

# save uploaded files into a folder
def save_uploadedfile(uploadedfile, dirName):
    with open(os.path.join(dirName, uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())

# load prompt template
def loadPrompt(promptR):
    st.write('<p style="font-size:20px; color:green;">Scroll down as needed inside each box to see and edit the whole prompt. \
        Do not change the varaiables in curvy bracket, namely {text}, {question}, {research_summary}; please also keep the split/division lines </p>', unsafe_allow_html=True)
    rolePrompt = st.text_area("Use This Template to Customize the Role of Your Assistant:",promptR['rolePrompt'], height=100)
    summaryPrompt = st.text_area("Use This Template to Customize Your Summarization Prompt:", promptR['summaryPrompt'], height=300)
    reportPrompt = st.text_area("Use This Template to Customize Your Report Generation Prompt:", promptR['reportPrompt'], height=300)
    prompt = {'rolePrompt' : rolePrompt, 'summaryPrompt' : summaryPrompt, 'reportPrompt' : reportPrompt} 
    #
    return prompt

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Ask me questions now!"}] #
# load multimodal RAG prompt template
def loadMragPrompt(promptR):
    # st.write('<p style="font-size:20px; color:green;">Scroll down as needed inside each box to see and edit the whole prompt. \
    #     Do not change the varaiables in curvy bracket, namely {context_str}, {metaData_prompt}, {query_prompt}; please also keep the split/division lines </p>', unsafe_allow_html=True)
    st.write('<p style="font-size:20px; color:green;">Scroll down as needed inside each box to see and edit the whole prompt.</p>', unsafe_allow_html=True)
    query_prompt = st.text_area("Use This Template to Customize Your Query Prompt:", promptR['query_prompt'], height=100)
    metaData_prompt = st.text_area("Use This Template to Provide Any General Information of Your Videos:",promptR['metaData_prompt'], height=100)
    role_prompt = st.text_area("Use This Template to Customize the Role of Your Assistant:", promptR['role_prompt'], height=100)
    rag_prompt = promptR['rag_prompt']
    prompt = {'query_prompt' : query_prompt, 'metaData_prompt' : metaData_prompt, 'role_prompt' : role_prompt, 'rag_prompt' : rag_prompt} 
    #
    return prompt

# cache the weaviate client
# @st.cache_resource(show_spinner=False)
def get_weaviate_client():
    #
    weaviate_client = weaviate.Client(url=os.getenv('WEAVIATE_URL'), 
                                    auth_client_secret=weaviate.AuthApiKey(api_key=os.getenv('WEAVIATE_APIKEY')))
    return weaviate_client
# function to create a new graph space
def create_kg_space(spaceName):
    client = None
    try:
        config = Config()
        config.max_connection_pool_size = 8
        # init connection pool
        connection_pool = ConnectionPool()
        assert connection_pool.init([(os.getenv('NEBULA_BASE_URL'), os.getenv('NEBULA_PORT'))], config)

        # get session from the pool
        client = connection_pool.get_session(os.getenv('NEBULA_USER'), os.getenv('NEBULA_PASSWORD'))
        assert client is not None

        # create space
        client.execute(
            # 'CREATE SPACE IF NOT EXISTS test(vid_type=FIXED_STRING(30)); USE test;'
            'CREATE SPACE IF NOT EXISTS ' + spaceName + '(vid_type=FIXED_STRING(256), partition_num=1, replica_factor=1);'
            'USE ' + spaceName + ';'
            'CREATE TAG IF NOT EXISTS entity(name string);'
            'CREATE EDGE IF NOT EXISTS relationship(relationship string);'
            'CREATE TAG INDEX entity_index ON entity(name(256));'
        )

        # insert data need to sleep after create schema
        time.sleep(10)

        print("New Graph Space " + spaceName + " Created Successfully")

    except Exception as x:
        import traceback

        print(traceback.format_exc())
        if client is not None:
            client.release()
        exit(1)

# function to drop a graph space
def delete_kg_space(spaceName):
    client = None
    try:
        config = Config()
        config.max_connection_pool_size = 8
        # init connection pool
        connection_pool = ConnectionPool()
        assert connection_pool.init([(os.getenv('NEBULA_BASE_URL'), os.getenv('NEBULA_PORT'))], config)

        # get session from the pool
        client = connection_pool.get_session(os.getenv('NEBULA_USER'), os.getenv('NEBULA_PASSWORD'))
        assert client is not None

        # drop space
        resp = client.execute('DROP SPACE ' + spaceName)
        assert resp.is_succeeded(), resp.error_msg()

        print("Graph Space " + spaceName + " has been deleted successfully")

    except Exception as x:
        import traceback

        print(traceback.format_exc())
        if client is not None:
            client.release()
        exit(1)

# extract filename from a path
def extract_filename(input_string):
    # Split the input string by '/' and get the last part
    parts = input_string.split('/')
    last_part = parts[-1]
    
    # Use regex to match the desired pattern
    match = re.match(r'^(.*?)_part_\d+$', last_part)
    if match:
        return match.group(1)
    else:
        return last_part

# extract filenames from a list of paths
def extract_filenames(filepaths):
    filenames = [extract_filename(fp) for fp in filepaths]
    return list(set(filenames))

# setup directories for new Sage
@st.cache_data(show_spinner=False)
def setupSageDir(sageName):
    # sageName should be unique, if not, it will raise an error and stop creating the directory
    os.makedirs('./ssData/' + sageName + '/chatDocText', exist_ok=False)
    os.makedirs('./ssData/' + sageName + '/audioData', exist_ok=False)
    os.makedirs('./ssData/' + sageName + '/imageData', exist_ok=False)
    os.makedirs('./ssData/' + sageName + '/videoData', exist_ok=False)
    os.makedirs('./ssData/' + sageName + '/textData', exist_ok=False)
    st.write('---New Sage Directory is Created---') 

# get sage names
def getSageNames():
    sageNames = ['PgSage']
    sageNamesR = os.listdir('./ssData')
    # no need to show .DS_Store and demoSage
    if '.DS_Store' in sageNamesR:
        sageNamesR.remove('.DS_Store')
    #
    sageNamesR.remove('demoSage')
    sageNamesR.remove('PgSage')
    sageNames.extend(sageNamesR)
    sageNames.append('Create a new knowlege base')
    return sageNames

class rag:
    def __init__(self, llm_tag):
        self.llm_tag = llm_tag
        ### Using Azure ChatGPT
        if  self.llm_tag == 'AzureChatGPT':
            # Azure OpenAI The context window of gpt-4 is 32k tokens
            token = env_credential.get_token(CONGNITIVE_SERVICES).token
            # print(token)
            try:
                monitor_handler=GalileoObserveCallback(project_name=os.getenv("SS_PROJECT_NAME"))
                self.llm = AzureChatOpenAI(
                    azure_endpoint=GENAI_PROXY,
                    azure_deployment=LLM_DEPLOYMENT_NAME,
                    api_version=OPEN_API_VERSION,
                    api_key=token,
                    temperature=0.02,
                    default_headers=HEADERS,
                    max_tokens = 600,
                    callbacks=[monitor_handler],
                    )
            except:
                self.llm = AzureChatOpenAI(
                    azure_endpoint=GENAI_PROXY,
                    azure_deployment=LLM_DEPLOYMENT_NAME,
                    api_version=OPEN_API_VERSION,
                    api_key=token,
                    temperature=0.02,
                    default_headers=HEADERS,
                    max_tokens = 600,
                    )
            # # AzureOpenAIEmbeddings is working
            # self.embed_model = AzureOpenAIEmbeddings(azure_endpoint=GENAI_PROXY,
            #                         deployment=EMBEDDING_DEPLOYMENT_NAME,
            #                         api_version=OPEN_API_VERSION,
            #                         api_key=token,
            #                         default_headers=HEADERS,)
            # self.chunkSize = 7000
        ### Using Mistral_7B model
        elif self.llm_tag == 'Mixtral_8X7B':
            # 8K context window for Mistral_7B
            try:
                monitor_handler=GalileoObserveCallback(project_name=os.getenv("SS_PROJECT_NAME"))
                # 8K context window for Mistral_7B
                self.llm = ChatOpenAI( temperature=0.02, max_tokens = 600, 
                                    request_timeout=600, max_retries=10,
                                    openai_api_key=os.getenv('MISTRAL_7B_API_KEY'),
                                    openai_api_base=os.getenv('MISTRAL_7B_API_BASE'),
                                    callbacks=[monitor_handler],
                                    )
            except:
                self.llm = ChatOpenAI( temperature=0.02, max_tokens = 600, 
                    request_timeout=600, max_retries=10,
                    openai_api_key=os.getenv('MISTRAL_7B_API_KEY'),
                    openai_api_base=os.getenv('MISTRAL_7B_API_BASE'),
                    )
            # # HuggingFaceEmbeddings is working
            # model_name = "sentence-transformers/all-mpnet-base-v2"
            # model_kwargs = {'device': 'cpu'}
            # encode_kwargs = {'normalize_embeddings': False}
            # myEmbedding = HuggingFaceEmbeddings(
            #         model_name=model_name,
            #         model_kwargs=model_kwargs,
            #         encode_kwargs=encode_kwargs
            # )        
            # self.embed_model = LangchainEmbedding(myEmbedding)
            # self.chunkSize = 7000     
        else:
            raise ValueError('---llam_tag need to be either AzureChatGPT or Mixtral_8X7B---')
        # HuggingFaceEmbeddings is working
        model_name = "sentence-transformers/all-distilroberta-v1"
        model_kwargs = {'device': 'cpu'} # mps:
        encode_kwargs = {'normalize_embeddings': False}
        myEmbedding = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
        )        
        self.embed_model = LangchainEmbedding(myEmbedding)
        # self.chunkSize = 7000         
        #          
    ## function to extract reference information related to the answer
    def outputReferences(self,response):
        # 
        for item in response.metadata.values():
            if 'file_name' in item:
                st.write(f"File name: {item['file_name']}")        
            if 'page_label' in item:
                st.write(f"Page number: {item['page_label']}")    

    # query the KG
    def query_nebulagraph(self,query, space_name="researchAssistant", address=os.getenv('NEBULA_BASE_URL'), 
                        port=os.getenv('NEBULA_PORT'), user=os.getenv('NEBULA_USER'), password=os.getenv('NEBULA_PASSWORD')):
        # from nebula3.Config import SessionPoolConfig
        # from nebula3.gclient.net.SessionPool import SessionPool

        config = SessionPoolConfig()
        session_pool = SessionPool(user, password, space_name, [(address, port)])
        session_pool.init(config)
        return session_pool.execute(query)
    # 
    def result_to_df(self,result):
        # from typing import Dict

        columns = result.keys()
        d: Dict[str, list] = {}
        for col_num in range(result.col_size()):
            col_name = columns[col_num]
            col_list = result.column_values(col_name)
            d[col_name] = [x.cast() for x in col_list]
        return pd.DataFrame(d)
    #
    def render_pd_item(self, g, item):
        # from nebula3.data.DataObject import Node, PathWrapper, Relationship

        if isinstance(item, Node):
            node_id = item.get_id().cast()
            tags = item.tags()  # list of strings
            props = dict()
            for tag in tags:
                props.update(item.properties(tag))
            g.add_node(node_id, label=node_id, title=str(props))
        elif isinstance(item, Relationship):
            src_id = item.start_vertex_id().cast()
            dst_id = item.end_vertex_id().cast()
            edge_name = item.edge_name()
            #edge_name = item.name()
            props = item.properties()
            # ensure start and end vertex exist in graph
            if not src_id in g.node_ids:
                g.add_node(src_id)
            if not dst_id in g.node_ids:
                g.add_node(dst_id)
            g.add_edge(src_id, dst_id, label=edge_name, title=str(props))
        elif isinstance(item, PathWrapper):
            for node in item.nodes():
                self.render_pd_item(g, node)
            for edge in item.relationships():
                self.render_pd_item(g, edge)
        elif isinstance(item, list):
            for it in item:
                self.render_pd_item(g, it)
    #
    def create_pyvis_graph(self, result_df):
        # from pyvis.network import Network

        g = Network(
            notebook=True,
            directed=True,
            cdn_resources="in_line",
            height="500px",
            width="100%",
        )
        for _, row in result_df.iterrows():
            for item in row:
                self.render_pd_item(g, item)
        g.repulsion(
            node_distance=100,
            central_gravity=0.2,
            spring_length=200,
            spring_strength=0.05,
            damping=0.09,
        )
        return g
    #
    #Extract keys in response
    def extractKeys(self,response):
        related_entities = list(
                list(response.metadata.values())[0]["kg_rel_map"].keys()
            )
        pattern = r'([^{]+)\{name:\s[^}]+\}'
        extracted_keys = [re.search(pattern, s).group(1).strip() for s in related_entities]
        return extracted_keys
    #
    #Create download link for KG CSV related to response
    def getKGTable(self,extracted_keys): 
        # > RAG Subgraph Query(depth=2)
        nebula_query=f"""
        CREATE TAG IF NOT EXISTS entity(name string);
        CREATE EDGE IF NOT EXISTS relationship(relationship string);
        CREATE TAG INDEX IF NOT EXISTS entity_index ON entity(name(256));
        MATCH (Subject:`entity`)-[Predicate:`relationship`]->(Object:`entity`)
        WHERE id(Subject) in  {extracted_keys}
        RETURN Subject.`entity`.`name`, Predicate.`relationship`, Object.`entity`.`name`
        """

        # Execute the Nebula Graph query
        result_set = self.query_nebulagraph(nebula_query)
        result_set_str = f"{result_set}"
        #
        match = re.search(r"ResultSet\(keys: (\[.*?\]), values: (\[.*?\])\)", result_set_str)
        #
        if match:
            keys_str = match.group(1)
            values_str = match.group(2)

            # Convert string representation to actual lists
            keys = literal_eval(keys_str)
            values = literal_eval(values_str)

            # Create a DataFrame
            df = pd.DataFrame(values, columns=keys)
            #unique_values =  df.drop_duplicates(subset=[selected_column]).reset_index(drop=True)
            unique_values = df.drop_duplicates().reset_index(drop=True)
            # Display the DataFrame in a nice table using Streamlit
            st.write("####  Here is Your Knowledge Graph Table Related to This Response.")
            st.table(unique_values)

            csv = unique_values.to_csv(index=False)
            b64_csv = base64.b64encode(csv.encode()).decode()
            href_csv = f'<a href="data:file/csv;base64,{b64_csv}" download="unique_values.csv">Download Your Knowledge Graph Table</a>'
            st.markdown(href_csv, unsafe_allow_html=True)
    #
    # get and visualize KG related to a response
    def visualizeKG(self, extracted_keys): 
        st.write("#### Here is Your Knowledge Graph Visualization Related to This Response.")
        render_query = (
                f"MATCH (Subject:`entity`)-[Predicate:`relationship`]->(Object:`entity`) \n  WHERE id(Subject) IN {extracted_keys} \n RETURN Subject, Predicate, Object" 
                )
        result = self.query_nebulagraph(render_query)
        result_df = self.result_to_df(result)
        # create pyvis graph
        g = self.create_pyvis_graph(result_df)
        # render with random file name
        graph_html = g.generate_html(f"graph_d.html")

        components.html(graph_html, height=500, scrolling=True)

## Chat with your Images
class ragImage:
    def __init__(self, llavam_tag):
        self.llavam_tag = llavam_tag
        ### LLaVa
        if  self.llavam_tag == 'LLAVA':
            # Azure OpenAI The context window of gpt-4 is 32k tokens
            self.LLAVA_MODEL_URL  = os.getenv('LLAVA_MODEL_URL')
            # st.write(self.LLAVA_MODEL_URL)
            # self.LLAVAL_MODEL_URL = LLAVAL_MODEL_URL
        else:
            raise ValueError('---llavam_tag need to be LLAVA for now---')
    
    #load the image
    def encode_file_to_json(self,file_path):
        # with open(file_path, 'rb') as f:
        #     file_bytes = f.read()
        file_bytes = file_path.read()
        file_base64 = base64.b64encode(file_bytes).decode('utf-8')
        return file_base64
    #
    def imgCaption(self, file_path, prompt):
        # backend = self.LLAVAL_MODEL_URL
        file_name = file_path.name
        file_bytes = self.encode_file_to_json(file_path)
        # prompt_input = "describe the image in details."

        value = json.dumps({"prompt_input": prompt,
                            "file_bytes": file_bytes,
                            "file_name": file_name})

        res = requests.post(self.LLAVA_MODEL_URL+"/llava/stream", data=value)

        output = res.json()
        llm_message = output.get("llm_message")
        return llm_message

## Chat with your Videos
class ragVideo:
    def __init__(self, llvm_tag, sageName):
        self.llavam_tag = llvm_tag
        if sageName is not None:
            self.output_video_folder = './ssData/' + sageName + '/videoData/'
        else:
            self.output_video_folder = './ssData/pgSage/videoData/'
        # self.output_folder = "./mixed_data/"
        self.output_image_folder = "./ssData/" + sageName + "/imageData/"
        self.output_text_folder = "./ssData/" + sageName + "/textData/"
        self.output_audio_folder = "./ssData/" + sageName + "/audioData/"
        # self.output_audio_path = "./mixed_data/output_audio.wav"
        # self.filepath = self.output_video_path + "input_vid.mp4"
        # select multi-modal model
        if llvm_tag == 'GPT4V':
            token = env_credential.get_token(CONGNITIVE_SERVICES).token
            try: 
                monitor_handler=GalileoObserveCallback(project_name=os.getenv("SS_PROJECT_NAME"))
                # LLM GPT-4-VISION
                llvm = AzureChatOpenAI(
                    azure_endpoint=GENAI_PROXY,
                    azure_deployment=DEPLOYMENT_NAME_GPT4V,
                    api_version=GPT4V_VERSION,
                    api_key=token,
                    temperature=0.1,
                    default_headers=HEADERS,
                    callbacks=[monitor_handler],
                )
            except:
                llvm = AzureChatOpenAI(
                    azure_endpoint=GENAI_PROXY,
                    azure_deployment=DEPLOYMENT_NAME_GPT4V,
                    api_version=GPT4V_VERSION,
                    api_key=token,
                    temperature=0.1,
                    default_headers=HEADERS,
                )
            #
            self.llvm = llvm
            embed_model = AzureOpenAIEmbeddings(azure_endpoint=GENAI_PROXY,
                                    deployment=EMBEDDING_DEPLOYMENT_NAME,
                                    api_version=OPEN_API_VERSION,
                                    api_key=token,
                                    default_headers=HEADERS,)
            #
            self.embed_model = embed_model
            # 
            try: 
                monitor_handler=GalileoObserveCallback(project_name=os.getenv("SS_PROJECT_NAME"))
                # GPT-4-VISION
                mmllvm = AzureOpenAIMultiModal(
                    azure_endpoint=GENAI_PROXY,
                    engine=DEPLOYMENT_NAME_GPT4V,
                    api_version=GPT4V_VERSION,
                    max_new_tokens=1200,
                    #
                    api_key=token,
                    temperature=0.1,
                    default_headers=HEADERS,
                    callbacks=[monitor_handler],
                )
            except:
                mmllvm = AzureOpenAIMultiModal(
                    azure_endpoint=GENAI_PROXY,
                    engine=DEPLOYMENT_NAME_GPT4V,
                    api_version=GPT4V_VERSION,
                    max_new_tokens=1200,
                    #
                    api_key=token,
                    temperature=0.1,
                    default_headers=HEADERS,
                )
            #
            self.mmllvm = mmllvm            
    #
    # on Mac, install ffmpeg using homebrew: brew install ffmpeg
    # os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/Cellar/ffmpeg/7.0/bin/ffmpeg"
    def video_to_images(self, fileName):
        """
        Convert a video to a sequence of images and save them to the output folder.

        Parameters:
        filepath (str): The path to the video file.
        output_image_folder (str): The path to the folder to save the images to.

        """
        videoFile = self.output_video_folder + fileName
        fileNameS = Path(fileName).stem
        # clip = VideoFileClip(self.filepath)
        clip = VideoFileClip(videoFile)
        clip.write_images_sequence(
            os.path.join(self.output_image_folder, fileNameS + "_frame%04d.png"), fps=0.2 #configure this for controlling frame rate.
        )
    #
    def video_to_audio(self, fileName):
        """
        Convert a video to audio and save it to the output path.

        Parameters:
        video_path (str): The path to the video file.
        output_audio_path (str): The path to save the audio to.

        """
        videoFile = self.output_video_folder + fileName
        fileNameS = Path(fileName).stem
        # clip = VideoFileClip(self.filepath)
        clip = VideoFileClip(videoFile)
        audio = clip.audio
        output_audio_path = self.output_audio_folder + fileNameS + ".wav"
        audio.write_audiofile(output_audio_path)
    #
    def audio_to_text(self, fileName):
        """
        Convert an audio file to text.

        Parameters:
        audio_path (str): The path to the audio file.

        Returns:
        text (str): The text recognized from the audio.

        """
        # fileNameS = Path(fileName).stem
        # output_audio_path = self.output_audio_folder + fileNameS + ".wav"
        output_audio_path = self.output_audio_folder + fileName
        recognizer = sr.Recognizer()
        audio = sr.AudioFile(output_audio_path)

        with audio as source:
            # Record the audio data
            audio_data = recognizer.record(source)

            try:
                # Recognize the speech
                text = recognizer.recognize_whisper(audio_data)
            except sr.UnknownValueError:
                print("Speech recognition could not understand the audio.")
            except sr.RequestError as e:
                print(f"Could not request results from service; {e}")

        return text
    # setup context for llm and storage
    def setupContext(self, dbName):
        # setup service context and storage context
        service_context = ServiceContext.from_defaults(
            llm=self.llvm, 
            embed_model=self.embed_model, 
            # chunk_size=ragVideo.chunkSize,
            )
        text_store = LanceDBVectorStore(uri=dbName, table_name="text_collection")
        image_store = LanceDBVectorStore(uri=dbName, table_name="image_collection")
        storage_context = StorageContext.from_defaults(
            vector_store=text_store, image_store=image_store,)
        return service_context, storage_context   

    # show images from response
    def plot_images(self, response):
        image_nodes=response.metadata["image_nodes"]
        # need at least one image node
        if len(image_nodes) != 0: 
            st.write('Retrieved Images:')
            for _, scored_img_node in enumerate(image_nodes):
                img_node = scored_img_node.node
                image = None
                if img_node.image_url:
                    img_response = requests.get(img_node.image_url)
                    image = Image.open(BytesIO(img_response.content))
                elif img_node.image_path:
                    image = Image.open(img_node.image_path).convert("RGB")
                else:
                    raise ValueError(
                        "A retrieved image must have image_path or image_url specified."
                    )
                #
                st.image(image, width=500, use_column_width=False)
        
    # remove old files as needed
    def removeOldFiles(self, sageName):
        # remove files after building index
        dirs = ['textData','imageData','audioData','videoData']
        for dir in dirs:
            files = glob.glob("./ssData/"+sageName+"/"+dir+"/*")
            for f in files:
                os.remove(f)
    # processing video
    @st.cache_data(show_spinner=False)
    def processVideo(_self, fileName):
        ### process the video
        fileNameS = Path(fileName).stem
        _self.video_to_images(fileName)
        st.write("---Images extracted from " + fileName + " were saved---")
        _self.video_to_audio(fileName)
        text_data = _self.audio_to_text(fileNameS + ".wav")
        #
        with open(_self.output_text_folder + fileNameS + "_output_text.txt", "w") as fileI:
            fileI.write(text_data)
        fileI.close()
        st.write("---Transcript of audio from " + fileName + " was saved---")
        return None

    # processing audio
    @st.cache_data(show_spinner=False)
    def processAudio(_self, fileName):
        fileNameS = Path(fileName).stem
        text_data = _self.audio_to_text(fileName)
        #
        with open(_self.output_text_folder + fileNameS + "_output_text.txt", "w") as fileI:
            fileI.write(text_data)
        fileI.close()
        # shutil.copy2(fileNameS + "_output_text.txt", './ssData/'+sageName+'/chatDocText/' + fileNameS + '_output_text.txt')
        st.write("---Transcript of audio from " + fileName + " was saved---")
        return None


            



