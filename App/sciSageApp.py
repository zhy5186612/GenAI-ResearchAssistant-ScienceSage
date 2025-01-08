# This script prototype a web App for ScienceSage the AI research assistant 
# last updated 6/20/2024
# Yong Zhang

#
import streamlit as st
import streamlit_ext as ste 
import utils as ut
import os
import json
# langchain
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
# llama_index
from llama_index.embeddings import LangchainEmbedding
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import (VectorStoreIndex,
                        SimpleDirectoryReader,
                        KnowledgeGraphIndex,
                        ServiceContext,)
from llama_index.storage.storage_context import StorageContext
from llama_index.graph_stores import NebulaGraphStore
#
from llama_index import load_index_from_storage
from llama_index import get_response_synthesizer
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.retrievers import VectorIndexRetriever, KGTableRetriever
# use gpt4v
from llama_index.indices.multi_modal.base import MultiModalVectorStoreIndex
from llama_index.vector_stores import LanceDBVectorStore
from llama_index.schema import ImageNode
from matplotlib import pyplot as plt
import shutil
#
import glob
from pathlib import Path
#
import prompts as pt
#
from dotenv import load_dotenv
load_dotenv()

# import weaviate 
from llama_index.vector_stores.weaviate import WeaviateVectorStore
#
if "weaviate_client" not in st.session_state:
    st.session_state.weaviate_client = ut.get_weaviate_client()
weaviate_client = st.session_state.weaviate_client

def quickStart():
    st.write('<p style="font-size:40px; color:blue;">Procedure to Use ScienceSage</p>', unsafe_allow_html=True)

    #
    st.markdown(
        """
        ## Introduction

        Last updated on 1/7/2025

        **ScienceSage** is a GenAI powered MVP Web Application. It can serve as a research assisstant to disrupt the speed and scope of
        any research work. It has three main functionalities. First, the **'Generate Research Report'** tab can generate a comprehensive research 
        report based a user's research question. **ScienceSage** first decomposes the reserach question into several relevant queries and use them to query 
        internet. The report is generated based on **the latest contents** from generic internet search or specific scientific paper 
        database search (e.g., Arxiv database). Second, the **'Chat With Your Documents'** tab allows a user to upload his own textual documents (e.g., 
        journal articles) and then chat with a chatbot to get quick and concise answers from these documents. Third, the **'Chat With Anything'** tab allow a user to 
        upload **multimodal** data to extract information and insights. **Multimodal data includes images, videos, audios and texts**. Last but not least, 
        the **knowledge/inforamtion extracted from the generated report, uploaded documents, information extracted from videos and audios are saved in the corresponding Knowledge Base** (e.g. PgSage). 
        Next time you come back, just load the same Knowledge Base in "Chat With Your Documents" and ask your questions!

        ## Quick Start
        You will need to (1) setup your own vector database,graph database and multimodal database, (2) to modify the utils.py and sciSageApp.py to your own files and APIs of LLM/LVM . 
        - Install required packages:
        pip install -r requirements.txt
        - Run streamlit app locally:
        streamlit run sciSageApp.py
        - Deploy the app to a server( or cloud) using Docker:
        Modify the Dockerfile and docker-compose.yml files to your own settings. then on your server run:
        docker-compose build
        docker-compose up -d

        ## Unique Advantages 
        - (1) **Latest Information:** Knowledge from ChatPG/ChatGPT/BingChat/Gemini generally has a cutoff time for obtaining the training data. If some information comes out after 
          the cutoff time of traiming these LLMs, they will **NOT** be used to generate a answer. **ScienceSage** is a live agent, it will always search the internet 
          or scientific databases to get the latest information when generating a report.
        - (2) **Structural & Comprehensive Report:** We designed specific prompts to instruct the LLM to generate a structural and comprehensive research report. 
            It also list the top relevant references. Users can customize their instructions based on several well designed built-in prompt templates.
        - (3) **Customizable:** Users can customize the instructions to generate a report based on their specific needs.
        - (4) **Chat with Documents:** Users can get the answer and also **references** and/or **Knowledge Graph** realated to the answer.
        - (5) **Chat with Anything:** Users can extract information from multimodal data such as videos, audios, images and texts.  
        - (6) **Data Security:** Mixtral_8X7B was internally deployed LLM on a P&G server. We have total control on the model and generated contents. 
            These generated contents and user's documents never reach to any external parties. 
        - (7) **Cost Avoidance:** Mixtral_8X7B is opensource and free to use. We don't have to pay based on token usage. 
        - (8) **Avoid Filtering by Microsoft**: Microsoft will filter out and block your ansers/response if it thinks your input or output is offensive. 
            As a non-consumer facing and internal research tool, we don't have any filtering policy on Mixtral_8X7B. Users are guaranteed to have a valid response.  
        - (9) **Knowledge/Information are saved in the corresponding Knowledge Base**. You can **use existing Knowledge Base** or **create a new KB** from scratch. 


        ## Technical Features
        - (1) **ScienceSage** is powered by the best performing opensource Large Languange Model (LLM) [Mixtral_8X7B](https://mistral.ai/news/mixtral-of-experts/) 
            and the [huggingface embeding model all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2).
             We also enabled Azure ChatGPT in the backend as a backup model. We are working to seek approval of a related proposal by Turing Team.   
        - (2) A [Langchain Agent](https://python.langchain.com/docs/modules/agents/) was implemented to perform the tasks of **'Generate Research Report'**. 
             The Agent decomposes a research question into 3 relevant queries. It then uses the generated queries to search and scrape internet or to retrieve papers from Arxchive databases. 
             Finally, the Agent generates a comprehensive report based on top relevant websites or scientific papers.    
        - (3) A [Retrieval Augmented Generation (RAG)](https://docs.llamaindex.ai/en/stable/getting_started/concepts.html) was implemented to 
             perform tasks of **'Chat With Your Documents'**. Users can choose to use the RAG based on Vector Index, Knowledge Graph Index or both of them.
        - (4) Currently, **ScienceSage** supports two methods to get latest inforamtion: (1) generic internet search; and (2) search scientific papers in Arxiv databases. 
             We will add more scientific databases in the backend in future. 
        - (5) We use **multimodal model GPT4-Vision** to extract information and insights from multimodal data. We plan to add **Gemini-Pro-1.5** and open source multimodal model **LLAVA** later.
        - (6) We also use **Galileo** (on P&G server) to monitor backend and improve different functionalities based on the data. 
        - (7) We use **Weaviate** and **Nebula Graph** to store the Vector Embedding and Graph Embedding, respectively. **LanceDB** is under test to store multimodal embeding and index.
        
        ## Team members & Contributors
        - Members: Yong Zhang, Kelly Anderson, Eric (Herrison) Gyamfi (@UC Digital Accelorator)
        - Contributors: Matt Barker, Sasha Roberts, Ming Chen, George Gabone 

        ## Contributions
        - We welcome contributions through seperate branch or pull request.
        - Please create an issues if you have questions.
        """
    )
#
def getReport():
    st.write('<p style="font-size:40px; color:blue;">Use GenAI to Generate a Research Report</p>', unsafe_allow_html=True)
    st.write('<p style="font-size:30px; color:purple;">Note 1: Most of your issues can be resolved by refresh/reload the Web App. </p>', unsafe_allow_html=True)
    st.write('<p style="font-size:30px; color:purple;">Note 2: You can download the report. It is also saved in the corresponding Knowledge Base (e.g. PgSage). \
                Next time you come back, just load the same Knowledge Base in "Chat With Your Documents" and ask your questions! </p>', unsafe_allow_html=True)

    # Users can create a new sage name from Chat With Your Documents tab or Chat With Anything tab
    sageNames = ut.getSageNames()
    # sageNames.remove('Create a new knowlege base')
    sageName= st.sidebar.selectbox("Select a Knowledge Base to Save Report", sageNames)
    if sageName == 'Create a new knowlege base':
        sageName = st.text_input("Please type in your unique Knowledge Base Name (begin with an uppercase letter):", 'PgSage')
        try:
            ut.setupSageDir(sageName)
            st.write('<p style="font-size:30px; color:purple;">---Now, please refresh the Web App now and select your knowlege base from "Select a Knowledge Base". </p>', unsafe_allow_html=True)
        except:
            st.error('---Knowledge Base Name already exists, please type in another one---', icon="🚨")  
    else:
        # 'what are the pros and cons of using sustainable surfactants in detergents?'
        question = st.text_area("Please type in your research question:", 'what are the most promising new treatment of bacterial vaginosis?', height=40)
        #
        # dataSource = st.sidebar.selectbox('Select Data Source:',['internet','localFile'])
        dataSource = 'internet'
        if dataSource == 'internet':
            searchMethod = st.sidebar.selectbox('Select Search Method:',['searchInternet','searchArxiv'])
            numQuery = st.sidebar.slider('Number of Queries to Decompose Your Question:', 2, 6, 3, 1)
            numRecords = st.sidebar.slider('Top N Records Per Query:', 2, 20, 3, 1)
        elif dataSource == 'localFile':
            searchMethod = None
            numQuery = None
            numRecords = None
            st.write('<p style="font-size:40px; color:red;">This module is under development, come back and check it again later!</p>', unsafe_allow_html=True)
        #
        reportTemplate = st.sidebar.selectbox('Select a Report Template:',['Generic Research Report'])
        if reportTemplate=='Generic Research Report':
            st.write('<p style="font-size:30px; color:green;">Use Next Three Template Boxes from Generic Research Report to Customize Your Instructions</p>', unsafe_allow_html=True)
            promptR = pt.gsr_prompt
            prompt = ut.loadPrompt(promptR)        
        
        #
        llm_tag = st.sidebar.selectbox("Select a LLM:",['AzureChatGPT', 'Mixtral_8X7B'])
        # llm_tag = 'Mixtral_8X7B'

        if st.sidebar.button('generateReport'):
            st.write('If you use a large number of queries and records per query, it may take a while to get your report.')
            st.write('<p style="font-size:40px; color:green;">---Here is Your Research Report---</p>', unsafe_allow_html=True)
            sciSage = ut.scienceSage(dataSource, searchMethod, llm_tag, prompt, question, numQuery, numRecords)
            report = sciSage.generateReport()
            st.write(report)

            ## save to SageName and enable pdf file download
            sciSage.save_download_report(report, sageName)
       
# get vector index
def getVectorIndex(_service_context, sageName, docs):
    # set up directory
    sageDir = './ssData/' + sageName
    #
    # vector_store = WeaviateVectorStore(weaviate_client = weaviate_client, index_name=sageName, text_key="text")
    # storage_contextVI = StorageContext.from_defaults(vector_store=vector_store)
    # # local version
    # if os.path.exists(sageDir + "/storageVI"):
    # try:
    # # weaviate V4
    # if weaviate_client.collections.exists(sageName):
    # weaviate V3
    if weaviate_client.schema.exists(sageName):
        # storage_contextVI = StorageContext.from_defaults(persist_dir = sageDir + "/storageVI")
        # vector_index= load_index_from_storage(storage_context=storage_contextVI, service_context=_service_context)
        vector_store = WeaviateVectorStore(weaviate_client = weaviate_client, index_name=sageName, text_key="text")
        # storage_contextVI = StorageContext.from_defaults(vector_store=vector_store)
        # vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store, service_context=_service_context)
        storage_contextVI = StorageContext.from_defaults(vector_store=vector_store, persist_dir = sageDir + "/storageVI")
        # vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store, service_context=_service_context, storage_context=storage_contextVI)
        vector_index= load_index_from_storage(storage_context=storage_contextVI, service_context=_service_context)
        st.write('---Vector Index is loaded from storage---')
        if docs is not None:
            refreshDocs = vector_index.refresh_ref_docs(docs)
            # st.write(refreshDocs)
            if sum(refreshDocs) > 0:
                st.write('---Vector Index is refreshed with new documents---')
                # storage_contextVI.flush()
                vector_index.storage_context.persist(persist_dir = sageDir + "/storageVI")
    else:
    # except:
        if docs is not None:
            vector_store = WeaviateVectorStore(weaviate_client = weaviate_client, index_name=sageName, text_key="text")
            # storage_contextVI = StorageContext.from_defaults(vector_store=vector_store)            
            # vector_index = VectorStoreIndex.from_documents(docs, service_context=_service_context, storage_context=storage_contextVI)
            storage_contextVI = StorageContext.from_defaults(vector_store=vector_store)            
            vector_index = VectorStoreIndex.from_documents(docs, service_context=_service_context, storage_context=storage_contextVI)
            vector_index.storage_context.persist(persist_dir = sageDir + "/storageVI")
            st.write('---Vector Index is built and saved to storage---')
        else:
            # vector_index = None
            st.error('---You need upload at least one document to start to build a new ScienceSage knowledge base---', icon="🚨")
    # st.write(vector_index.docstore.get_document)
    return vector_index

# get knowledge graph index
def getKGIndex(_service_context, sageName, docs):
    sageDir = './ssData/' + sageName
    edge_types, rel_prop_names = ["relationship"], ["relationship"]  # default, could be omit if create from an empty kg
    tags = ["entity"]
    #      
    if os.path.exists(sageDir + "/storageKG"):
    # try:
        graph_store = NebulaGraphStore(
                space_name=sageName,
                edge_types=edge_types,
                rel_prop_names=rel_prop_names,
                tags=tags,)
        #          
        storage_contextKG = StorageContext.from_defaults(persist_dir = sageDir + "/storageKG", graph_store=graph_store)

        kg_index = load_index_from_storage(
                storage_context=storage_contextKG,
                service_context=_service_context,
                max_triplets_per_chunk=10,
                space_name=sageName,
                edge_types=edge_types,
                rel_prop_names=rel_prop_names,
                tags=tags,
                verbose=True,
            )
        st.write('---Knowledge Graph Index is loaded from storage---')
        #
        if docs is not None:
            refreshDocs = kg_index.refresh_ref_docs(docs)
            # st.write(refreshDocs)
            if sum(refreshDocs) > 0:
                st.write('---Knowledge Graph Index is refreshed with new documents---')
                # storage_contextKG.flush()
                kg_index.storage_context.persist(persist_dir=sageDir + '/storageKG')
    else:
    # except:
        
        if docs is not None:
            #
            ut.create_kg_space(sageName)
            graph_store = NebulaGraphStore(
                    space_name=sageName,
                    edge_types=edge_types,
                    rel_prop_names=rel_prop_names,
                    tags=tags,)
            #
            storage_contextKG = StorageContext.from_defaults(graph_store=graph_store)
            #
            kg_index = KnowledgeGraphIndex.from_documents(docs,
                                storage_context=storage_contextKG,
                                service_context=_service_context,
                                max_triplets_per_chunk=10,
                                space_name=sageName,
                                edge_types=edge_types,
                                rel_prop_names=rel_prop_names,
                                tags=tags,
                                include_embeddings=True,
                                )
            kg_index.storage_context.persist(persist_dir=sageDir + '/storageKG')
            st.write('---Knowledge Graph Index is built and saved to storage---')
        else: 
            st.error('---You need upload at least one document to start to build a new ScienceSage knowledge base---',icon="🚨")
    #
    return kg_index  

# get indices based on usage case
def getIndice(_service_context, sageName, usageCase):
    #
    currentFiles = os.listdir('./ssData/' + sageName + '/chatDocText')
    if '.DS_Store' in currentFiles:
        currentFiles.remove('.DS_Store')
    #
    if usageCase=='Use Vector':
        docstore = './ssData/' + sageName + '/storageVI/docstore.json'
    elif usageCase == 'Use Knowledge Graph':
        docstore = './ssData/' + sageName + '/storageKG/docstore.json'
    try:
        with open(docstore) as f:
            docInfo = json.load(f)
        #
        if usageCase=='Use Vector':
            fileIDpaths = list(docInfo['docstore/metadata'].keys())
        elif usageCase == 'Use Knowledge Graph':
            fileIDpaths = list(docInfo['docstore/ref_doc_info'].keys())
        #
        indexedFiles = ut.extract_filenames(fileIDpaths)
        st.write('---Files Already in Selected Knowledge Base---')
        st.write(indexedFiles)
    except:
        indexedFiles = []

    # files not indexed yet
    filePathsNI = ['./ssData/' + sageName + '/chatDocText/' + file for file in list(set(currentFiles) - set(indexedFiles))]
    # st.write(filePathsNI)

    # read new documents
    if len(filePathsNI)>0: 
        # reader = SimpleDirectoryReader(input_dir=  './ssData/' + sageName + '/chatDocText', recursive=True, filename_as_id=True)
        reader = SimpleDirectoryReader(input_dir=  './ssData/' + sageName + '/chatDocText', input_files=filePathsNI,
                                       recursive=True, filename_as_id=True)
        docs = reader.load_data()
        st.write('---Updating Index With New Files Below---')
        st.write(filePathsNI)
        # for doc_num in range(len(docs)):
        #     st.write(f"document-{doc_num} --> {docs[doc_num].id_}")
    else:
        docs = None
        st.write('---No New Documents Uploaded---')
    #
    if usageCase=='Use Vector':
        vector_index = getVectorIndex(_service_context, sageName, docs)
        kg_index = None
    elif usageCase == 'Use Knowledge Graph':
        kg_index = getKGIndex(_service_context, sageName, docs)
        vector_index = None
    elif usageCase == 'Use Knowledge Graph & Vector':
        vector_index = getVectorIndex(_service_context, sageName, docs)
        kg_index = getKGIndex(_service_context, sageName, docs)
    #
    return vector_index, kg_index

# query vector index
def VectorS(vectorIndex,rag):
    st.write('<p style="font-size:40px; color:green;">Chat with your Research Assistant</p>', unsafe_allow_html=True)
    st.success('Proceed to entering your question!', icon='👉')
    if "messages" not in st.session_state.keys(): # Initialize the chat messages history
        st.session_state.messages = [
        {"role": "assistant", "content": "Ask me questions now!"}
        ]

    st.session_state.vdChat = vectorIndex.as_query_engine(llm=rag.llm)
    #
    if prompt := st.chat_input("Ask your question"): # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
    #
    for message in st.session_state.messages: # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Display or clear chat messages
    st.sidebar.button('Clear Chat History', on_click=ut.clear_chat_history)

    # If last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking with Vector Index"):
                # response = st.session_state.chat_engine.chat(prompt)
                response = st.session_state.vdChat.query(prompt)
                st.write(response.response)
                # st.write(response)
                message = {"role": "assistant", "content": response.response}
                # message = {"role": "assistant", "content": response}
                st.session_state.messages.append(message) # Add response to message history
                rag.outputReferences(response)

# query knowledge graph index                
def KnowledgeI(kgIndex, rag):
    st.write('<p style="font-size:40px; color:navy;">Chat with your Research Assistant</p>', unsafe_allow_html=True)
    st.success('Proceed to entering your question!', icon='👉')

    

    if "messages" not in st.session_state.keys(): # Initialize the chat messages history
        st.session_state.messages = [
        {"role": "assistant", "content": "Ask me questions now!"}
        ]
    ## 
    # if 'kgChat' not in st.session_state:
    st.session_state.kgChat = kgIndex.as_query_engine(
        # setting to false uses the raw triplets instead of adding the text from the corresponding nodes
        include_text=False,
        retriever_mode="keyword",
        response_mode="tree_summarize",
        explore_global_knowledge=True,
        verbose=True,     
        llm=rag.llm,)
    #
    if promptK := st.chat_input("Ask your question"): # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": promptK})

    for message in st.session_state.messages: # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Display or clear chat messages
    st.sidebar.button('Clear Chat History', on_click=ut.clear_chat_history)

    # If last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking with Knowledge Graph Index"):
                response = st.session_state.kgChat.query(promptK)
                st.write(response.response)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message) # Add response to message history
                #
                ### get tables and visuals for KG related to this response            
                try:
                    extracted_keys = rag.extractKeys(response)
                    rag.getKGTable(extracted_keys)
                    rag.visualizeKG(extracted_keys)
                except:
                    st.write('---I can not find any related entities in KG related to this response---')

# query both vector index and knowledge graph index
def BothVandK(vectorIndex, kgIndex, service_context, rag):
    st.write('<p style="font-size:40px; color:fuchsia;">Chat with your Research Assistant</p>', unsafe_allow_html=True)
    st.success('Proceed to entering your question!', icon='👉')
    #
    vector_retriever = VectorIndexRetriever(index=vectorIndex)
    # Use key word to retrieve from KG
    kg_retriever = KGTableRetriever(index=kgIndex, retriever_mode="keyword", include_text=False)
    # Use hybrid embedding to retrieve from KG
    # kg_retriever = KGTableRetriever(index=kgIndex, retriever_mode="hybrid", include_text=True)
    custom_retriever = ut.CustomRetriever(vector_retriever, kg_retriever)
    # create response synthesizer
    response_synthesizer = get_response_synthesizer(
        service_context=service_context,
        response_mode="tree_summarize",)
    #
    # if 'customChat' not in st.session_state:
    st.session_state.customChat = RetrieverQueryEngine(
        retriever=custom_retriever,
        response_synthesizer=response_synthesizer,)

    #
    if "messages" not in st.session_state.keys(): # Initialize the chat messages history
        st.session_state.messages = [
        {"role": "assistant", "content": "Ask me questions now!"}
        ]
    
    #
    if prompt := st.chat_input("Ask your question"): # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages: # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Display or clear chat messages
    st.sidebar.button('Clear Chat History', on_click=ut.clear_chat_history)
    # If last message is not from assistant, generate a new response
    # https://docs.llamaindex.ai/en/stable/examples/query_engine/CustomRetrievers.html
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking with both Knowledge Graph Index and Vector Index"):
                response = st.session_state.customChat.query(prompt)
                st.write(response.response)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message) # Add response to message history  

                ## get tables and visuals for KG related to this response
                try:
                    extracted_keys = rag.extractKeys(response)
                    rag.getKGTable(extracted_keys)
                    rag.visualizeKG(extracted_keys)
                    rag.outputReferences(response)
                except:
                    st.write('---I can not find any related entities in KG related to this response---')
                    rag.outputReferences(response)                

# create RAG
def getRag():
    st.write('<p style="font-size:40px; color:blue;">Use GenAI to Chat with Your Documents</p>', unsafe_allow_html=True)
    # select a scienceSage assistant
    sageNames = ut.getSageNames()
    sageName= st.sidebar.selectbox("Select a Knowledge Base", sageNames)
    # select usage scenario
    usageCase = st.sidebar.selectbox("Select a Usage Scenario:", ['Use Vector', 'Use Knowledge Graph & Vector', 'Use Knowledge Graph']) 
    # select a LLM
    llm_tag = st.sidebar.selectbox("Select a LLM:",['AzureChatGPT', 'Mixtral_8X7B'])
    #
    # set up service context
    rag = ut.rag(llm_tag)
    service_context = ServiceContext.from_defaults(
    llm=rag.llm, 
    embed_model=rag.embed_model,
    # chunk_size=rag.chunkSize
    )    
    #
    if sageName == 'Create a new knowlege base':
        sageName = st.text_input("Please type in your unique Knowledge Base Name (begin with an uppercase letter):", 'PgSage')
        try:
            ut.setupSageDir(sageName)
            st.write('<p style="font-size:30px; color:purple;">---Now, please refresh the Web App now and select your knowlege base from "Select a Knowledge Base". </p>', unsafe_allow_html=True)
        except:
            st.error('---Knowledge Base Name already exists, please type in another one---', icon="🚨")

        
    #
    else:
        # st.write('<p style="font-size:30px; color:purple;">Note 1: if you want to add new or additional files after initial uploading, please refresh the tab/App to index them. </p>', unsafe_allow_html=True)
        st.write('<p style="font-size:30px; color:purple;">Note 1: Most of your issues can be resolved by refresh/reload the Web App. </p>', unsafe_allow_html=True)
        st.write('<p style="font-size:30px; color:purple;">Note 2: Knowledge/Information extracted in your documents is saved in the corresponding Knowledge Base (e.g. PgSage). \
                 Next time you come back, just load the same Knowledge Base in "Chat With Your Documents" and ask your questions! </p>', unsafe_allow_html=True)

        # upload files or not
        st.write('<p style="font-size:40px; color:fuchsia;">Do you have new textual documents to upload?</p>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader("Choose Your Input Files", type=['csv','txt','docx','xlsx','pdf'], accept_multiple_files=True)
        # fileDir = 'chatDocText'
        # sageName = 'pgSage'
        currentFiles = os.listdir('./ssData/' + sageName + '/chatDocText')
        if (len(uploaded_files) > 0): # and st.sidebar.button('runChatbot'):
            for _, file in enumerate(uploaded_files):
                ut.save_uploadedfile(file, './ssData/' + sageName + '/chatDocText')
            #
            st.success("---loaded files are ready for indexing---") 
            newFiles = os.listdir('./ssData/' + sageName + '/chatDocText')
            # delete session_state conaining index and query engine if there are new files uploaded
            if (currentFiles != newFiles):
                st.write('---New Files Uploaded---')
                if 'vectorIndex' in st.session_state:
                    del st.session_state.vectorIndex
                    st.write('---session state vectorIndex deleleted---')
                # if 'vdChat' in st.session_state:
                #     del st.session_state.vdChat
                #     st.write('---session state vdChat deleleted---')
                if 'kgIndex' in st.session_state:
                    del st.session_state.kgIndex
                    st.write('---session state kgIndex deleleted---') 
        
        # update index if switch to a new sageName
        if 'sageName' not in st.session_state:
            # first sageName in a user session, no need to delete index
            st.session_state.sageName = sageName
        elif sageName != st.session_state.sageName:
            if 'vectorIndex' in st.session_state:
                del st.session_state.vectorIndex
                st.write('---session state kgIndex deleleted---')
            if 'kgIndex' in st.session_state:
                del st.session_state.kgIndex
                st.write('---session state kgIndex deleleted---')             
            st.session_state.sageName = sageName
        
        # clear chat history if switch to a new usage case
        if 'chatSenario'  not in st.session_state:
            st.session_state.chatSenario = sageName + ': ' + usageCase
        elif st.session_state.chatSenario != sageName + ': ' + usageCase:
            ut.clear_chat_history()          
            st.session_state.chatSenario = sageName + ': ' + usageCase          
        
        # get index and chat with documents
        if usageCase == 'Use Vector':# and st.sidebar.button('runQ&A'):
            #
            with st.spinner(text="Loading or building index – hang tight! This may take a while."):
                # cache the index in session_state if it is first run
                if "vectorIndex" not in st.session_state:
                    st.write('---Vector Index is not in session_state---')
                    st.session_state.vectorIndex, _ = getIndice(service_context, sageName, usageCase)                             
            VectorS(st.session_state.vectorIndex, rag)
        elif usageCase == 'Use Knowledge Graph':# and st.sidebar.button('runQ&A'):
            #
            with st.spinner(text="Loading or building index – hang tight! This may take a while."):
                # cache the index in session_state if it is first run
                if "kgIndex" not in st.session_state:
                    st.write('---Knowledge Graph Index is not in session_state---')
                    _, st.session_state.kgIndex = getIndice(service_context, sageName, usageCase)               
            KnowledgeI(st.session_state.kgIndex, rag)
        elif usageCase == 'Use Knowledge Graph & Vector':# and st.sidebar.button('runQ&A'):
            with st.spinner(text="Loading or building index – hang tight! This may take a while."):
                # cache the index in session_state if it is first run
                if "kgIndex" not in st.session_state:
                    _, st.session_state.kgIndex = getIndice(service_context, sageName, usageCase)  
                if "vectorIndex" not in st.session_state:
                    st.session_state.vectorIndex, _ = getIndice(service_context, sageName, usageCase)                     
            BothVandK(st.session_state.vectorIndex,st.session_state.kgIndex,service_context,rag)     

# retreive information from Video
def retrieve(retriever_engine, query_prompt):
    retrieval_results = retriever_engine.retrieve(query_prompt)

    retrieved_image = []
    retrieved_text = []
    st.write('<p style="font-size:25px; color:green;">Retrieved Text Chunks/Nodes</p>', unsafe_allow_html=True)
    for res_node in retrieval_results:
        if isinstance(res_node.node, ImageNode):
            retrieved_image.append(res_node.node.metadata["file_path"])
        else:
            # display_source_node(res_node, source_length=200)
            st.write(res_node)
            retrieved_text.append(res_node.text)

    return retrieved_image, retrieved_text

# get multi-modal index
def getMultiModalIndex(ragVideo, service_context, storage_context):
    # Create the MultiModal index
    st.write('<p style="font-size:30px; color:blue;">Building multimodal index......</p>', unsafe_allow_html=True)

    ## method 1
    # documents = SimpleDirectoryReader(ragVideo.output_folder).load_data()
    # index = MultiModalVectorStoreIndex.from_documents(
    #     documents,
    #     storage_context=storage_context,
    #     service_context=service_context,
    # )

    ## method 2    
    # len(os.listdir(dir)) == 1 if dir is empty, because dir itself is counted as a file
    # if len(os.listdir(ragVideo.output_image_folder)) == 1: 
    if len(glob.glob(os.path.join(ragVideo.output_image_folder, '*')))==0:
        # st.write(str(len(os.listdir(ragVideo.output_image_folder))))
        text_documents = SimpleDirectoryReader(ragVideo.output_text_folder).load_data() 
        index = MultiModalVectorStoreIndex.from_documents(text_documents,service_context=service_context,)
    # elif len(os.listdir(ragVideo.output_text_folder)) == 1: 
    elif len(glob.glob(os.path.join(ragVideo.output_text_folder, '*')))==0:
        # st.write(str(len(os.listdir(ragVideo.output_text_folder))))
        image_documents = SimpleDirectoryReader(ragVideo.output_image_folder).load_data()
        index = MultiModalVectorStoreIndex.from_documents(image_documents,service_context=service_context,)
    else:
        # st.write(str(len(os.listdir(ragVideo.output_image_folder))))
        # st.write(str(len(os.listdir(ragVideo.output_text_folder))))
        image_documents = SimpleDirectoryReader(ragVideo.output_image_folder).load_data()
        text_documents = SimpleDirectoryReader(ragVideo.output_text_folder).load_data() 
        index = MultiModalVectorStoreIndex.from_documents(
            image_documents + text_documents,
            service_context=service_context,
            # # commment out to make it session based for now 
            # storage_context=storage_context,
            )
        
    st.write("---Index building finished---")
    return index

# query multimodal index
def mmIndexQuery(mmIndex, ragVideo):
    # setup multimodal query engine
    azure_openai_mm_llm = ragVideo.mmllvm
    qa_tmpl_str = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, "
        "answer the query.\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    qa_tmpl = PromptTemplate.from_template(qa_tmpl_str)

    #
    st.write('<p style="font-size:40px; color:green;">Chat with your Research Assistant based on multimodal data </p>', unsafe_allow_html=True)
    st.success('Proceed to entering your question!', icon='👉')
    if "messages" not in st.session_state.keys(): # Initialize the chat messages history
        st.session_state.messages = [
        {"role": "assistant", "content": "Ask me questions about your multimodal data!"}
        ]


    # Get latest GenAI token through azure_openai_mm_llm and get latest index
    # if st.session_state.mmChat is None:
    st.session_state.mmChat = mmIndex.as_query_engine(
                                multi_modal_llm=azure_openai_mm_llm, text_qa_template=qa_tmpl,)
    #
    if prompt := st.chat_input("Ask your question"): # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
    #
    for message in st.session_state.messages: # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Display or clear chat messages
    st.sidebar.button('Clear Chat History', on_click=ut.clear_chat_history)

    # If last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking with Multimodal Index"):
                # response = st.session_state.chat_engine.chat(prompt)
                response = st.session_state.mmChat.query(prompt)
                st.write(response.response)
                st.write(response.get_formatted_sources()) #response.get_formatted_sources()
                ragVideo.plot_images(response)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message) # Add response to message history
                # rag.outputReferences(response)

# chat with video
def getMmRAG():
    st.write('<p style="font-size:40px; color:blue;">Use GenAI to Chat with Multimodal Data</p>', unsafe_allow_html=True)
    # st.write('<p style="font-size:30px; color:purple;">Note 1: if you want to add new or additional files after initial uploading, please refresh the tab/App to index them. </p>', unsafe_allow_html=True)
    selection= st.radio("Choose an Action", ['Test With Demo Multimodal Data', 'Upload Your Multimodal Data'])

    if selection == 'Test With Demo Multimodal Data':
        llvm_tag = st.sidebar.selectbox("Select a Multimodal Model:",['GPT4V'])
        sageName = 'demoSage' #'demoSage' #
        dbName = './ssData/' + sageName + '/'+ 'lanceDB' #hcRoutine1
        ragVideo = ut.ragVideo(llvm_tag, sageName)
         # setup service context and storage context
        service_context, storage_context = ragVideo.setupContext(dbName)

        # Demo Video
        st.write('<p style="font-size:20px; color:green;">Demo Video</p>', unsafe_allow_html=True)
        _,col2,_=st.columns([1,1,1])
        with col2:
            st.video('./ssData/demoSage/videoData/your_videofile1.mp4')
        st.write('<p style="font-size:20px; color:green;">Demo Audio</p>', unsafe_allow_html=True)
        # Demo Audio
        _,col2,_=st.columns([1,1,1])
        with col2:
            st.audio('./ssData/demoSage/audioData/your_audiofile1.wav')
        st.write('<p style="font-size:20px; color:green;">Demo Images</p>', unsafe_allow_html=True)
        # Demo Image
        col1,col2,col3=st.columns([1,1,1])
        with col1:
            st.image('./ssData/demoSage/imageData/your_imagefile1.png')
        with col2:
            st.image('./ssData/demoSage/imageData/your_imagefile2.png')
        with col3:
            st.image('./ssData/demoSage/imageData/your_imagefile3.png')
        # Demo Text
        st.write('<p style="font-size:20px; color:green;">Demo Textual Data</p>', unsafe_allow_html=True)
        st.markdown("Find the demo pdf file, word document, and other files in [Demo Files](%s)" % 'https://url_to_demo_textual_files')

        # Example Chats
        st.write('<p style="font-size:20px; color:green;">Example Chats: (1) tell me something about hair care. (2) what products and tools are used in the hair care routine? (3) how many steps are involved in the hair care routine?</p>', unsafe_allow_html=True)
        # clear chat history
        if 'demo_run' not in st.session_state:
            ut.clear_chat_history()
            # clear the sessions state so that users not use the earlier index
            # st.session_state.mmChat = None
            st.session_state.demo_run = True
        # load multimodal index
        if "mmIndexD" not in st.session_state:
            st.session_state.mmIndexD = MultiModalVectorStoreIndex.from_vector_store(vector_store=LanceDBVectorStore(uri=dbName, table_name="text_collection"), 
                                                                                     service_context=service_context, 
                                                                                     image_vector_store=LanceDBVectorStore(uri=dbName, table_name="image_collection"))
        # chat with multimodal index
        mmIndexQuery(st.session_state.mmIndexD, ragVideo)
    #
    elif selection == 'Upload Your Multimodal Data':

        #
        st.write('<p style="font-size:30px; color:purple;">Note 1: Most of your issues can be resolved by refresh/reload the Web App. </p>', unsafe_allow_html=True)
        st.write('<p style="font-size:30px; color:purple;">Note 2: Knowledge/Information extracted in your multimodal data is saved in the corresponding Knowledge Base (e.g. PgSage). \
                 Next time you come back, just load the same Knowledge Base in "Chat With Your Documents" and ask your questions! </p>', unsafe_allow_html=True)
        sageNames = ut.getSageNames()
        sageName= st.sidebar.selectbox("Select a Knowledge Base", sageNames)
        #
        llvm_tag = st.sidebar.selectbox("Select a Multimodal Model:",['GPT4V'])        
        # sageName = 'demoSage'
        if sageName == 'Create a new knowlege base':
            sageName = st.text_input("Please type in your unique Knowledge Base Name (begin with an uppercase letter):", 'PgSage')
            try:
                ut.setupSageDir(sageName)
                st.write('<p style="font-size:30px; color:purple;">---Now, please refresh the Web App now and select your knowlege base from "Select a Knowledge Base". </p>', unsafe_allow_html=True)
            except:
                st.error('---Knowledge Base Name already exists, please type in another one---', icon="🚨")
        else:
        #
            st.write('<p style="font-size:40px; color:fuchsia;">Upload Your Multimodal Data</p>', unsafe_allow_html=True)
            uploaded_files = st.file_uploader("Upload your videos/audios/images/texts", type=['mp4','mpeg4','wav','mp3','png','jpg','jpeg','csv','txt','docx','xlsx','pdf'], accept_multiple_files=True)
            # st.write(uploaded_files)
            # sageName = 'pgSage' #'demoSage' #
            dbName = './ssData/' + sageName + '/'+ 'lanceDB' #hcRoutine1
            #
            ragVideo = ut.ragVideo(llvm_tag, sageName)

            # setup service context and storage context
            service_context, storage_context = ragVideo.setupContext(dbName)
            #
            if (len(uploaded_files) > 0): 
                st.write('<p style="font-size:30px; color:blue;">Processing uploaded files, only show first TWO videos/audios/images if applicable......</p>', unsafe_allow_html=True)
                
                # remove data from the early run
                # if 'chatAnything_run' not in st.session_state:
                if sageName not in st.session_state:
                    # remove old data files
                    ragVideo.removeOldFiles(sageName=sageName)
                    # if this is a new session, clear caching in case users used same video files again
                    ragVideo.processVideo.clear()
                    ragVideo.processAudio.clear()
                    # # clear the session state so that users not use the earlier index
                    # st.session_state.mmChat = None
                    # clear chat history
                    ut.clear_chat_history()
                    # set current_run to True to avoid removing files of current run
                    # st.session_state.chatAnything_run = True
                    st.session_state[sageName] = True
                #
                currentFiles = os.listdir('./ssData/' + sageName + '/textData') + os.listdir('./ssData/' + sageName + '/imageData')
                # process uploaded files
                vcount = 0
                acount = 0
                icount = 0
                for _, file in enumerate(uploaded_files):
                    fileName = file.name
                    # st.write(file.type)
                    # st.write(fileName)
                    # process video
                    fileNameS = Path(fileName).stem
                    if file.type in ['video/mp4','video/mpeg4']:
                        ut.save_uploadedfile(file,'./ssData/'+sageName+'/videoData/')
                        if vcount < 2:
                            _,col2,_=st.columns([1,1,1])
                            with col2:
                                st.video(file)
                        st.write("--->Processing " + fileName + "......")
                        ragVideo.processVideo(fileName)
                        # processVideo(ragVideo, fileName)
                        # copy the text file to chatDocText so that the knowledge will persist in the SageName
                        shutil.copy2('./ssData/'+sageName+'/textData/' + fileNameS + "_output_text.txt", './ssData/'+sageName+'/chatDocText/' + fileNameS + '_output_text.txt')
                        vcount += 1
                    # process audio
                    elif file.type in ['audio/wav','audio/mp3']:
                        ut.save_uploadedfile(file,'./ssData/'+sageName+'/audioData/')
                        if acount < 2:
                            _,col2,_=st.columns([1,1,1])
                            with col2:
                                st.audio(file)
                        st.write("--->Processing " + fileName + "......")
                        ragVideo.processAudio(fileName)
                        # processAudio(ragVideo, fileName)
                        # copy the text file to chatDocText so that the knowledge will persist in the SageName
                        shutil.copy2('./ssData/'+sageName+'/textData/' + fileNameS + "_output_text.txt", './ssData/'+sageName+'/chatDocText/' + fileNameS + '_output_text.txt')
                        acount += 1
                    # process image
                    elif file.type in ['image/png','image/jpg','image/jpeg']:
                        st.write("--->Saving " + fileName + "......")
                        ut.save_uploadedfile(file,'./ssData/'+sageName+'/imageData/')
                        if icount < 2:
                            _,col2,_=st.columns([1,1,1])
                            with col2:
                                st.image(file, caption=fileName, width=500, use_column_width=False)
                        icount += 1  
                    # process text
                    elif file.type in ['text/csv','text/plain','application/vnd.openxmlformats-officedocument.wordprocessingml.document','application/vnd.ms-excel','application/pdf']:
                        st.write("--->Saving " + fileName + "......")
                        ut.save_uploadedfile(file,'./ssData/'+sageName+'/textData/')
                        # also save textual data to ChatDocText for Chat With Your Documents
                        ut.save_uploadedfile(file,'./ssData/'+sageName+'/chatDocText/')
                # check if there are new files uploaded
                newFiles = os.listdir('./ssData/' + sageName + '/textData') + os.listdir('./ssData/' + sageName + '/imageData')
                if currentFiles != newFiles:
                    st.write('---New Files Uploaded---')
                    if 'mmIndex' in st.session_state:
                        del st.session_state.mmIndex
                        st.write('---session state mmIndex deleleted---')


                # build multimodal index
                if "mmIndex" not in st.session_state:
                    st.session_state.mmIndex = getMultiModalIndex(ragVideo, service_context, storage_context)
                    st.write('---session state mmIndex updated---')
                
                # # update service context so that the index will always have a new GenAI token
                # st.session_state.mmIndex.service_context = service_context 

                # chat with multimodal index
                mmIndexQuery(st.session_state.mmIndex, ragVideo)

# create different tabs on App
tabs = {
    "Quick Start Menu": quickStart,
    "Generate Research Report": getReport, 
    "Chat With Your Documents": getRag,
    "Chat With Anything": getMmRAG,
}

# Set up your Streamlit app
st.set_page_config(  # Alternate names: setup_page, page, layout
	layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
	initial_sidebar_state="expanded",  # Can be "auto", "expanded", "collapsed"
	page_title="ScienceSage - Your GenAI Powered Research Bestie to Extract Insigt from Voice of Science ",  # String or None. Strings get appended with "• Streamlit".
	page_icon=None,  # String, anything supported by st.image, or None.
)

# Add a title and logo
col1, col2 = st.columns([1, 3])
with col1:
    st.image("scienceSage.png", width=200)
with col2:
    st.title("ScienceSage - GenAI Powered Research Assistant to Extract Insight from Voice of Science")

# Create a sidebar with your tab names
tab_name = st.sidebar.radio("Select a Tab", list(tabs.keys()))

# Call the function associated with the selected tab
tabs[tab_name]()

