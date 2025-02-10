# Build Your Knowledge Base Using GenAI Powered ScienceSage
ScienceSage enables researchers to build, store, update and query a knowledge base (KB) using GenAI and multimodal documents and internet search..

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

- Evaluate the RAG component:

Please use the notebook and data in RAGEvaluation folder to evaluate different RAGs. You will need to setup your own LLMs and likley databases if needed.

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
- Please create an issue if you have questions.

## Citation
@article{zhang2025build,
  title={Build Your Knowledge Base Using GenAI Powered ScienceSage},
  author={Zhang, Yong and Gyamfi, Eric Herrison and Anderson, Kelly and Roberts, Sasha and Barker, Matt},
  journal={ResearchGate preprint DOI: 10.13140/RG.2.2.24050.21444},
  year={2025}
}
