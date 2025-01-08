# This script contains the prompts for different reporting tasks
# Yong Zhang, 01/26/2024

### Generic Scientific Report Prompts

# prompt to define role of assistant 
gsr_rolePrompt = "You are an AI critical thinker research assistant. Your sole purpose is to write well written, critically acclaimed, objective and structured reports on given text."

# prompt to summarize each records 
gsr_summaryPrompt = """
{text}
-----------
Using the above text, answer in short the following question: 
{question}
-----------
if the question cannot be answered using the text, imply summarize the text. Include all factual information, numbers, stats etc if available."""


# prompt to generate reports based summaries
gsr_reportPrompt = """
Information:
--------
{research_summary}
--------
Using the above information, answer the following question or topic: "{question}" in a detailed report. 
The report should focus on the answer to the question, should be well structured, informative, 
in depth, with facts and numbers if available and a minimum of 1,200 words.
You should strive to write the report as long as you can using all relevant and necessary information provided.
You must write the report with markdown syntax.
You MUST determine your own concrete and valid opinion based on the given information. Do NOT deter to general and meaningless conclusions.
Write all used source urls at the end of the report, and make sure to not add duplicated sources, but only one reference for each.
You must write the report in apa format.
""" 

# put all sub-prompts together
gsr_prompt = {'rolePrompt':gsr_rolePrompt, 'summaryPrompt':gsr_summaryPrompt, 'reportPrompt':gsr_reportPrompt}

### prompt template for multimodal RAG
query_prompt = 'Using information from the videos, tell me what hair care tips and steps I should follow to have healthier hair.'
metaData_prompt = 'These are videos about hair care routine.'
role_prompt = 'You are an AI critical thinker. \
Given the provided information, including relevant images and retrieved context from the video, \
accurately and precisely answer the query without any additional prior knowledge. \
Please ensure honesty and responsibility, refraining from any racist or sexist remarks.' 
               
rag_prompt = role_prompt + ' ' + (
    """
        ---------------------\n
        Context: {context_str}\n
        Metadata for video: {metaData_prompt} \n
        ---------------------\n
        Query: {query_prompt}\n
        Answer:
    """
)
mrag_prompt = {'metaData_prompt':metaData_prompt, 'query_prompt':query_prompt, 'role_prompt':role_prompt,'rag_prompt':rag_prompt}


