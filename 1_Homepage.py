import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
import os
import pdfplumber
import markdown

st.set_page_config(
    page_title="Neuro-linguistic Recommendation Engine",
    page_icon="👨‍💻",
    layout="wide",
    initial_sidebar_state="collapsed"
    
)

st.subheader('Neuro-linguistic Recommendation Engine - Healthcare Plans Chatbot')

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "Neuro-linguistic Recommendation Engine - Healthcare Plans Chatbot"

llm = ChatOpenAI(model_name = "gpt-4o",temperature=0.2)

# uploaded_file = st.file_uploader("", type=['pdf'])


# st.session_state.pdf_context = ""
# if uploaded_file is not None:
#     with pdfplumber.open(uploaded_file) as pdf:
#         total_pages = len(pdf.pages)
#         full_text = ""
#         for page in pdf.pages:
#             text = page.extract_text()
#             if text:
#                 full_text += text + "\n" 
#         st.session_state.pdf_context = full_text

template = """You are a health insurance Sales Representative who will have conversations with potential customers to understand their insurance needs and recommend insurance plans available in a convincing manner.
Following context has information on insurance plan products with their details. Answer the user's question based on the product information.
Be proactive and ask questions from the user to understand more about the user's needs see which product best fits the user and answer the questions very convincingly.
Limit your answer to 60 words or less at a time.
Your Name is Alex, Cigna Healthcare Assistant. introduce yourself as an Insurance Intelligent Agent and ask the customer's name at the first interaction. After customer’s reply, ask questions from the customer and gather information to select a best plan for the customer

Wait for the customer's reply with the name and then ask a few questions and suggest a plan when you are sure that plan is the best fit for the customer
You can start by asking the following questions one at a time as a guide to deciding the best-fit plan. Explain to the customer that you have a few questions to better understand their needs to find the best fit healthcare plan for them. 

1) you may qualify to receive a subsidy, which is a tax credit that lowers your monthly premium. 
Would you like to see if you qualify for financial assistance?
    a)Yes 
    b)No
2) What is your age?
3)Do you have any pre-existing conditions? 
    a)Yes 
    b)No
4) Do you have any chronic diabetes condition?
    a)Yes 
    b)No
5) how much healthcare do you think you'll use in 2024? 
    a)Low 
    b)Medium 
    c)High

Give the customer the best two suggestions and explain the most suitable one and how it outperforms the second one.
If you detect a negative sentiment and the customer is not convinced, elaborate on the advantages of the product and how they outweigh the disadvantages and convince the customer. Ask the customer whether they have more questions and how else you can helpt them. 
Only reply as the sales representative and do not write the responses from the customer.
Answer only based on the topic of healthcare insurance plans and If the customer's questions are outside the context, just say that you don't know and steer the conversation back to the healthcare topic you know. Don't give any answer outside the context of insurance plans.



Context: {pdf_context}

Current conversation:
{history}
Human: {input}
AI Assistant:"""

def generate_the_response(prompt, memory, pdf_context):
    # PROMPT = PromptTemplate(input_variables=["history", "input", "pdf_context"], template=template)
    PROMPT =  PromptTemplate.from_template(template).partial(pdf_context=pdf_context)
    llm_chain = ConversationChain(
        prompt=PROMPT,
        llm=llm,
        verbose=True,
        memory=memory,
    )
    result = llm_chain.predict(input=prompt)
    return result


st.markdown(
    """
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
    
    #neuro-linguistic-recommendation-engine-healthcare-plans-chatbot {
        font-size: 22px;
        text-align: center;
    } 
    [data-testid="stChatInputTextArea"] {
        color: black;
        background: #ffffff;
        font-size: 20px;
        font-weight: 600;
    }
</style>
""",
    unsafe_allow_html=True,
)

# st.title("Chat with the Source Code with LLMs")


main_context = """Following is a line of health insurance plans and their details.

Product 1 -  Connect Bronze 5500 Individual Med Deductible

Product 1 Details

Monthly Premium - $328.99
Individual Deductible - $9450
Family Deductible - not applicable
Max out of pocket - $9450
Network - HMO
Primary Care - $0 after deductible
Specialist - $0 after deductible
Emergency Room - $0 after deductible
Hospital stay - $0 after deductible

Product 1 details Ends

Product 2 - Connect Bronze 5500 Individual Med Deductible

Product 2 Details

Monthly Premium - $343
Individual Deductible - $5500 
Family Deductible - Not applicable
Max out of pocket - $ 9450
Network - HMO
Primary Care - $35
Specialist - 50% after deductible
Emergency Room - 50% after deductible
Hospital stay - 50% after deductible

Product 2 details ends

Product 3 - Connect Gold 2500 Individual Med Deductible Enhanced Diabetes Care

Product 3 Details

Monthly Premium - $425.86
Individual Deductible - $2500
Family Deductible - Not Applicable 
Max out of pocket - $7700
Network - HMO
Primary Care - $15
Specialist - $50
Emergency Room - 40% after deductible 
Hospital stay - 20% after deductible 
This package has special care for diabetes 

Product 3 details ends

Product 4 - Connect Gold 0 individual Med Deductible
Product 4 details
Monthly Premium - $443.45
Individual Deductible - $0
Family Deductible - Not applicable
Max out of pocket - $8500
Network - HMO
Primary Care - $40
Specialist - $75
Emergency Room - $750
Hospital stay - $1200 per day

Product 4 details ends

Product 5 - Connect Gold CMS Standard Product 5 details
Monthly Premium - $427.56
Individual Deductible - $1500
Family Deductible - $3000
Max out of pocket - $8700
Network - HMO
Primary Care - $30
Specialist - $60
Emergency Room - 25% after deductible
Hospital stay - 25% after deductible

Product 5 details ends

Product 6-  Connect Gold 3500 Individual Med Deductible
Product 6 details

Monthly Premium - $427.30
Individual Deductible - $3500
Family Deductible - Not applicable
Max out of pocket - $5700
Network - HMO
Primary Care - $25
Specialist - $50
Emergency Room - 30% after deductible
Hospital stay - 30% after deductible

Product 6 details ends
"""


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages_memory = ConversationBufferMemory(memory_key="history")


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        message_content_with_newlines = message["content"].replace('\n', '<br>')
        message_html_content = markdown.markdown(message_content_with_newlines)
        st.markdown(f"""<div style="font-size: 70px">{message_html_content}</div>""", unsafe_allow_html=True)


# React to user input
if prompt := st.chat_input("How can i help you?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    messages_memory = st.session_state.messages_memory
    # pdf_context = st.session_state.pdf_context
    pdf_context = main_context
    response = generate_the_response(prompt, messages_memory, pdf_context)
    # Display assistant response in chat message container
    
    
    with st.chat_message("assistant"):
        response_with_newlines = response.replace('\n', '<br>')
        response_html_content = markdown.markdown(response_with_newlines)
        st.markdown(f"""<div style="font-size: 70px">{response_html_content}</div>""", unsafe_allow_html=True)
            
    st.session_state.messages.append({"role": "assistant", "content": response})
