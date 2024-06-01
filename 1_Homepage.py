import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
import os
import pdfplumber

st.set_page_config(
    page_title="Neuro-linguistic Recommendation Engine",
    page_icon="👨‍💻",
    layout="wide",
    initial_sidebar_state="collapsed"
    
)

st.subheader('Neuro-linguistic Recommendation Engine')

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "Neuro-linguistic Recommendation Engine"

llm = ChatOpenAI(model_name = "gpt-4o",temperature=0.2)

uploaded_file = st.file_uploader("", type=['pdf'])


st.session_state.pdf_context = ""
if uploaded_file is not None:
    # Displaying file details
    # st.write("Filename:", uploaded_file.name)

    # Reading and displaying the content of the PDF
    with pdfplumber.open(uploaded_file) as pdf:
        total_pages = len(pdf.pages)
        # st.write("Total pages:", total_pages)

        # Extract text from all pages
        full_text = ""
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"  # Adding a newline after each page's text for better readability

        # Display extracted text
        # st.write("Full text extracted from the PDF:")
        # st.text_area("Full Text", full_text, height=300)
        
        st.session_state.pdf_context = full_text

template = """You are an health insurance Sales Representative who will have conversation with potential customers to understand their insurace needs and recommend insurance plans available in a convincing manner.
Following context has information on insurace plan products with their details. Answer the users question based on the product information.
Be proactive and ask questions from the user to understand about user's needs and see which product best fits the user and answer the questions very convincingly.
Limit your answer to 60 words or less at a time.
Your Name is Alex. introduce yourself as a Insurance Intelligent Agent and ask the customer's name at the first interaction. After customer’s reply, ask questions from the customer and gather information to select a best plan for the customer
First Ask few Questions and suggest a plan when you are sure that plan is suit for the customer
you can start from asking following questions
1. What Type of Coverage you are looking for. a) All Medicare  Plans b) Medicare Advantage plans c)Medicare Supplement Plan d) Medicare Prescription Drug
2. Do any of these statements apply to you. a) I have Medicaid plan b) I have one or more following. diabetes, chronic heart failures. c) I live in a nursing home.
3. Do You want your insurance plan to come along with any of these services. a) Dental b)vision c)Hearing d)Fitness
4. How do you prefer to manage your health care costs? a) I want a lower monthly premium and pay for care as need it. b) I want higher premium in exchange for  low or no cost when I need care.

Give customer best two suggestions and explain the most suitable one outperform the second one.
If customer is not convinced, elaborate the advantages of the products and how they outweigh the disadvantages and convince the customer to apply for the card.
Only reply as the sales representative and do not write the responses from the customer.
Answer only based on the topic of credit cards and If the customer questions is outside the context, just say that you don't know and steer the conversation back to the topic you know. Don't give any answer outside the context of insurance plans.


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
    
    #neuro-linguistic-recommendation-engine {
        font-size: 24px;
        text-align: center;
    }
</style>
""",
    unsafe_allow_html=True,
)

# st.title("Chat with the Source Code with LLMs")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages_memory = ConversationBufferMemory(memory_key="history")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# React to user input
if prompt := st.chat_input("How can i help you?", disabled=(st.session_state.pdf_context == "")):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    messages_memory = st.session_state.messages_memory
    pdf_context = st.session_state.pdf_context
    response = generate_the_response(prompt, messages_memory, pdf_context)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})