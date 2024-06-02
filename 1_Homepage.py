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

template = """You are an health insurance Sales Representative who will have conversation with potential customers to understand their insurance needs and recommend insurance plans available in a convincing manner.
Following context has information on insurance plan products with their details. Answer the users question based on the product information.
Be proactive and ask questions from the user to understand more about user's needs and see which product best fits the user and answer the questions very convincingly.
Limit your answer to 60 words or less at a time.
Your Name is Alex. introduce yourself as a Insurance Intelligent Agent and ask the customer's name at the first interaction. After customer’s reply, ask questions from the customer and gather information to select a best plan for the customer

Wait for customer's reply with the name and then ask few questions and suggest a plan when you are sure that plan is the best fit for the customer
You can start by asking following questions one at a time as a guide to decide the best fit plan. Explain to the customer that you have a few questions to better understand their needs to find the best fit healthcare plan for them. 
1. What Type of Coverage you are looking for. a) All Medicare  Plans b) Medicare Advantage plans c)Medicare Supplement Plan d) Medicare Prescription Drug
2. Do any of these statements apply to you. a) I have Medicaid plan b) I have one or more following. diabetes, chronic heart failures. c) I live in a nursing home.
3. Do You want your insurance plan to come along with any of these services. a) Dental b)vision c)Hearing d)Fitness
4. How do you prefer to manage your health care costs? a) I want a lower monthly premium and pay for care as need it. b) I want higher premium in exchange for  low or no cost when I need care.

Give customer the best two suggestions and explain the most suitable one and how it outperforms the second one.
If you detect a negative sentiment and the customer is not convinced, elaborate the advantages of the products and how they outweigh the disadvantages and convince the customer to apply for the healthcare plan.
Only reply as the sales representative and do not write the responses from the customer.
Answer only based on the topic of healthcare insruance plans and If the customer questions is outside the context, just say that you don't know and steer the conversation back to the healthcare topic you know. Don't give any answer outside the context of insurance plans.


Context: {pdf_context}

Current conversation:
{history}
Human: {input}
AI Assistant: Each time chat with the markdown format"""

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
</style>
""",
    unsafe_allow_html=True,
)

# st.title("Chat with the Source Code with LLMs")


main_context = """
Following is a line of health insurance plans and their details.

Product 1 -AARP Medicare Supplement Insurance Plan F + wellness extras
Product 1 Details
Monthly premium starting at
160.261
Out-of-pocket maximum
$0
Plan F is only available to those first eligible for Medicare before 2020.
Plan F has no out-of-pocket costs, covers Part A and B deductibles, plus wellness extras; may be a good choice for someone who’d rather pay a higher monthly premium for the most comprehensive benefits

Ability to see any provider nationwide that accepts Medicare patients
A separate prescription drug plan is needed for prescription drug coverage
Plans come along with wellness extras including Dental discounts, Vision and hearing discounts, 24/7 Nurse line, Brain health, Driver safety
Dental
Discounts for dental services from in-network dentists through DentegraIn-network discounts generally average 30-40%11 off of contracted rates nationally for a range of dental services including: cleanings, exams, fillings, crowns
Covered
Vision
Routine eye exams at participant providers: $5012
Take an additional $50 off the AARP Vision Discount or best instore offer on no-line progressive lenses with frame purchase at LensCrafters13
Hearing
An additional $100 off the AARP member rate on select hearing aids

Plus a 15% discount on hearing aid accessories
Fitness
A gym membership at no additional cost to you
OTC coverage
Not applicable
Medical costs - what you'll pay
Deductible
Part A (Hospital): $0
Part B (Medical): $0
Doctor visit
$0 After Part B deductible is met
Referral to specialist required?
No, however, provider must accept Medicare patients
Inpatient hospital care7
$0 for days 1-60
$0 for days 61-90
$0 while using 60 lifetime reserve days for days 91 and later
$0 for an additional 365 days, after lifetime reserve days are used8
All costs beyond the additional 365 days

Product 1 Details Ends

Product 2 - AARP Medicare Supplement Insurance Plan F

Product 2 Details 

Monthly premium starting at
139.631
Out-of-pocket maximum
$0
Plan F is only available to those first eligible for Medicare before 2020.
Plan F has no out-of-pocket costs, covers Part A and B deductibles; may be a good choice for someone who’d rather pay a higher monthly premium in exchange for the most comprehensive benefits
Ability to see any provider nationwide that accepts Medicare patients
A separate prescription drug plan is needed for prescription drug coverage
Wellness extras do not come along with this plan
Dental
Not applicable
Vision
Not applicable
Hearing
Not applicable
Fitness
Not applicable
OTC coverage
Not applicable
Deductible
Part A (Hospital): $0
Part B (Medical): $0
Doctor visit
$0
Referral to specialist required?
No, however, provider must accept Medicare patients
Inpatient hospital care7
$0 for days 1-60
$0 for days 61-90
$0 while using 60 lifetime reserve days for days 91 and later
$0 for an additional 365 days, after lifetime reserve days are used8
All costs beyond the additional 365 days

Product 2 Details Ends

Product 3 - AARP Medicare Supplement Insurance Plan G + wellness extras

Product 3 Details

Monthly premium starting at
121.791
Out-of-pocket maximum
$240

Plan G is the most popular Medicare supplement plan, covers Part A deductible, plus wellness extras; may be a good choice for someone who wants comprehensive coverage with low to no out-of-pocket costs
Ability to see any provider nationwide that accepts Medicare patients
A separate prescription drug plan is needed for prescription drug coverage
Plans come along with wellness extras including Dental discounts, Vision and hearing discounts, 24/7 Nurse line, Brain health, Driver safety
Dental
Discounts for dental services from in-network dentists through DentegraIn-network discounts generally average 30-40%11 off of contracted rates nationally for a range of dental services including: cleanings, exams, fillings, crowns
Covered
Vision
Routine eye exams at participant providers: $5012
Take an additional $50 off the AARP Vision Discount or best instore offer on no-line progressive lenses with frame purchase at LensCrafters13
Hearing
An additional $100 off the AARP member rate on select hearing aids

Plus a 15% discount on hearing aid accessories
Fitness
A gym membership at no additional cost to you
OTC coverage
Not applicable
Deductible
Part A (Hospital): $0
Part B (Medical): $240
Doctor visit
$0 After Part B deductible is met
Referral to specialist required?
No, however, provider must accept Medicare patients
Inpatient hospital care7
$0 for days 1-60
$0 for days 61-90
$0 while using 60 lifetime reserve days for days 91 and later
$0 for an additional 365 days, after lifetime reserve days are used8
All costs beyond the additional 365 days
Product 3 Details ends

Product 4 - AARP Medicare Supplement Insurance Plan N + wellness extras

Product 4 Details

Monthly premium starting at

104.381

Out-of-pocket maximum - N/A

Plan N is a more affordable Medicare supplement plan, covers Part A deductible, plus wellness extras; may be a good choice for someone who prefers a lower monthly premium and to pay out-of-pockets costs for care when needed
Ability to see any provider nationwide that accepts Medicare patients
A separate prescription drug plan is needed for prescription drug coverage
Plans come along with wellness extras including Dental discounts, Vision and hearing discounts, 24/7 Nurse line, Brain health, Driver safety
Dental
Discounts for dental services from in-network dentists through DentegraIn-network discounts generally average 30-40%11 off of contracted rates nationally for a range of dental services including: cleanings, exams, fillings, crowns
Covered
Vision
Routine eye exams at participant providers: $5012
Take an additional $50 off the AARP Vision Discount or best instore offer on no-line progressive lenses with frame purchase at LensCrafters13
Hearing
An additional $100 off the AARP member rate on select hearing aids

Plus a 15% discount on hearing aid accessories
Fitness
A gym membership at no additional cost to you
OTC coverage
Not applicable
Deductible
Part A (Hospital): $0
Part B (Medical): $240
Doctor visit
Up to $20 copay After Part B deductible is met
Referral to specialist required?
No, however, provider must accept Medicare patients
Inpatient hospital care7
$0 for days 1-60
$0 for days 61-90
$0 while using 60 lifetime reserve days for days 91 and later
$0 for an additional 365 days, after lifetime reserve days are used8
All costs beyond the additional 365 days
Product 4 Details Ends

Product 5 - AARP Medicare Supplement Insurance Plan G

Product 5 Details
Monthly premium starting at
102.571
Out-of-pocket maximum
$240

Plan G is the most popular Medicare supplement plan, covers Part A deductible; may be a good choice for someone who wants comprehensive coverage with low to no out-of-pocket costs
Ability to see any provider nationwide that accepts Medicare patients
A separate prescription drug plan is needed for prescription drug coverage
Wellness extras do not come along with this plan
Dental
Not applicable
Vision
Not applicable
Hearing
Not applicable
Fitness
Not applicable
OTC coverage
Not applicable
Deductible
Part A (Hospital): $0
Part B (Medical): $240
Doctor visit
$0 After Part B deductible is met
Referral to specialist required?
No, however, provider must accept Medicare patients
Inpatient hospital care7
$0 for days 1-60
$0 for days 61-90
$0 while using 60 lifetime reserve days for days 91 and later
$0 for an additional 365 days, after lifetime reserve days are used8
All costs beyond the additional 365 days
Product 5 Details Ends

Product 6 - UHC Complete Care TX-003P (HMO-POS C-SNP)
Product 6 Details 
Monthly premium
$0
Out-of-pocket maximum
$3,700
Take advantage of extra services and programs designed to help you better manage your cardiovascular disorder, chronic heart failure or diabetes.
Access to doctors participating in the UnitedHealthcare Medicare network
This plan includes prescription drug coverage.
Benefits including Preventive dental services, Comprehensive dental services, Hearing aids, Hearing exams, Eyewear (glasses/contacts), Eye exams, Fitness program through Renew Active
Dental
$2,000 every year towards covered dental services

$0 copay for network preventive dental including oral exams, X-rays, routine cleanings and fluoride
Vision
Routine eye exam: $0 copay, 1 per year
$0 copay

Plan pays up to $250 every year for frames or contact lenses. Standard single, bifocal, trifocal, or progressive lenses are covered in full.

Home-delivered eyewear available nationwide only through UnitedHealthcare Vision (select products only).
Hearing
Routine hearing exam: $0 copay, 1 per year
Copays from $99 - $1,249 for a broad selection of OTC and brand-name prescription hearing aids through UnitedHealthcare Hearing, up to 2 hearing aids every year
Fitness
$0 copay for Renew Active®, which includes a free gym membership, plus online fitness classes and brain health challenges.
OTC coverage
Not applicable
Annual medical deductible
No deductible
Primary care provider
$0 copay In-Network
Specialist
$20 copay (referral required) In-Network
Inpatient hospital care
$175 copay per day: days 1-5
$0 copay per day after that for unlimited days

Product 6 Details ends

Product 7 - UHC Complete Care TX-001A (Regional PPO C-SNP)
Product 7 details 
Monthly premium
$0 - $10
Out-of-pocket maximum
$0 - $8,850
Take advantage of extra benefits, programs and services designed especially for people with diabetes, chronic heart failure or cardiovascular disorders. This plan has $0 out-of-pocket costs on most services for people with Medicare and Medicaid.
Access to in-network costs when you see doctors participating in the UnitedHealthcare Medicare network
This plan includes prescription drug coverage.
Benefits including Preventive dental services, Comprehensive dental services, Hearing aids, Hearing exams, Eyewear (glasses/contacts), Eye exams, OTC benefit, Fitness program through Renew Active

Hearing
Routine hearing exam: $0 copay, 1 per year
$2,000 allowance for a broad selection of OTC and brand-name prescription hearing aids through UnitedHealthcare Hearing, up to 2 hearing aids every year
Fitness
$0 copay for Renew Active®, which includes a free gym membership, plus online fitness classes and brain health challenges.
OTC coverage
$220 credit per quarter to buy covered OTC products.
Annual medical deductible
$0 - $240 combined in and out-of-network
Primary care provider
$0 copay or 20% of the cost In-Network
20% of the cost Out-Of-Network
Specialist
$0 copay - 20% of the cost In-Network
20% of the cost Out-Of-Network
Inpatient hospital care
$0 - $1,925 per stay for unlimited days

Product 7 Details ends
Product 8 - AARP Medicare Advantage from UHC TX-0027 (HMO-POS)

Product 8 Details

Monthly premium
$0
Out-of-pocket maximum
$3,800

Take advantage of extra benefits available with our Medicare Advantage plans. This plan is a good choice for someone who wants more coverage than Original Medicare but doesn't want to pay an additional monthly premium.
Access to doctors participating in the UnitedHealthcare Medicare network
This plan includes prescription drug coverage.
Benefits including Preventive dental services, Comprehensive dental services, Hearing aids, Hearing exams, Eyewear (glasses/contacts), Eye exams, OTC benefit, Fitness program through Renew Active

Dental
$2,500 every year towards covered dental services

$0 copay for network preventive dental including oral exams, X-rays, routine cleanings and fluoride
Vision
Routine eye exam: $0 copay, 1 per year
$0 copay

Plan pays up to $250 every year for frames or contact lenses. Standard single, bifocal, trifocal, or progressive lenses are covered in full.

Home-delivered eyewear available nationwide only through UnitedHealthcare Vision (select products only).
Hearing
Routine hearing exam: $0 copay, 1 per year
Copays from $99 - $1,249 for a broad selection of OTC and brand-name prescription hearing aids through UnitedHealthcare Hearing, up to 2 hearing aids every year
Fitness
$0 copay for Renew Active®, which includes a free gym membership, plus online fitness classes and brain health challenges.
OTC coverage
$50 credit per quarter to buy covered OTC products.
Annual medical deductible
No deductible
Primary care provider
$0 copay In-Network
Specialist
$20 copay (referral required) In-Network
Inpatient hospital care
$295 copay per day: days 1-5
$0 copay per day after that for unlimited days
Product 8 details ends
"""


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages_memory = ConversationBufferMemory(memory_key="history")


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.text(message["content"].replace('*', ''))


# React to user input
if prompt := st.chat_input("How can i help you?"):
    # Display user message in chat message container
    st.chat_message("user").text(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    messages_memory = st.session_state.messages_memory
    # pdf_context = st.session_state.pdf_context
    pdf_context = main_context
    response = generate_the_response(prompt, messages_memory, pdf_context)
    # Display assistant response in chat message container
    
    
    with st.chat_message("assistant"):
        st.text(response.replace('*', ''))
            
    st.session_state.messages.append({"role": "assistant", "content": response})