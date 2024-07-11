#Importing all the important modules
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from pymongo import MongoClient
from streamlit_feedback import streamlit_feedback


#loading the API key
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


#Setting up the MongoDB Client
MONGODB_URI=os.getenv("MONGODB_URI")
client = MongoClient(MONGODB_URI)

db = client['case_testing']

log_collection = db['cases']


#Initializing all the paths for data and other local files
dirname = os.path.dirname(__file__)
faiss_path = os.path.join(dirname, "faiss_index")
confo_path = os.path.join(dirname, "final_text_confo.txt")
ik_path = os.path.join(dirname, "final_text_ik.txt")
nch_path = os.path.join(dirname, "final_text_nch.txt")
final_path = [confo_path, ik_path, nch_path]


#Diving the text into little chunks for faster processing
@st.cache_data()
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=5000)
    chunks = text_splitter.split_text(text)
    return chunks


#Converting those chunks into vectors and then saving them in vector database
@st.cache_data()
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


#Here is the prompt and Conversational chain formation using Gemini-pro
@st.cache_resource
def get_conversational_chain():
    #prompt template for the conversational chain
    prompt_template = """
    You are an expert at analyzing doctor reports invoice and history return everything you know about the case which is similar to the question but do not show any people names. 
    Also return billing range not exact amount according to similar cases at the top. 
    Return the nearest similar answer in detail as possible and also return 100 words summary of similar cases. 
    Must return the Case Number of the similar cases but do not show the case number of provided case. 
    Do not create random Case Number on your own. Only provide case number present in the provided content. Show dates and money compensation. 
    The reply format should be: 
    1. Expected Medical or Billing range 
    2. Similar cases with their Case No. and their Pdf links 
    3. Applicability to the current case. 
    Context: \n{context}\n
    Question: \n{question}\n
    Answer:
    """
    #creating a ChatGoogleGenerativeAI object with the specified model and temperature
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    # Creating a PromptTemplate object with the prompt template and input variables
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    # Loading the question answering chain with the model and prompt
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


#This function is for processing user input and finding the similar case, this is kind of an important function
@st.cache_data()
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]


#This is for regenerating output again and again if user wants
def regenerate_user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]


#This functions get all the textual data and previous functions to fnally form the vector database
def faiss_db(path):
    final_data = ""
    for i in path:
        file = open(i, "r", encoding='utf-8')
        file_content = file.read()
        string_list = file_content.split("\nsyndaxlevenis\n")
        final_text = [element for element in string_list] 
        for text in final_text:
            final_data = final_data + "\n\n" + text
        
    chunk_text = get_text_chunks(final_data)
    get_vector_store(chunk_text)


#This is for handelling user feedback
def handle_feedback(feedback_choice, optional_feedback_text):
    feedback_value = {
        'choice': feedback_choice,
        'optional_text': optional_feedback_text
    }
    log_collection.insert_one({
        'Query': st.session_state.get('query', ''),
        'Solution': st.session_state.get('solution', ''),
        'Feedback': feedback_value
    })
    st.session_state['button_clicked'] = True
    
def main():      
    # Setting up the page configuration
    st.set_page_config("Online Query Resolver")
    
    # Checking if FAISS vector database already exist or not and if not then a new one is created
    if os.path.exists(faiss_path):
        pass
    else:
        faiss_db(final_path)
        
    # Creating a form for user queries
    with st.form(key='my_form'):
        # Text area for user input
        user_question = st.text_area("Having a grievance with your e-commerce experience? Discover whether you're eligible for by exploring our platform.", height=150)
        
        # Create columns for buttons
        col1, col2 = st.columns(2)
        with col1:
            submit_button = st.form_submit_button("Submit")
        with col2:
            regenerate_button = st.form_submit_button("Regenerate")
            
        # Check if there is a user question
        if user_question:
            st.session_state['query'] = user_question
            st.session_state['solution'] = user_input(user_question)
            
            # Handle submit button
            if submit_button:
                st.write(st.session_state['solution'])
            
            # Handle regenerate button
            if regenerate_button:
                st.session_state['query'] = user_question
                st.session_state['solution'] = regenerate_user_input(user_question)
                st.write(st.session_state['solution'])
            
    if 'fb_k' not in st.session_state:
        st.session_state['fb_k'] = None
    
    st.session_state['button_clicked'] = False
        
    # Creating a form for storing feedback
    with st.form('form'):
        col1, col2 = st.columns(2)
        with col1:
            feedback_choice = st.radio("Provide your feedback:", ('👍', '👎'), key='feedback_choice', horizontal=True)
        with col2:
            optional_feedback_text = st.text_input("Enter your feedback here (Optional)", key='optional_feedback_text')
        
        submit_button = st.form_submit_button('Save feedback')
        
        if submit_button:
            handle_feedback(feedback_choice, optional_feedback_text)
            st.session_state['button_clicked'] = True

    if st.session_state.get('button_clicked', False):
        st.write("Thank you for your feedback!")
                

if __name__ == "__main__":
    main()
