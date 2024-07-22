import os
from langchain.memory import ConversationTokenBufferMemory
from langchain_community.llms import Cohere
import cohere
from image import extract_text_from_image
from image import model
import fitz  # PyMuPDF
from difflib import SequenceMatcher

# Set the Cohere API key
os.environ['COHERE_API_KEY'] = '7vOJrMLIUJyuIf0UFk3JzcqsqS3oqoLZ18NIDSzA'   #Environment variables are a way to store configuration settings and secrets securely.
api_key = os.environ['COHERE_API_KEY']
cohere_client = cohere.Client(api_key)  #it's a class allows you to interact with Cohere's language models.

# Initialize the Cohere LLM
llm = Cohere()

# Initialize conversation memory with the LLM
memory = ConversationTokenBufferMemory(llm=llm, max_token_length=1000)     #max_token_length=1000: sets the maximum number of tokens that the memory buffer can hold. 

# Initialize conversation history
conversation_history = []

model = extract_text_from_image("temp_image.jpg", model)

# Generate text using Cohere
def generate_text(full_prompt, model='command-xlarge-nightly'):
    response = cohere_client.generate(
        model=model,
        prompt=full_prompt,
        max_tokens=300,     #maximum number of tokens in the generated text
        temperature=0.7     #controls the randomness
    )
    return response.generations[0].text.strip()

# Summarize text using Cohere
def summarize_text(text, model='command-xlarge-nightly'):
    response = cohere_client.summarize(
        model=model,
        text=text,
        length='short'
    )
    return response.summary.strip()

# Create a topic prompt
def create_topic_prompt(topic):
    prompt = f"Write a detailed explanation about the topic: {topic}"
    return prompt

# Create a question prompt
def create_question_prompt(text, question):
    prompt = f"Here is some information: {text}\n\nQuestion: {question}\n\nAnswer:"
    return prompt

# Class for conversation memory
class ConversationMemory:   #class that will be used to manage conversation history.
    def __init__(self):     #special method used for initializing objects of the class.
        self.history = []   #an empty list self.history to store the conversation history.

    def add_to_history(self, user_input, response):     #add_to_history is used to add entries to the conversation history
        self.history.append({"user_input": user_input, "response": response})     #a dictionary with keys "user_input" and "response" and appends this dictionary to the self.history list.

    def get_history(self):     #returns the entire conversation history stored in self.history
        return self.history

# Call Cohere API
def call_cohere_api(prompt, cohere_client, model="command-xlarge-nightly"):     #will be used to interact with the Cohere AP
    try:
        response = cohere_client.generate( 
            model=model,
            prompt=prompt,
            max_tokens=150,     # Limits the maximum number of tokens in the generated text to 150. 
            temperature=0.7,
        )
        return response.generations[0].text.strip()
    except Exception as e:
        print(f"API Error: {e}")
        return None
    
def parse_response(response):
    return response.strip() if response else "No response from the API."     #If it's not empty, it strips any leading or trailing whitespace from the response using response.strip() and returns the cleaned response.

# Initialize memory
memory = ConversationMemory()

# Generate text about a topic
def generate_text_about_topic(topic, cohere_client):
    prompt = create_topic_prompt(topic)
    response = call_cohere_api(prompt, cohere_client)
    parsed_response = parse_response(response)
    memory.add_to_history(prompt, parsed_response)
    return parsed_response

# Ask questions about the generated text
def ask_question_about_text(text, question, cohere_client):
    prompt = create_question_prompt(text, question)
    response = call_cohere_api(prompt, cohere_client)
    parsed_response = parse_response(response)
    memory.add_to_history(prompt, parsed_response)
    return parsed_response

# Class for summarizing conversation memory
class ConversationSummaryMemory(ConversationMemory):     #a new class ConversationSummaryMemory that inherits from the ConversationMemory class
    def __init__(self):
        super().__init__()     #calling the parent class's constructor (super().__init__()), which sets up the history list
        self.qa_pairs = {}     # initialized a new dictionary to store question-answer pairs.

    def add_to_history(self, user_input, response):
        super().add_to_history(user_input, response)    # parent class's add_to_history
        if "Question" in user_input:
            question = user_input.split("Question:")[1].split("\n")[0].strip()     
            self.qa_pairs[question] = response

    def get_answer_for_question(self, question):
        return self.qa_pairs.get(question, None)

    def summarize_history(self, cohere_client):     #The method generates a text summary of the conversation history by combining user inputs and bot responses.
        history_text = "\n".join([f"User: {entry['user_input']}\nBot: {entry['response']}" for entry in self.history])
        summary_prompt = f"Summarize the following conversation:\n\n{history_text}\n\nSummary:"
        summary = call_cohere_api(summary_prompt, cohere_client)
        return summary

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.
    """
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Function to evaluate prediction accuracy
def evaluate_prediction(extracted_text, actual_text):
    """
    Evaluate the accuracy of the extracted text compared to the actual text.
    """
    matcher = SequenceMatcher(None, extracted_text, actual_text)
    accuracy = matcher.ratio() * 100  # Convert to percentage
    return accuracy

