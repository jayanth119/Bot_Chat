!pip install langchain
!pip install pinecone-client
!pip install PyMuPDF
!pip install sentence-transformers
!pip install google-generativeai langchain-google-genai
!pip install -U langchain-community
!pip install pypdf
!pip install --upgrade langchain
!pip install langchain-google-genai
!pip install --upgrade langchain-pinecone



# Import necessary libraries
from langchain.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

import pinecone

# API keys and environment settings
PineconeAPI = "746547ad-1b9f-4bd7-8f7b-b630ed392d68"
GOOGLE_API_KEY = "AIzaSyDmq4KatLOIFEWBukEbarEQ_Zysrl0j9UQ"

# Function to load data from PDFs
from langchain.document_loaders import PyPDFLoader
def load_data(directory):
    loader = PyPDFLoader(directory)
    data = loader.load()
    return data

data = load_data(r"/content/1000-indian-recipe-cookbook-9781782122531.pdf")
print("Data loaded successfully.")



# Function to feed chunks
def feed(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(data)
    return text_chunks

chunks = feed(data)
print("Length of chunks is", len(chunks))

# Function to convert text to embeddings
def convert_embedding():
    embeddings = HuggingFaceEmbeddings(model_name="describeai/gemini")  # Example model
    return embeddings

embeddings = convert_embedding()
print("Embeddings model loaded successfully.")

# Test the embeddings
test = embeddings.embed_query("jayanth")
print("Test embedding:", len(test))


from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key=PineconeAPI)





# Initialize Pinecone

pc.create_index(
  name="demo",
  dimension=1024,
  metric="cosine",
  spec=ServerlessSpec(
    cloud="aws",
    region="us-east-1"
  )
)



index_name = "demo"  # Replace with your actual index name

# Assuming you have 'chunks' and 'embeddings' defined somewhere
texts = [chunk.page_content for chunk in chunks]
index = pc.Index("demo")



#

for i, t in zip(range(len(texts)), texts):
   query_result = embeddings.embed_query(texts[i])
   index.upsert(
   vectors=[
        {
            "id": str(i),  # Convert i to a string
            "values": query_result,
            "metadata": {"text":str(texts[i])} # meta data as dic
        }
    ],
    namespace="real"
)

# pc.from_texts(texts, embeddings, index_name=index_name)
# print("Texts converted to embeddings and uploaded to Pinecone.")

# # Access the existing index
# index = Pinecone(index_name)
# print("Pinecone index initialized.")

# # Access the index for search

!pip install  langchain_pinecone


from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
namespace = "real"
docsearch = PineconeVectorStore(
    index_name=index_name,
    embedding=embeddings,
    namespace=namespace,
    pinecone_api_key = PineconeAPI
)
print("Docsearch initialized.")



prompt_template = """
You are a culinary expert and a helpful assistant. Your task is to provide detailed and easy-to-follow recipes based on user input. Always be friendly, concise, and clear in your instructions.

### Instructions:
1. If the user wants to prepare a specific dish, provide a list of ingredients followed by step-by-step cooking instructions.
2. If the user wants to know the recipe for a specific dish, provide the same detailed instructions.
3. If the user provides specific ingredients, suggest a recipe using those ingredients.
4. If the user asks for a type of cuisine (e.g., Italian, Chinese), suggest a popular recipe from that cuisine.
5. If the user asks for a type of meal (e.g., breakfast, lunch, dinner, dessert), suggest a suitable recipe.
6. Include any necessary tips or variations that could enhance the dish.

### Examples:
#### Example 1:
User: "I want to prepare a dish with chicken, tomatoes, and basil."
Response:
"You can make a delicious Chicken Caprese. Here’s the recipe:
Ingredients:
- 2 chicken breasts
- 2 large tomatoes, sliced
- Fresh basil leaves
- 1 cup mozzarella cheese, sliced
- 2 tablespoons olive oil
- Salt and pepper to taste

Instructions:
1. Preheat your oven to 375°F (190°C).
2. Season the chicken breasts with salt and pepper.
3. In a skillet, heat olive oil over medium-high heat. Sear the chicken breasts for 3-4 minutes on each side until golden brown.
4. Place the chicken in a baking dish. Top each breast with tomato slices, basil leaves, and mozzarella cheese.
5. Bake in the oven for 20-25 minutes, or until the chicken is cooked through and the cheese is melted and bubbly.
6. Serve hot and enjoy your Chicken Caprese!

Tips: You can drizzle some balsamic glaze over the top before serving for an extra burst of flavor."

#### Example 2:
User: "Can you tell me the recipe for Tiramisu?"
Response:
"Sure, here’s the recipe for Tiramisu:
Ingredients:
- 6 egg yolks
- 3/4 cup sugar
- 1 cup mascarpone cheese
- 1 1/2 cups heavy cream
- 2 cups brewed espresso, cooled
- 1 package ladyfinger cookies
- Cocoa powder for dusting

Instructions:
1. In a medium bowl, beat the egg yolks and sugar together until thick and pale. Add the mascarpone cheese and mix until smooth.
2. In a separate bowl, whip the heavy cream to stiff peaks. Gently fold the whipped cream into the mascarpone mixture.
3. Dip each ladyfinger into the cooled espresso for 1-2 seconds and layer them in the bottom of a serving dish.
4. Spread half of the mascarpone mixture over the ladyfingers. Repeat with another layer of dipped ladyfingers and the remaining mascarpone mixture.
5. Dust the top with cocoa powder.
6. Refrigerate for at least 4 hours or overnight to allow the flavors to meld together.
7. Serve chilled and enjoy your Tiramisu!

Tips: For an extra touch, you can add a splash of coffee liqueur to the espresso before dipping the ladyfingers."

### User Input:
{user_input}

### Response:
"""

# Create the PromptTemplate object
prompt = PromptTemplate(template=prompt_template, input_variables=["user_input"])
prompt_template = """
You are a culinary expert and a helpful assistant. Your task is to provide detailed and easy-to-follow recipes based on user input. Always be friendly, concise, and clear in your instructions.

### Instructions:
1. If the user wants to prepare a specific dish, provide a list of ingredients followed by step-by-step cooking instructions.
2. If the user wants to know the recipe for a specific dish, provide the same detailed instructions.
3. If the user provides specific ingredients, suggest a recipe using those ingredients.
4. If the user asks for a type of cuisine (e.g., Italian, Chinese), suggest a popular recipe from that cuisine.
5. If the user asks for a type of meal (e.g., breakfast, lunch, dinner, dessert), suggest a suitable recipe.
6. Include any necessary tips or variations that could enhance the dish.

### Examples:
#### Example 1:
User: "I want to prepare a dish with chicken, tomatoes, and basil."
Response:
"You can make a delicious Chicken Caprese. Here’s the recipe:
Ingredients:
- 2 chicken breasts
- 2 large tomatoes, sliced
- Fresh basil leaves
- 1 cup mozzarella cheese, sliced
- 2 tablespoons olive oil
- Salt and pepper to taste

Instructions:
1. Preheat your oven to 375°F (190°C).
2. Season the chicken breasts with salt and pepper.
3. In a skillet, heat olive oil over medium-high heat. Sear the chicken breasts for 3-4 minutes on each side until golden brown.
4. Place the chicken in a baking dish. Top each breast with tomato slices, basil leaves, and mozzarella cheese.
5. Bake in the oven for 20-25 minutes, or until the chicken is cooked through and the cheese is melted and bubbly.
6. Serve hot and enjoy your Chicken Caprese!

Tips: You can drizzle some balsamic glaze over the top before serving for an extra burst of flavor."

#### Example 2:
User: "Can you tell me the recipe for Tiramisu?"
Response:
"Sure, here’s the recipe for Tiramisu:
Ingredients:
- 6 egg yolks
- 3/4 cup sugar
- 1 cup mascarpone cheese
- 1 1/2 cups heavy cream
- 2 cups brewed espresso, cooled
- 1 package ladyfinger cookies
- Cocoa powder for dusting

Instructions:
1. In a medium bowl, beat the egg yolks and sugar together until thick and pale. Add the mascarpone cheese and mix until smooth.
2. In a separate bowl, whip the heavy cream to stiff peaks. Gently fold the whipped cream into the mascarpone mixture.
3. Dip each ladyfinger into the cooled espresso for 1-2 seconds and layer them in the bottom of a serving dish.
4. Spread half of the mascarpone mixture over the ladyfingers. Repeat with another layer of dipped ladyfingers and the remaining mascarpone mixture.
5. Dust the top with cocoa powder.
6. Refrigerate for at least 4 hours or overnight to allow the flavors to meld together.
7. Serve chilled and enjoy your Tiramisu!

Tips: For an extra touch, you can add a splash of coffee liqueur to the espresso before dipping the ladyfingers."

### User Input:
{user_input}

### Response:
"""

# Create the PromptTemplate object
prompt = PromptTemplate(template=prompt_template, input_variables=["user_input"])


# Create the prompt template with specified input variables

# Create a dictionary for chaining options
chango = {"prompt": prompt}

print(chango)


# Instantiate the language model
llm = ChatGoogleGenerativeAI(model="models/gemini-1.0-pro", google_api_key=GOOGLE_API_KEY)

PINECONE_API_KEY = PineconeAPI


# Create the RetrievalQA chain
from langchain_google_vertexai import ChatVertexAI
# qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever= docsearch.as_retriever(search_kwargs={'k': 2}),
#     return_source_documents=True,
#     chain_type_kwargs=chango
# )

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt, "document_variable_name": "context"}
)



# User interaction loop
while True:
    ui = input("Enter query: ")
    result = qa({"query": ui})
    print(result["result"])

print("Docsearch initialized.")

# Define the prompt template
prompt_template = """ You are a culinary expert and a helpful assistant. Your task is to provide detailed and easy-to-follow recipes based on user input. Always be friendly, concise, and clear in your instructions.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

# Create the PromptTemplate object
prompt = PromptTemplate(template=prompt_template, input_variables=["user_input", "context"])

# Instantiate the language model
llm = ChatGoogleGenerativeAI(model="models/gemini-1.0-pro", google_api_key=GOOGLE_API_KEY)  # Replace with your Google API key

# Create the RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# User interaction loop
while True:
    ui = input("Enter query: ")
    result = qa({"query": ui})  # Pass both 'query' and 'user_input' variables here
    print(result["result"])
