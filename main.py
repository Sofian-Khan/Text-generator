# Import necessary libraries and modules
from lamini.api.embedding import Embedding
from lamini import Lamini
import faiss
import time
import numpy as np
from tqdm import tqdm
from dir_help import DirectoryLoader

# Define a class for creating an index using Lamini
class LaminiIndex:
    def __init__(self, loader, api_key=None):
        self.loader = loader
        self.api_key = api_key  # Store the api_key
        self.build_index()  # Call the build_index method upon initialization

    # Method to build the index
    def build_index(self):
        self.content_chunks = []  # Initialize an empty list to store content chunks
        # Iterate over the loader, which presumably yields data chunks
        for chunk_batch in tqdm(self.loader):
            # Get embeddings for the chunk_batch using the provided api_key
            embeddings = self.get_embeddings(chunk_batch, self.api_key)
            # Create an index using Faiss and add the embeddings
            self.index = faiss.IndexFlatL2(len(embeddings[0]))
            self.index.add(embeddings)
            # Extend the content_chunks list with the current chunk_batch
            self.content_chunks.extend(chunk_batch)

    # Method to get embeddings for examples
    def get_embeddings(self, examples, api_key):
        # Initialize an Embedding object with the provided api_key
        ebd = Embedding(api_key=api_key)
        # Generate embeddings for the examples
        embeddings = ebd.generate(examples)
        # Extract the embeddings from the generated embeddings
        embedding_list = [embedding[0] for embedding in embeddings]
        return np.array(embedding_list)  # Return the embeddings as a numpy array

    # Method to query the index
    def query(self, query, k=5):
        # Get the embedding for the query using the provided api_key
        embedding = self.get_embeddings([query], self.api_key)[0]
        # Convert the embedding to a numpy array
        embedding_array = np.array([embedding])
        # Search the index for the nearest neighbors to the query embedding
        _, indices = self.index.search(embedding_array, k)
        # Return the content chunks corresponding to the indices
        return [self.content_chunks[i] for i in indices[0]]

# Define a class for the QueryEngine
class QueryEngine:
    def __init__(self, index, k=5, api_key=None):
        self.index = index  # Store the index
        self.k = k  # Store the number of nearest neighbors to retrieve
        # Initialize a Lamini model with the provided model_name and api_key
        self.model = Lamini(model_name="mistralai/Mistral-7B-Instruct-v0.1", api_key=api_key)

    # Method to answer a question using the index and the Lamini model
    def answer_question(self, question):
        # Query the index for the most similar content chunks to the question
        most_similar = self.index.query(question, k=self.k)
        # Construct a prompt by joining the most similar chunks with the question
        prompt = "\n".join(reversed(most_similar)) + "\n\n" + question
        # Print the prompt
        print("------------------------------ Prompt ------------------------------\n" + prompt + "\n----------------------------- End Prompt -----------------------------")
        # Generate a response using the Lamini model and the constructed prompt
        return self.model.generate("<s>[INST]" + prompt + "[/INST]")

# Define a class for the RetrievalAugmentedRunner
class RetrievalAugmentedRunner:
    def __init__(self, dir, k=5, api_key=None):
        self.k = k  # Store the number of nearest neighbors to retrieve
        # Initialize a DirectoryLoader with the provided directory
        self.loader = DirectoryLoader(dir)
        self.api_key = api_key  # Store the api_key

    # Method to train the RetrievalAugmentedRunner
    def train(self):
        # Create an index using LaminiIndex and the loader
        self.index = LaminiIndex(self.loader, api_key=self.api_key)

    # Method to call the RetrievalAugmentedRunner with a query
    def __call__(self, query):
        # Initialize a QueryEngine with the index, k, and api_key
        query_engine = QueryEngine(self.index, k=self.k, api_key=self.api_key)
        # Answer the question using the QueryEngine
        return query_engine.answer_question(query)
def main():
    api_key = "11a5515dc8fca9dbafd27c682965d7f194d3e628ec96917c824cd2383bc82dfb"  
    model = RetrievalAugmentedRunner(dir="data", api_key=api_key)
    start = time.time()
    model.train()
    print("Time taken to build index: ", time.time() - start)
    while True:
        prompt = input("\n\nEnter another question : ")
        start = time.time()
        print(model(prompt))
        

main()
