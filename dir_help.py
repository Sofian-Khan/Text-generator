import os  # Import the os module for interacting with the operating system

# Define a class for chunking data into smaller chunks
class DefaultChunker:
    def __init__(self, chunk_size=512, step_size=256):  # Initialize the chunker with default chunk size and step size
        self.chunk_size = chunk_size  # Store the chunk size
        self.step_size = step_size  # Store the step size

    def get_chunks(self, data):  # Define a method to generate chunks from input data
        for text in data:  # Iterate over each text in the input data
            for i in range(0, len(text), self.step_size):  # Iterate over the text with the step size
                max_size = min(self.chunk_size, len(text) - i)  # Calculate the maximum size of the chunk
                yield text[i:i+max_size]  # Return a chunk of text from position i to i+max_size

# Define a class for loading data from a directory
class DirectoryLoader:
    def __init__(self, directory, batch_size=512, chunker=DefaultChunker()):  # Initialize the loader with directory, batch size, and chunker
        self.directory = directory  # Store the directory path
        self.chunker = chunker  # Store the chunker
        self.batch_size = batch_size  # Store the batch size

    def load(self):  # Define a method to load data from files in the directory
        for root, dirs, files in os.walk(self.directory):  # Traverse the directory
            for file in files:  # Iterate over files in the directory
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:  # Open each file
                    print("Loading file: %s", os.path.join(root, file))  # Print the loading status
                    yield f.read()  # Read the contents of the file and yield it as a chunk

    def get_chunks(self):  # Define a method to get chunks of data
        return self.chunker.get_chunks(self.load())  # Get chunks of data using the chunker and loader

    def get_chunk_batches(self):  # Define a method to get batches of chunks
        chunks = []  # Initialize an empty list to store chunks
        for chunk in self.get_chunks():  # Iterate over chunks
            chunks.append(chunk)  # Append the current chunk to the list
            if len(chunks) == self.batch_size:  # If the batch size is reached
                yield chunks  # Yield the batch of chunks
                chunks = []  # Reset the list of chunks

        if len(chunks) > 0:  # If there are remaining chunks
            yield chunks  # Yield the remaining chunks as a batch

    def __iter__(self):  # Define the iterator method
        return self.get_chunk_batches()  # Return the generator for iterating over chunk batches
