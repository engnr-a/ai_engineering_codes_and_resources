import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# new instance of chroma client to interact with chroma db
client = chromadb.Client(Settings(anonymized_telemetry=False))

# collection name
collection_name = "grocery_collection"

def perform_similarity_search(collection, query_terms):
    try:
        results  = collection.query(
            query_texts=query_terms,
            n_results = 3
        )
        if not results or not results['ids'] or len(results['ids'][0]) == 0:
            # Log a message indicating that no similar documents were found for the query term
            print(f'No documents found similar to "{query_terms}"')
            return
    
        # print(f"Query result for {query_terms}")
        # print(results)
        """
        Query result for ['apple']
        {'ids': [['food_13', 'food_1', 'food_14']], 'embeddings': None, 'documents': [['golden apple', 'fresh red apples', 'red fruit']], 'uris': None, 'included': ['metadatas', 'documents', 'distances'], 'data': None, 'metadatas': [[{'source': 'texts', 'category': 'food items'}, {'category': 'food items', 'source': 'texts'}, {'category': 'food items', 'source': 'texts'}]], 'distances': [[0.3824649453163147, 0.480892539024353, 0.5965152978897095]]}
        """
        for i in range(min(3, len(results['ids'][0]))):
            doc_id = results['ids'][0][i]  # Get ID from 'ids' array
            score = results['distances'][0][i]  # Get score from 'distances' array
            # Retrieve text data from the results
            text = results['documents'][0][i]
            if not text:
                print(f' - ID: {doc_id}, Text: "Text not available", Score: {score:.4f}')
            else:
                print(f' - ID: {doc_id}, Text: "{text}", Score: {score:.4f}')
        
        return results
    except Exception as error:
        print(f"Error in similarity search: {error}")



def main(query_terms:str):
    try:
        # creat collection using the client
        collection = client.create_collection(
            name=collection_name,
            metadata={"description":"A collection for storing grocery data"},
            configuration={
                "hnsw":{"space":"cosine"},
                "embedding_function":ef
            },  
        )

        # Array of grocery-related text items
        texts = [
            'fresh red apples',
            'organic bananas',
            'ripe mangoes',
            'whole wheat bread',
            'farm-fresh eggs',
            'natural yogurt',
            'frozen vegetables',
            'grass-fed beef',
            'free-range chicken',
            'fresh salmon fillet',
            'aromatic coffee beans',
            'pure honey',
            'golden apple',
            'red fruit'
        ]
        # create unique ides per document
        ids =[f"food_{index+1}" for index,_ in enumerate(texts)]
        collection.add(
            documents=texts,
            ids=ids,
            metadatas=[{"source":"texts", "category":"food items"} for item in texts]
        )
        all_items = collection.get()
        print("Collection contents:")
        #print(all_items)
        """
        {'ids': ['food_1', 'food_2', 'food_3', 'food_4', 'food_5', 'food_6', 'food_7', 'food_8', 'food_9', 'food_10', 'food_11', 'food_12', 'food_13', 'food_14'], 'embeddings': None, 'documents': ['fresh red apples', 'organic bananas', 'ripe mangoes', 'whole wheat bread', 'farm-fresh eggs', 'natural yogurt', 'frozen vegetables', 'grass-fed beef', 'free-range chicken', 'fresh salmon fillet', 'aromatic coffee beans', 'pure honey', 'golden apple', 'red fruit'], 'uris': None, 'included': ['metadatas', 'documents'], 'data': None, 'metadatas': [{'source': 'texts', 'category': 'food items'}, {'category': 'food items', 'source': 'texts'}, {'source': 'texts', 'category': 'food items'}, {'source': 'texts', 'category': 'food items'}, {'category': 'food items', 'source': 'texts'}, {'source': 'texts', 'category': 'food items'}, {'source': 'texts', 'category': 'food items'}, {'category': 'food items', 'source': 'texts'}, {'source': 'texts', 'category': 'food items'}, {'category': 'food items', 'source': 'texts'}, {'category': 'food items', 'source': 'texts'}, {'source': 'texts', 'category': 'food items'}, {'source': 'texts', 'category': 'food items'}, {'source': 'texts', 'category': 'food items'}]}
        """
        results = perform_similarity_search(collection, query_terms)
        return results

       

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    search_query = ["apple"]
    main(search_query)