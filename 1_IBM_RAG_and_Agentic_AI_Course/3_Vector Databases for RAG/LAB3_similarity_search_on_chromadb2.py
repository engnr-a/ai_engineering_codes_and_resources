import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings

# Defining a list of employee dictionaries
# Each dictionary represents an individual employee with comprehensive information
employees = [
            {
                "id": "employee_1",
                "name": "John Doe",
                "experience": 5,
                "department": "Engineering",
                "role": "Software Engineer",
                "skills": "Python, JavaScript, React, Node.js, databases",
                "location": "New York",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_2",
                "name": "Jane Smith",
                "experience": 8,
                "department": "Marketing",
                "role": "Marketing Manager",
                "skills": "Digital marketing, SEO, content strategy, analytics, social media",
                "location": "Los Angeles",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_3",
                "name": "Alice Johnson",
                "experience": 3,
                "department": "HR",
                "role": "HR Coordinator",
                "skills": "Recruitment, employee relations, HR policies, training programs",
                "location": "Chicago",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_4",
                "name": "Michael Brown",
                "experience": 12,
                "department": "Engineering",
                "role": "Senior Software Engineer",
                "skills": "Java, Spring Boot, microservices, cloud architecture, DevOps",
                "location": "San Francisco",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_5",
                "name": "Emily Wilson",
                "experience": 2,
                "department": "Marketing",
                "role": "Marketing Assistant",
                "skills": "Content creation, email marketing, market research, social media management",
                "location": "Austin",
                "employment_type": "Part-time"
            },
            {
                "id": "employee_6",
                "name": "David Lee",
                "experience": 15,
                "department": "Engineering",
                "role": "Engineering Manager",
                "skills": "Team leadership, project management, software architecture, mentoring",
                "location": "Seattle",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_7",
                "name": "Sarah Clark",
                "experience": 8,
                "department": "HR",
                "role": "HR Manager",
                "skills": "Performance management, compensation planning, policy development, conflict resolution",
                "location": "Boston",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_8",
                "name": "Chris Evans",
                "experience": 20,
                "department": "Engineering",
                "role": "Senior Architect",
                "skills": "System design, distributed systems, cloud platforms, technical strategy",
                "location": "New York",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_9",
                "name": "Jessica Taylor",
                "experience": 4,
                "department": "Marketing",
                "role": "Marketing Specialist",
                "skills": "Brand management, advertising campaigns, customer analytics, creative strategy",
                "location": "Miami",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_10",
                "name": "Alex Rodriguez",
                "experience": 18,
                "department": "Engineering",
                "role": "Lead Software Engineer",
                "skills": "Full-stack development, React, Python, machine learning, data science",
                "location": "Denver",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_11",
                "name": "Hannah White",
                "experience": 6,
                "department": "HR",
                "role": "HR Business Partner",
                "skills": "Strategic HR, organizational development, change management, employee engagement",
                "location": "Portland",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_12",
                "name": "Kevin Martinez",
                "experience": 10,
                "department": "Engineering",
                "role": "DevOps Engineer",
                "skills": "Docker, Kubernetes, AWS, CI/CD pipelines, infrastructure automation",
                "location": "Phoenix",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_13",
                "name": "Rachel Brown",
                "experience": 7,
                "department": "Marketing",
                "role": "Marketing Director",
                "skills": "Strategic marketing, team leadership, budget management, campaign optimization",
                "location": "Atlanta",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_14",
                "name": "Matthew Garcia",
                "experience": 3,
                "department": "Engineering",
                "role": "Junior Software Engineer",
                "skills": "JavaScript, HTML/CSS, basic backend development, learning frameworks",
                "location": "Dallas",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_15",
                "name": "Olivia Moore",
                "experience": 12,
                "department": "Engineering",
                "role": "Principal Engineer",
                "skills": "Technical leadership, system architecture, performance optimization, mentoring",
                "location": "San Francisco",
                "employment_type": "Full-time"
            },
        ]


employee_documents= []
for employee in employees:
    document = f"{employee['role']} with {employee['experience']} years of experience in {employee['department']}. "
    document += f"Skills: {employee['skills']}. Located in {employee['location']}. "
    document += f"Employment type: {employee['employment_type']}."
    employee_documents.append(document)


ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# new instance of chroma client to interact with chroma db
client = chromadb.Client(Settings(anonymized_telemetry=False))

# collection name
collection_name = "employee_collection"

def perform_search_by_metadata(collection, key_value_pair):
    return collection.get(
            where=key_value_pair, limit=5
    )

def perform_combined_search(collection, query_text, where_filter):
    """
    Performs a query using both a semantic query and a metadata filter.
    """
    print(f"\nSearching for '{query_text}' where {where_filter}")
    try:
        results = collection.query(
            query_texts=[query_text],  # The 'data' (semantic) part
            where=where_filter,        # The 'metadata' (strict) part
            n_results=3
        )
        return results
    except Exception as e:
        print(f"An error occurred during the combined search: {e}")
        return None

def perform_search_by_document(collection, query_text):
    try:
        results = collection.query(
            query_texts = query_text if isinstance(query_text, list) else [query_text],
            n_results=3
        )
        if results:
            print("Search successful!")
        return results
    except Exception as error:
        print(f"Error in advanced search: {error}")

# Defining a function named 'main'
# This function is used to encapsulate the main operations for creating collections,
# generating embeddings, and performing similarity search
def main(filter_by, query_text=None, where_filter=None):
    try:
        # Creating a collection using the ChromaClient instance
        # The 'create_collection' method creates a new collection with the specified configuration
        collection = client.create_collection(
            # Specifying the name of the collection to be created
            name=collection_name,
            # Adding metadata to describe the collection
            metadata={"description": "A collection for storing employee data"},
            # Configuring the collection with cosine distance and embedding function
            configuration={
                "hnsw": {"space": "cosine"},
                "embedding_function": ef
            }
        )
        print(f"Collection created: {collection.name}")
        collection.add(
                # Extracting employee IDs to be used as unique identifiers for each record
                ids=[employee["id"] for employee in employees],
                # Using the comprehensive text documents we created
                documents=employee_documents,
                # Adding comprehensive metadata for filtering and search
                metadatas=[{
                    "name": employee["name"],
                    "department": employee["department"],
                    "role": employee["role"],
                    "experience": employee["experience"],
                    "location": employee["location"],
                    "employment_type": employee["employment_type"]
                } for employee in employees]
            )
        
        all_items = collection.get()
        # Logging the retrieved items to the console for inspection or debugging
        print("Collection contents:")
        print(f"Number of documents: {len(all_items['documents'])}")


        ### 
        if filter_by == "metadata":
            if not where_filter:
                print("Error: A where_filter is required for metadata search.")
                return None
            return perform_search_by_metadata(collection, where_filter)
        
        elif filter_by=="data":
            return perform_search_by_document(collection, query_text)
        elif filter_by =="combined":
            pass
            return perform_combined_search(collection, query_text, where_filter)
        else: 
            print("Invalid filtering strategry")
            return

    except Exception as error:
        # Catching and handling any errors that occur within the 'try' block
        # Logs the error message to the console for debugging purposes
        print(f"Error: {error}")

if __name__ == "__main__":
    ##################################################################################################
    ### Example 1: Search for Python developers
    # print("\n1. Searching for Python developers:")
    # query_text = "Python developer with web development experience"
    #filter_by="data"
    # results = main(filter_by, query_text=query_text)

    # for i, (doc_id, document, distance) in enumerate(zip( results['ids'][0], results['documents'][0], results['distances'][0])):
    #     metadata = results['metadatas'][0][i]
    #     print(f"  {i+1}. {metadata['name']} ({doc_id}) - Distance: {distance:.4f}")
    #     print(f"     Role: {metadata['role']}, Department: {metadata['department']}")
    #     print(f"     Document: {document[:100]}...")

    """
     1. John Doe (employee_1) - Distance: 0.5156
        Role: Software Engineer, Department: Engineering
        Document: Software Engineer with 5 years of experience in Engineering. Skills: Python, JavaScript, React, Node...
    2. Matthew Garcia (employee_14) - Distance: 0.5724
        Role: Junior Software Engineer, Department: Engineering
        Document: Junior Software Engineer with 3 years of experience in Engineering. Skills: JavaScript, HTML/CSS, ba...
    3. Alex Rodriguez (employee_10) - Distance: 0.5967
        Role: Lead Software Engineer, Department: Engineering
        Document: Lead Software Engineer with 18 years of experience in Engineering. Skills: Full-stack development, R...
    """
    ##################################################################################################
    print("\n=== Metadata Filtering Examples ===")
    # Example 2: Filter by department
    print("\n2. Finding all Engineering employees:")

    #########################
    ###!! --FURTHER EXAMPLES OF WHERE FILTER
    #where_filter = dict({"department": "Engineering"})
    #where_filter = {"experience": {"$gte": 10}} # Finding employees with 10+ years experience
    #where_filter={"location": {"$in": ["San Francisco", "Los Angeles"]}} #Filter by location
    #############################################
    #filter_by ="metadata"
    #results = main(filter_by, where_filter=where_filter)
    # print("==================")
    # print(results)
    # """
    # {'ids': ['employee_1', 'employee_4', 'employee_6', 'employee_8', 'employee_10'], 'embeddings': None, 'documents': ['Software Engineer with 5 years of experience in Engineering. Skills: Python, JavaScript, React, Node.js, databases. Located in New York. Employment type: Full-time.', 'Senior Software Engineer with 12 years of experience in Engineering. Skills: Java, Spring Boot, microservices, cloud architecture, DevOps. Located in San Francisco. Employment type: Full-time.', 'Engineering Manager with 15 years of experience in Engineering. Skills: Team leadership, project management, software architecture, mentoring. Located in Seattle. Employment type: Full-time.', 'Senior Architect with 20 years of experience in Engineering. Skills: System design, distributed systems, cloud platforms, technical strategy. Located in New York. Employment type: Full-time.', 'Lead Software Engineer with 18 years of experience in Engineering. Skills: Full-stack development, React, Python, machine learning, data science. Located in Denver. Employment type: Full-time.'], 'uris': None, 'included': ['metadatas', 'documents'], 'data': None, 'metadatas': [{'role': 'Software Engineer', 'name': 'John Doe', 'experience': 5, 'department': 'Engineering', 'location': 'New York', 'employment_type': 'Full-time'}, {'name': 'Michael Brown', 'role': 'Senior Software Engineer', 'employment_type': 'Full-time', 'location': 'San Francisco', 'experience': 12, 'department': 'Engineering'}, {'department': 'Engineering', 'experience': 15, 'location': 'Seattle', 'role': 'Engineering Manager', 'name': 'David Lee', 'employment_type': 'Full-time'}, {'role': 'Senior Architect', 'name': 'Chris Evans', 'department': 'Engineering', 'experience': 20, 'location': 'New York', 'employment_type': 'Full-time'}, {'experience': 18, 'name': 'Alex Rodriguez', 'department': 'Engineering', 'location': 'Denver', 'role': 'Lead Software Engineer', 'employment_type': 'Full-time'}]}
    # """
    # print("==================")
    # print(f"Found {len(results['ids'])} Engineering employees:")
    # for i, doc_id in enumerate(results['ids']):
    #     metadata = results['metadatas'][i]
    #     print(f"  - {metadata['name']}: {metadata['role']} ({metadata['experience']} years)")

    ##################################################################################################
    print("\n=== Metadata and Data/Document Filtering Examples ===")
    # Example 3: Filter by department
    print("\n3. Finding senior Python developers in major tech cities:")

    #########################
    filter_by ="combined"
    query_text = "senior Python developer full-stack"
    where_filter={
        "$and":[
            {"experience":{"$gte":8}},
            {"location": {"$in":["New York", "Seattle"]}}
        ]
    }
    #############################################
    filter_by ="metadata"
    results = main(filter_by, query_text=query_text,where_filter=where_filter)
    print("==================")
    print(results)