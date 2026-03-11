from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline
from deep_translator import GoogleTranslator
from langdetect import detect

# Load dataset
loader = TextLoader("schemes.txt", encoding="utf-8")
documents = loader.load()

# Split documents
text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector database
db = FAISS.from_documents(docs, embeddings)

# Load FLAN-T5 model
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    device=-1
)

print("\nSmart Panchayat Assistant Ready!")
print("Type 'exit' to quit.\n")

# similarity threshold
RELEVANCE_THRESHOLD = 1.0

while True:

    query = input("Ask your question: ")

    if query.lower() == "exit":
        break

    # Detect user language
    user_lang = detect(query)

    if user_lang not in ["en", "te", "hi"]:
        user_lang = "en"

    # Translate query to English
    query_en = GoogleTranslator(source='auto', target='en').translate(query)

    # Retrieve relevant documents with score
    results = db.similarity_search_with_score(query_en, k=3)

    # Check relevance
    best_score = results[0][1]

    if best_score > RELEVANCE_THRESHOLD:
        message = "Please ask a question related to government schemes."

        if user_lang != "en":
            message = GoogleTranslator(source='en', target=user_lang).translate(message)

        print("\nAnswer:\n")
        print(message)
        print("\n-----------------------------\n")
        continue

    # Extract context
    context = " ".join([doc.page_content for doc, score in results])

    prompt = f"""
You are a helpful assistant explaining Indian government schemes.

Use the context below to answer the question clearly.
Provide a complete explanation in 1–2 sentences.

Context:
{context}

Question:
{query_en}

Answer:
"""

    # Generate answer
    response = generator(prompt, max_new_tokens=200)

    answer = response[0]["generated_text"]

    # Translate answer back
    if user_lang == "en":
        final_answer = answer
    else:
        final_answer = GoogleTranslator(source='en', target=user_lang).translate(answer)

    print("\nAnswer:\n")
    print(final_answer)
    print("\n-----------------------------\n")