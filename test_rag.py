
from tools.rag import (
    build_index,
    retrieve,
    answer_question
)


store = build_index(force=True)

print(store["message"])

print("Total chunks:", len(store["chunks"]))

print("FAISS vectors:", store["index"].ntotal)


question = "What information is available in the knowledge documents?"

results = retrieve(
    question,
    store["index"],
    store["chunks"]
)

print("\nRETRIEVAL RESULTS:")

for result in results:

    print("\nSource:", result["source"])
    print("Score:", round(result["score"], 3))
    print("Text:", result["text"][:300])


print("\nCRAG ANSWER:")

print(answer_question(question))
