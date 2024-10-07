from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-mpnet-base-v2')

# Example sentences
sentences = [
    "What does the white house look like?",
    "Can anyone describe the insides of the white house?",
    "What is your age?"
]

# Encode the sentences into embeddings
embeddings = model.encode(sentences)

# Compute cosine similarity matrix
cosine_sim_matrix = cosine_similarity(embeddings)

# Set a threshold to group similar sentences (adjust threshold as needed)
similarity_threshold = 0.7

# Group sentences based on cosine similarity and store the scores
groups = []
used = set()

for i in range(len(sentences)):
    if i not in used:
        group = [sentences[i]]
        used.add(i)
        for j in range(i + 1, len(sentences)):
            if j not in used and cosine_sim_matrix[i][j] > similarity_threshold:
                group.append(sentences[j])
                used.add(j)
        groups.append(group)

# Display the grouped sentences with similarity scores
for idx, group in enumerate(groups):
    print(f"Group {idx+1}:")
    for sentence in group:
        print(f"  - Sentence: '{sentence}'")

# Print the cosine similarity matrix for reference
print("\nCosine Similarity Matrix:")
print(np.round(cosine_sim_matrix, 3))
