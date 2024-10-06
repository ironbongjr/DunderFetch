import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

df = pd.read_excel('the-office-lines.xlsx')
print("file loaded")


df = df.drop('id', axis=1) # drop id column
df = df[df['deleted'] == 0] # select only undeleted lines from the script
print("id dropped")



df['Scene_ID'] = df['season'].astype(str) + '-' + df['episode'].astype(str) + '-' + df['scene'].astype(str)
grouped = df.groupby('Scene_ID')

def combine_scene_lines(group):
    """Combine lines for each scene into a single chunk of text."""
    dialogues = group['line_text'].tolist()
    speakers = group['speaker'].tolist()
    scene_chunk = ' '.join([f"{speaker}: {line}" for speaker, line in zip(speakers, dialogues)])
    return scene_chunk

# Create a new DataFrame with combined scene chunks
df_chunks = grouped.apply(combine_scene_lines).reset_index()
df_chunks.columns = ['Scene_ID', 'Scene_Chunk']


def extract_scene_info(scene_id):
    parts = scene_id.split('-')
    return f"Season {parts[0]} Episode {parts[1]} Scene {parts[2]}"

print("\n extracting scene info")
df_chunks['scene_info'] = df_chunks['Scene_ID'].apply(extract_scene_info)


df_chunks['combined_text'] = df_chunks['scene_info'] + " | " + df_chunks['Scene_Chunk']


file_path = os.path.join(os.getcwd(), "chunks.pkl")
df_chunks.to_pickle(file_path)




# Load pre-trained Sentence Transformer
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert chunks to a list and generate embeddings
print("starting embedding")
scene_embeddings = embedding_model.encode(df_chunks['combined_text'].tolist())
# print("embedding done, saved")
np.save('scene_embeddings.npy', scene_embeddings)

