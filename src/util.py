import os
import ndjson
import pandas as pd
import numpy as np


id_to_emotion = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}


def load_transform_sentiment(path):

    with open(path) as fin:
        episode_sentiment_dist = ndjson.load(fin)
        episode_tag = os.path.basename(path).replace('.ndjson', '')

    # flatten to List[List[float]] format
    emotion_matrix = []
    for doc in episode_sentiment_dist:
        flattened_doc = [emotion['score'] for emotion in doc[0]]
        emotion_matrix.append(flattened_doc)
        
    return emotion_matrix, episode_tag


def calc_ratio_of_emotions(emotion_matrix, episode_tag):

    mat = np.array(emotion_matrix)
    top_emotion_ids = np.argmax(mat, axis=1)

    top_emotion = [id_to_emotion[bit] for bit in top_emotion_ids]

    s = pd.Series(top_emotion)
    ratio = s.value_counts() / len(s) * 100

    ratio_row = pd.DataFrame(ratio).transpose()
    ratio_row['episode'] = int(episode_tag)
    
    return ratio_row