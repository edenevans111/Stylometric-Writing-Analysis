# this is copied from Cam's repo
turns_type = "wait" # Change to parag or wait

import re
import pandas as pd
import random

def split_item_on_wait(item):
    parts = re.split('(Wait|But\swait)',str(item))
    for i in range(len(parts)):
        if (i < len(parts)-1) and parts[i] in ['Wait','But wait']:
            parts[i+1] = parts[i]+parts[i+1]
            parts.pop(i)
    return [part for part in parts if part]

def split_item_on_parag(item):
    return [i for i in item.split('\n') if i.strip()]

df = pd.read_csv('reasoning_examples.csv')
sample_df = df.sample(n=1000)
if turns_type == "wait":
    sample_df['text'] = sample_df['text'].apply(split_item_on_wait)
elif turns_type == "parag":
    sample_df['text'] = sample_df['text'].apply(split_item_on_parag)
turns = []
for trace in sample_df['text']:
    turns.append(random.choice(trace))
turns_df = pd.DataFrame(turns, columns=['turn'])
turns_df.to_csv(f'sample_reasoning_turns_{turns_type}.csv', index=False)