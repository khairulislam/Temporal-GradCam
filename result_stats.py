import json
import pandas as pd

iterations = 5
root_folder = 'OASIS_2D/results'
model = 'ViT'

results = None

for itr in range(1, iterations+1):
    with open(f'{root_folder}/{model}/{itr}/test.json') as f:
        data = json.load(f)
        print(data)
        if results is None:
            results = {}
            for k, v in data.items():
                results[k] = [v]
        else:
            for k, v in data.items():
                results[k].append(v)

results = pd.DataFrame(results)         
print(results.mean(axis=0))
results.to_csv(f'{root_folder}/{model}/results.csv', index=False)