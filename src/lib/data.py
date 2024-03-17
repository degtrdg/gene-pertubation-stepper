import json

with open('./protein_names.txt') as f:
    protein_names = f.read().splitlines()
    with open('./protein_names.js', 'w') as f:
        # make it a js list
        f.write('var protein_names = ' + json.dumps(protein_names) + ';')
