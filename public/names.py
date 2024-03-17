with open('./protein_names.txt', 'r') as f:
    names = f.read().splitlines()
    nf = open('./protein_names.js', 'w')
    nf.write('export const proteinNames = [')
    for name in names:
        nf.write("'" + name + "',")
    nf.write(']')
