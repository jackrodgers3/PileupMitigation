import uproot

root_path = r"/depot/cms/private/users/gpaspala/WJetsToQQ_HT-800toInf/output_1.root"

file = uproot.open(root_path)
print(file.keys())

tree = file['Events;1']

print(tree.keys())