import csv
import pandas as pd

def get_matching_rows(df, label_value):
    matching_indices = df[df["label"] == label_value].index.tolist()
    return [i for i in matching_indices]

output_csv = 'data/video_annotation/kinetics400/kinetics400_val_sample_num.csv'
df = pd.read_csv('data/video_annotation/kinetics400/val.csv', header=None, names=["path", "label"])
label_df = pd.read_csv('data/kinetics400_classes.csv', header=0, names=["id", "name"])
label_dict = label_df.set_index("id")["name"].to_dict()

with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["id", "name", "samples_num"])  # Write the header

    for i, label in label_dict.items():
        matching_rows = get_matching_rows(df, i)
        writer.writerow([i, label, matching_rows])
            
#archery 5 [925, 1175, 1525, 1716, 2003, 2071, 2210, 3116, 3322, 4522, 4864, 4991, 5814, 6129, 6195, 6207, 6591, 7122, 8363, 8564, 8682, 9138, 9410, 9420, 9617, 9774, 9868, 10064, 10437, 11182, 11995, 12055, 12065, 12449, 12888, 13019, 13314, 14884, 15054, 15552, 15759, 16057, 16331, 16571, 17008, 17344, 17888, 18058, 18096, 19453]
#hitting baseball [762, 799, 1073, 1214, 1283, 3062, 3293, 3520, 3890, 4317]# 4660, 5308, 5949, 6028, 6048, 6132, 6168, 6291]#, 6295, 7587, 7621, 8579, 8607, 8787, 9363, 9729, 9783, 9859, 9990, 10275, 10721, 11045, 11194, 12124, 12190, 12471, 12482, 13327, 13633, 13674, 14517, 14597, 16094, 16481, 16937, 17242, 17360, 17829, 18544, 19577]
#somersaulting [0, 55, 587, 1201, 1595, 1651, 2057, 2723, 2901, 3077, 3873, 3960, 4083, 4280, 5092, 5177, 5441, 5892, 6278, 7185, 8390, 8884, 10080, 10172, 11078, 11233, 11582, 11751, 11755, 12066, 12964, 13058, 13959, 14119, 14189, 14383, 14667, 15399, 15436, 15483, 15655, 16199, 16563, 16727, 16735, 17913, 18261, 18472, 18728, 18988]
#bending back 20 [272, 1778, 2253, 2314, 4195, 4196, 4249, 5866, 6010, 6079, 6094, 6120, 6453, 6457, 6697, 8333, 8650, 8876, 9412, 9706, 9726, 9744, 10571, 10693, 10755, 11431, 11497, 12104, 12484, 12510, 12968, 13141, 13981, 14122, 14281, 14438, 14526, 15179, 15659, 16363, 17000, 17083, 17150, 18074, 18532, 18958, 19480, 19589]