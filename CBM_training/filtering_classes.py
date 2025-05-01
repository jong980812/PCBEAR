import pandas as pd

with open('data/kinetics100_classes.txt', 'r') as file:
    kinetics100_classes = [line.strip() for line in file]

kinetics400_df = pd.read_csv('data/kinetics400_classes.csv')
# class_id_map = kinetics400_df[kinetics400_df['name'].isin(kinetics100_classes)]

# anno_df = pd.read_csv('data/video_annotation/kinetics400/train.csv', header=None, names=['video_path', 'class_id'])
# filtered_anno_df = anno_df[anno_df['class_id'].isin(class_id_map['id'])]

# filtered_anno_df.to_csv('data/video_annotation/kinetics100/train.csv', index=False)

# anno_df = pd.read_csv('data/video_annotation/kinetics400/val.csv', header=None, names=['video_path', 'class_id'])
# filtered_anno_df = anno_df[anno_df['class_id'].isin(class_id_map['id'])]

# filtered_anno_df.to_csv('data/video_annotation/kinetics100/val.csv', index=False)

new_class_id_map = {row['id']: i for i, name in enumerate(kinetics100_classes) for _, row in kinetics400_df.iterrows() if row['name'] == name}
train_df = pd.read_csv('data/video_annotation/kinetics100/train.csv')
train_df['class_id'] = train_df['class_id'].map(new_class_id_map).fillna(train_df['class_id'])
train_df.to_csv('data/video_annotation/kinetics100/train_k100.csv', index=False)

train_df = pd.read_csv('data/video_annotation/kinetics100/val.csv')
train_df['class_id'] = train_df['class_id'].map(new_class_id_map).fillna(train_df['class_id'])
train_df.to_csv('data/video_annotation/kinetics100/val_k100.csv', index=False)
