
import os
import json
from dotenv import load_dotenv
import openai
import conceptset_utils
from datetime import datetime
import torch

def load_prompt(prompt_type, dataset=None, prompt_dir = '/data/jongseo/project/PCBEAR/CBM_training/prompts'):
    filepath = os.path.join(prompt_dir, f"{prompt_type}.txt")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Prompt file '{filepath}' does not exist.")
    
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def get_feature_dict(classes,base_prompt):
    feature_dict = {}
    for i, label in enumerate(classes):
        feature_dict[label] = set()
        print("\n", i, label)
        
        for _ in range(2):
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": base_prompt.format(label)
                }],
                temperature=0.2,
                max_tokens=512,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            
            # clean up responses
            features = response["choices"][0]["message"]["content"]
            features = features.split("\n")
            
            # 각 줄마다 '- ', '• ' 같은 기호 제거하고 공백 정리
            cleaned = []
            for feat in features:
                feat = feat.strip().lstrip("-•").strip()
                if feat:
                    cleaned.append(feat)
            
            feature_dict[label].update(cleaned)
        
        feature_dict[label] = sorted(list(feature_dict[label]))
        print(f"✅ Features for '{label}':")
        for j, feat in enumerate(feature_dict[label]):
            print(f"  {j+1}. {feat}")
    gpt_name = '-'.join(response.model.split('-')[:3])
    return feature_dict,gpt_name

import argparse

parser = argparse.ArgumentParser(description='Settings for creating CBM')
parser.add_argument("--dataset", type=str, default="ViT-B/16", help="Which CLIP model to use")
parser.add_argument("--prompt_name", type=str, default="ViT-B/16", help="Which CLIP model to use")
parser.add_argument("--max_len", type=int, default=30, help="Which CLIP model to use")

def main(args):
    load_dotenv()
    openai.api_key = os.getenv('API_KEY')
    device = "cuda" if torch.cuda.is_available() else "cpu"



    dataset = args.dataset#"HAA100"

    # 사용 예시
    prompt_name = args.prompt_name#"object_ver1"
    base_prompt = load_prompt(prompt_name)



    cls_file = f'/data/jongseo/project/PCBEAR/dataset/{dataset}/class_list.txt'
    with open(cls_file, "r") as f:
        classes = f.read().split("\n")
        
        
    concept_dict,gpt_name = get_feature_dict(classes=classes,
                                        base_prompt=base_prompt,
                                        ) 

    #! save concept
        
    json_object = json.dumps(concept_dict, indent=4)
    with open("/data/jongseo/project/PCBEAR/CBM_training/GPT_concept_set/{}_{}_{}.json".format(dataset, prompt_name,gpt_name), "w") as outfile:
        outfile.write(json_object)
        
        """
    CLASS_SIM_CUTOFF: Concenpts with cos similarity higher than this to any class will be removed
    OTHER_SIM_CUTOFF: Concenpts with cos similarity higher than this to another concept will be removed
    MAX_LEN: max number of characters in a concept

    PRINT_PROB: what percentage of filtered concepts will be printed
    """

    CLASS_SIM_CUTOFF = 0.84
    OTHER_SIM_CUTOFF = 0.9
    # args.max_len = 30
    PRINT_PROB = 1



    save_name = "/data/jongseo/project/PCBEAR/CBM_training/GPT_concept_set/filterd_concept/{}/{}_{}_filtered.txt".format(dataset,dataset,prompt_name)
    import re
    concepts = set()
    for values in concept_dict.values():
        concepts.update(set(values))
    print(len(concepts))

    if args.max_len>0:
        print("## Remove_too_long")
        concepts = conceptset_utils.remove_too_long(concepts, args.max_len, PRINT_PROB)

    print("\n\n #!# Remove parentheses ")
    concepts = [re.sub(r"\s*\(.*?\)", "", c).strip() for c in concepts]

    print("\n\n #!# Filter_too_similar_to_cls ")
    concepts = conceptset_utils.filter_too_similar_to_cls(concepts, classes, CLASS_SIM_CUTOFF, device, PRINT_PROB)

    print("\n\n #!# Filter_too_similar ")
    concepts = conceptset_utils.filter_too_similar(concepts, OTHER_SIM_CUTOFF, device, PRINT_PROB)

    with open(save_name, "w") as f:
        f.write(concepts[0])
        for concept in concepts[1:]:
            f.write("\n" + concept)
            
        
        
        
if __name__=='__main__':
    args = parser.parse_args()
    start_time = datetime.now()
    main(args)
    end_time = datetime.now()
    print(f"🚀 Run time: {(end_time-start_time).total_seconds():.2f} seconds")  # 초 단위로 변환