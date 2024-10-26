import re
import time
import json
import numpy as np
from tqdm import tqdm
import pandas as pd
from openai import OpenAI
from sklearn.metrics import classification_report

def embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def generate_and_save_embeddings(text_list, file_name):
    embeddings = {}  # 存放所有嵌入
    for text in tqdm(text_list):
        embeddings[text] = embedding(text)  # 不断扩展嵌入数据字典
        # 每次覆盖写入 JSON 文件，保持最新的嵌入
        with open(file_name + '.json', 'w') as f:
            json.dump(embeddings, f)
        time.sleep(0.1)  # 添加 0.1 秒延迟


def load_results():
    with open('results.json', 'r', encoding='utf-8') as f:
        results = []
        for line in f:
            line = json.loads(line)
            nested_json_str = line["content"].replace("```json\n", "").replace("\n```", "")
            match = re.match(r'(\{.*\}),(\{.*\})', nested_json_str)
            if match:
                nested_json_str = re.match(r'(\{.*\}),', nested_json_str).group(1)
                # print(nested_json_str)
            # print(nested_json_str)
            nested_data = json.loads(nested_json_str)
            results.append({"data_id": line["data_id"], **nested_data})
    return results


def compute_similarity(embedding1, embedding2):
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))


# def get_top5_similar_embeddings(text, text_embeddings, extra_file_name):
#     if text in text_embeddings.keys():
#         embed = text_embeddings[text]
#     else:
#         with open(extra_file_name + '.json', 'r') as f:
#             extra_data = json.load(f)
#         if text in extra_data.keys():
#             embed = extra_data[text]
#         else:
#             embed = embedding(text)
#             extra_data[text] = embed
#             with open(extra_file_name + '.json', 'w') as f:
#                 json.dump(extra_data, f)
#                 time.sleep(0.1)
#     similarity = {}
#     for text_embed in text_embeddings.keys():
#         similarity[text_embed] = compute_similarity(text_embeddings[text_embed], embed)
#     sorted_similarity = sorted(similarity.items(), key=lambda x: x[1], reverse=True)[:5]
#     return sorted_similarity

def get_top5_similar_embeddings(text, text_embeddings, extra_data):
    if text in text_embeddings.keys():
        embed = text_embeddings[text]
    else:
        if text in extra_data.keys():
            embed = extra_data[text]
        else:
            embed = embedding(text)
            extra_data[text] = embed
            time.sleep(0.1)
    similarity = {}
    for text_embed in text_embeddings.keys():
        similarity[text_embed] = compute_similarity(text_embeddings[text_embed], embed)
    sorted_similarity = sorted(similarity.items(), key=lambda x: x[1], reverse=True)[:5]
    return sorted_similarity, extra_data


# def calculate_accuracy(df):
#     for t in ["hazard-category", "product-category", "hazard", "product"]:
#         for i in range(1, 6):
#             df[f"{t}_top{i}_name"] = None
#             df[f"{t}_top{i}_similarity"] = None
#     for i in tqdm(range(len(df))):
#         hazard_category = df.iloc[i]["hazard-category"]
#         product_category = df.iloc[i]["product-category"]
#         hazard = df.iloc[i]["hazard"]
#         product = df.iloc[i]["product"]
#         hazard_category_top5 = get_top5_similar_embeddings(hazard_category, hazard_category_embeddings,
#                                                            "extra_embeddings")
#         product_category_top5 = get_top5_similar_embeddings(product_category, product_category_embeddings,
#                                                             "extra_embeddings")
#         hazard_top5 = get_top5_similar_embeddings(hazard, hazard_embeddings, "extra_embeddings")
#         product_top5 = get_top5_similar_embeddings(product, product_embeddings, "extra_embeddings")
#         for j, (name, similarity) in enumerate(hazard_category_top5):
#             df.at[i, f"hazard_category_top{j + 1}_name"] = name
#             df.at[i, f"hazard_category_top{j + 1}_similarity"] = similarity
#         for j, (name, similarity) in enumerate(product_category_top5):
#             df.at[i, f"product_category_top{j + 1}_name"] = name
#             df.at[i, f"product_category_top{j + 1}_similarity"] = similarity
#         for j, (name, similarity) in enumerate(hazard_top5):
#             df.at[i, f"hazard_top{j + 1}_name"] = name
#             df.at[i, f"hazard_top{j + 1}_similarity"] = similarity
#         for j, (name, similarity) in enumerate(product_top5):
#             df.at[i, f"product_top{j + 1}_name"] = name
#             df.at[i, f"product_top{j + 1}_similarity"] = similarity
#     return df
def calculate_accuracy(df, extra_data):
    for t in ["hazard-category", "product-category", "hazard", "product"]:
        for i in range(1, 6):
            df[f"{t}_top{i}_name"] = None
            df[f"{t}_top{i}_similarity"] = None
    for i in tqdm(range(len(df))):
        hazard_category = df.iloc[i]["hazard-category"]
        product_category = df.iloc[i]["product-category"]
        hazard = df.iloc[i]["hazard"]
        product = df.iloc[i]["product"]
        hazard_category_top5, extra_data = get_top5_similar_embeddings(hazard_category, hazard_category_embeddings,
                                                                       extra_data)
        product_category_top5, extra_data = get_top5_similar_embeddings(product_category, product_category_embeddings,
                                                                        extra_data)
        hazard_top5, extra_data = get_top5_similar_embeddings(hazard, hazard_embeddings, extra_data)
        product_top5, extra_data = get_top5_similar_embeddings(product, product_embeddings, extra_data)
        for j, (name, similarity) in enumerate(hazard_category_top5):
            df.at[i, f"hazard-category_top{j + 1}_name"] = name
            df.at[i, f"hazard-category_top{j + 1}_similarity"] = similarity
        for j, (name, similarity) in enumerate(product_category_top5):
            df.at[i, f"product-category_top{j + 1}_name"] = name
            df.at[i, f"product-category_top{j + 1}_similarity"] = similarity
        for j, (name, similarity) in enumerate(hazard_top5):
            df.at[i, f"hazard_top{j + 1}_name"] = name
            df.at[i, f"hazard_top{j + 1}_similarity"] = similarity
        for j, (name, similarity) in enumerate(product_top5):
            df.at[i, f"product_top{j + 1}_name"] = name
            df.at[i, f"product_top{j + 1}_similarity"] = similarity
        if i % 100 == 0:
            with open("extra_embeddings" + '.json', 'w') as f:
                print(len(extra_data.keys()))
                json.dump(extra_data, f)
            with open("extra_embeddings" + '.json', 'r') as f:
                extra_data = json.load(f)
    return df


if __name__ == '__main__':
    with open("api key.txt", "r") as f:
        api_key = f.read()
    client = OpenAI(api_key=api_key)
    df = pd.read_csv('incidents_train.csv')
    hazard_category_list = df['hazard-category'].unique()
    product_category_list = df['product-category'].unique()
    hazard_list = df['hazard'].unique()
    product_list = df['product'].unique()
    # generate_and_save_embeddings(hazard_category_list, 'hazard_category_embeddings')
    # generate_and_save_embeddings(product_category_list, 'product_category_embeddings')
    # generate_and_save_embeddings(hazard_list, 'hazard_embeddings')
    # generate_and_save_embeddings(product_list, 'product_embeddings')
    with open('hazard_category_embeddings.json', 'r') as f:
        hazard_category_embeddings = json.load(f)
    with open('product_category_embeddings.json', 'r') as f:
        product_category_embeddings = json.load(f)
    with open('hazard_embeddings.json', 'r') as f:
        hazard_embeddings = json.load(f)
    with open('product_embeddings.json', 'r') as f:
        product_embeddings = json.load(f)

    with open("extra_embeddings" + '.json', 'r') as f:
        extra_data = json.load(f)
    results = load_results()
    results = pd.DataFrame(results)
    results = calculate_accuracy(results,extra_data)
    results.to_csv('results.csv')
    results = pd.read_csv('results.csv')

    hazard_category_label, hazard_category_predict = [], []
    product_category_label, product_category_predict = [], []
    hazard_label, hazard_predict = [], []
    product_label, product_predict = [], []

    for data_id in results["data_id"]:
        labels = df.loc[df.iloc[:,0] == data_id]
        hazard_category_label.append(labels["hazard-category"].values[0])
        product_category_label.append(labels["product-category"].values[0])
        hazard_label.append(labels["hazard"].values[0])
        product_label.append(labels["product"].values[0])

        predicts = results.loc[results["data_id"] == data_id]
        hazard_category_predict.append(predicts["hazard-category_top1_name"].values[0])
        product_category_predict.append(predicts["product-category_top1_name"].values[0])
        hazard_predict.append(predicts["hazard_top1_name"].values[0])
        product_predict.append(predicts["product_top1_name"].values[0])

    print(classification_report(hazard_category_label,hazard_category_predict))
    print(classification_report(product_category_label,product_category_predict))
    print(classification_report(hazard_label,hazard_predict))
    print(classification_report(product_label,product_predict))


