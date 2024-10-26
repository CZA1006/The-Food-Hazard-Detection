import json
import time
from tqdm import tqdm
import pandas as pd
from openai import OpenAI
import tiktoken


def get_prompt(data):
    """
    :param data: List[dict],keys: title,text,hazard-category,product-category,hazard,product
    :return: prompt
    """
    prompt = """You are a powerful classifier. This is a Food Hazard Detection Task. Given title and text, classify and extract only the following fields: **hazard-category, product-category, hazard, and product**.
Ensure:
- `hazard-category` MUST be one of: ['biological','foreign bodies','chemical','fraud','organoleptic aspects','allergens','packaging defect','other hazard','food additives and flavourings','migration'].
- `product-category` MUST be one of: ['meat, egg and dairy products', 'prepared dishes and snacks', 'cereals and bakery products', 'confectionery', 'ices and desserts', 'alcoholic beverages', 'fruits and vegetables', 'other food product / mixed', 'cocoa and cocoa preparations, coffee and tea', 'nuts, nut products and seeds', 'seafood', 'soups, broths, sauces and condiments', 'fats and oils', 'non-alcoholic beverages', 'food contact materials', 'dietetic foods, food supplements, fortified foods', 'herbs and spices', 'food additives and flavourings', 'sugars and syrups', 'honey and royal jelly', 'feed materials', 'pet feed'].

Return output in this exact JSON format:
{
  "hazard-category": "category",
  "product-category": "category",
  "hazard": "specific hazard",
  "product": "specific product"
}

Examples:
"""
    for i, example in enumerate(data):
        prompt += (f"example {i + 1}:\ntitle: {example['title']}\ntext: {example['text']}\n"
                   f'output: {{"hazard-category": "{example["hazard-category"]}",'
                   f'"product-category": "{example["product-category"]}",'
                   f'"hazard": "{example["hazard"]}","product": "{example["product"]}"}}\n\n')
    return prompt


def classify(client, prompt, title, text):
    prompt += f"title:{title}\ntext:{text}\noutput:"
    # print(prompt)
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # print(completion.choices[0].message.content)
    return completion.choices[0].message.content


def save_result(data_id, content):
    record = {"data_id": int(data_id), "content": content}

    # 将字典转换为 JSON 字符串，并追加写入文件，每行一个完整 JSON 字典
    with open('results.json', 'a', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False)
        f.write('\n')  # 换行符用于区分记录

def compute_token(text):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    token_count = len(tokens)
    print("Token 数量:", token_count)

if __name__ == '__main__':
    with open("api key.txt", "r") as f:
        api_key = f.read()
    client = OpenAI(api_key=api_key)
    df = pd.read_csv('incidents_train.csv')

    sample_df = df.sample(n=5)
    train_id = sample_df.iloc[:, 0].tolist()
    print(train_id)
    sample_dict = sample_df.to_dict(orient='records')
    prompt = get_prompt(sample_dict)
    compute_token(prompt)
    for i in tqdm(range(len(df))):
        if df.iloc[i, 0] not in train_id:
            content = classify(client, prompt, df.iloc[i]["title"], df.iloc[i]["text"])
            save_result(df.iloc[i, 0], content)
            time.sleep(0.5)

