import random
import nltk
import pandas as pd
from nltk.corpus import wordnet
nltk.download('wordnet')
nltk.download('omw-1.4')

# 同义词替换
def synonym_replacement(words, n):
    new_words = words.copy()
    random_word_list = list(set(words))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:  # 达到替换上限
            break
    return new_words

# 获取同义词
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name()
            if synonym != word:  # 同义词不能是词本身
                synonyms.add(synonym)
    return list(synonyms)

# 随机插入
def random_insertion(words, n):
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    return new_words

def add_word(new_words):
    synonyms = []
    counter = 0
    while len(synonyms) < 1 and counter < 10:
        random_word = new_words[random.randint(0, len(new_words)-1)]
        synonyms = get_synonyms(random_word)
        counter += 1
    if len(synonyms) >= 1:
        synonym = random.choice(synonyms)
        random_idx = random.randint(0, len(new_words)-1)
        new_words.insert(random_idx, synonym)

# 随机交换
def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words

def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
        counter += 1
        if counter > 3:  # 避免死循环
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words

# 随机删除
def random_deletion(words, p):
    if len(words) == 1:
        return words  # 只有一个词时不删除
    new_words = []
    for word in words:
        if random.uniform(0, 1) > p:
            new_words.append(word)
    if len(new_words) == 0:  # 防止全部删除
        return [random.choice(words)]
    return new_words

def eda(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, prob_sr=0.5, prob_ri=0.5, prob_rs=0.5, prob_rd=0.5):
    """
    按概率对句子进行增强
    - prob_sr, prob_ri, prob_rs, prob_rd: 分别是每种增强操作的执行概率
    """
    words = sentence.split()
    num_words = len(words)

    # 1. 同义词替换
    if random.random() < prob_sr:
        n_sr = max(1, int(alpha_sr * num_words))
        words = synonym_replacement(words, n_sr)

    # 2. 随机插入
    if random.random() < prob_ri:
        n_ri = max(1, int(alpha_ri * num_words))
        words = random_insertion(words, n_ri)

    # 3. 随机交换
    if random.random() < prob_rs:
        n_rs = max(1, int(alpha_rs * num_words))
        words = random_swap(words, n_rs)

    # 4. 随机删除
    if random.random() < prob_rd:
        words = random_deletion(words, p_rd)

    # 返回增强后的句子
    return ' '.join(words)

def augument_text(data,target):

    # 计算每个标签的样本数量
    label_counts = data[target].value_counts()
    threshold = len(data) * 0.1  # 10% 的阈值
    small_labels = label_counts[label_counts < threshold].index.tolist()

    augmented_data = []

    # 针对小样本标签进行增强
    for label in small_labels:
        label_data = data[data[target] == label]
        cnt = 0
        for _, row in label_data.iterrows():
            text = row['text']  # 假设文本列名为 'text'
            title = row['title']
            # 进行数据增强（可以根据需要多次增强）
            augmented_title = eda(title)  # 增强 3 个样本
            augmented_text = eda(text)
            augmented_data.append({'title':augmented_title,'text': augmented_text, target: label})
            cnt += 1
            if label_counts[label]+ cnt >= threshold:
                break

    augmented_df = pd.DataFrame(augmented_data)
    # 合并原始数据和增强数据
    final_data = pd.concat([data, augmented_df], ignore_index=True)
    return final_data

# 保存增强后的数据
# final_data.to_csv(f'incidents_train_augmented_{target}.csv', index=False)