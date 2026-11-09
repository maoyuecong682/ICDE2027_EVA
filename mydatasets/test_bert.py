import torch
from transformers import BertModel, BertTokenizer

# 替换为你刚刚的真实路径
local_bert_path = '/root/autodl-tmp/bert-base-uncased/models/google-bert--bert-base-uncased/snapshots/master'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"当前使用的设备: {device}")
print("正在加载 Tokenizer...")
try:
    tokenizer = BertTokenizer.from_pretrained(local_bert_path)
    print("✅ Tokenizer 加载成功！")
except Exception as e:
    print(f"❌ Tokenizer 加载失败: {e}")
    exit()

print("正在加载 BERT 模型...")
try:
    model = BertModel.from_pretrained(local_bert_path)
    model.to(device)
    model.eval()
    print("✅ 模型加载成功！")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    exit()

# 模拟你代码中的文本输入（比如一段 IMDB 的评论）
texts = ["This is a fantastic movie!", "I really hated this film, terrible."]

print("正在进行一次前向传播测试...")
try:
    # 模拟你代码中的 tokenizer 处理
    encoded_inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
    input_ids = encoded_inputs['input_ids'].to(device)
    attention_mask = encoded_inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
    # 获取和提取你代码里需要的 pooler_output
    pooled_outputs = outputs.pooler_output.cpu().numpy()

    print("✅ 前向传播成功！")
    print(f"输出特征矩阵的形状: {pooled_outputs.shape}")
    print("（如果形状是 (2, 768)，说明完美契合你的代码预期！）")

except Exception as e:
    print(f"❌ 前向传播失败: {e}")
