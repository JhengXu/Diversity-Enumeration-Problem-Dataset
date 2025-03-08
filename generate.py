import os
# 批量设置环境变量
os.environ['HF_DATASETS_CACHE'] = '/root/autodl-tmp/.catch'
os.environ['HF_CACHE_DIR'] = '/root/autodl-tmp/.catch'
os.environ['HF_HOME'] = '/root/autodl-tmp/.catch/huggingface'
os.environ['HF_HUB_CACHE'] = '/root/autodl-tmp/.catch/huggingface/hub'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_TOKEN'] = 'your HF_TOKEN'
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import random
from tqdm import tqdm
import torch
from accelerate import Accelerator
# Load model directly
from transformers import AutoModelForCausalLM

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

torch.cuda.empty_cache()
accelerator = Accelerator()
accelerator.free_memory()

try:
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", torch_dtype=torch.float16, trust_remote_code=True).to(device)
    
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer) # 直接在 pipeline 中指定 device
    print("模型和分词器加载成功")
except Exception as e:
    print(f"模型加载失败: {str(e)}")
    exit(1)


# 配置生成参数
question_prompts = [
    # 核心列举型模板
    "Generate a clear list-style question requiring enumeration of items with specific attributes:",
    "Create a challenge that asks to compile words/phrases matching defined criteria:",
    
    # 结构化模板
    "Produce a question asking to list at least 5 items in [category] with [characteristic]:",
    "Formulate a question requiring enumeration of [number] examples of [type] from [domain]:",
    
    # 多条件模板
    "Compose a question that requires listing items meeting both [condition A] and [condition B]:",
    "Design a challenge to find items matching [primary feature] while excluding [secondary feature]:",
    
    # 领域限定模板  
    "Create a question asking to list [type] of [concept] in [specific field/domain]:",
    "Generate a history-based enumeration question about [historical period] requiring [item type]:",
    
    # 游戏化模板
    "Create a timed challenge to list as many [category] items with [attribute] as possible:",
    "Produce a competition-style question about finding unique [items] matching [rule]:"
]


dataset = []
max_attempts = 6000  # 最大尝试次数（预留缓冲）

with tqdm(total=5000, desc="生成问题集") as pbar:
    attempts = 0
    while len(dataset) < 5000 and attempts < max_attempts:
        attempts += 1
        seed_prompt = random.choice(question_prompts)
        
        try:
            # 生成完整问题
            response = generator(
                seed_prompt,
                pad_token_id=generator.tokenizer.eos_token_id,  # 显式设置
                max_new_tokens=50,  # 控制问题长度
                temperature=0.85 + random.random()*0.15,  # 更高的创造性
                top_p=0.95,
                repetition_penalty=1.2,
                do_sample=True,
                num_return_sequences=1
            )
            
            # 处理生成的文本
            raw_question = response[0]['generated_text'].strip()
            
           # 修改后的处理流程
            question = raw_question.replace(seed_prompt, "").strip()
            question = question.split("\n")[0].strip()  # 取第一行
            question = question.split("?")[0].strip() + "?"  # 确保以问号结尾

            # 新验证条件（更通用的列举问题检测）
            valid_conditions = (
                20 <= len(question) <= 150 and  # 调整长度范围
                question.endswith("?") and  # 必须是疑问句
                any(keyword in question.lower() for keyword in [
                    "list", "name", "identify","write down","what"
                ]) and  # 包含关键动词
                any(condition_word in question.lower() for condition_word in [
                    "with", "that", "based on", "according to",
                    "in", "from", "under", "requiring"
                ])  # 包含约束条件
            )

            if valid_conditions:
                # 标准化动态参数表示（可选）
                question = question.replace("a specific", "{param}")
                question = question.replace("certain", "{param}")
                question = question.replace("given", "{param}")
                
                dataset.append(question)
                print(f"\n[新生成] {question}")
                pbar.update(1)
                
        except Exception as e:
            continue

# 最终清洗和去重
final_dataset = list(set(dataset))[:5000]  # 确保最终数量

# 保存前打印统计信息
print("\n" + "="*40)
print(f"成功生成 {len(final_dataset)} 个唯一问题")
print("="*40 + "\n")

# 打印前20个示例
print("问题示例：")
for i, q in enumerate(final_dataset[:20], 1):
    print(f"{i}. {q}")

# 保存数据集
with open("qa_dataset.json", "w", encoding="utf-8") as f:
    json.dump(final_dataset, f, ensure_ascii=False, indent=2)  # 修正应保存final_dataset

print("\n数据集已保存为 qa_dataset.json！")