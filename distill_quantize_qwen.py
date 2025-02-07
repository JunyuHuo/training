import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    AutoConfig,
)
from datasets import load_dataset, concatenate_datasets, DatasetDict


##########################################
# 1. 数据加载与预处理
##########################################

def prepare_multitask_data(task, example):
    """
    针对不同任务，统一转换为文本格式：
    
    [A] 问答任务（SQuAD、Natural Questions、MS MARCO）:
       Format:
         Task: Question Answering
         Question: <question>
         Context: <context>
         Answer: <answer>
    
    [B] API 监测任务：  
       (1) AG News:
         Task: API Monitoring (Switch API if needed)
         Log: <text>
         Status: <label>
       (2) DBpedia:
         Task: API Monitoring (Switch API if needed)
         Log: <content>
         Status: <label>
       (3) MRPC:
         Task: API Monitoring - Log Similarity
         Log1: <sentence1>
         Log2: <sentence2>
         Should switch API? <yes/no>  (设 label==1 为 yes，否则 no)
    """
    if task in ["squad", "natural_questions", "ms_marco"]:
        question = example.get("question", "")
        context = example.get("context", "")
        # 取第一个答案；如果答案为空，则设为 "no_answer"
        answers = example.get("answers", {}).get("text", [])
        answer = answers[0] if answers and len(answers) > 0 else "no_answer"
        text = f"Task: Question Answering\nQuestion: {question}\nContext: {context}\nAnswer: {answer}\n"
    elif task == "ag_news":
        text = f"Task: API Monitoring (Switch API if needed)\nLog: {example.get('text', '')}\nStatus: {example.get('label', '')}\n"
    elif task == "dbpedia_14":
        text = f"Task: API Monitoring (Switch API if needed)\nLog: {example.get('content', '')}\nStatus: {example.get('label', '')}\n"
    elif task == "mrpc":
        s1 = example.get("sentence1", "")
        s2 = example.get("sentence2", "")
        label = "yes" if example.get("label", 0) == 1 else "no"
        text = f"Task: API Monitoring - Log Similarity\nLog1: {s1}\nLog2: {s2}\nShould switch API? {label}\n"
    else:
        text = ""
    return {"text": text}


def load_and_prepare_dataset(sample_size=200):
    # 加载问答数据集
    print("Loading SQuAD...")
    squad = load_dataset("squad")["train"].select(range(sample_size))
    print("Loading Natural Questions...")
    nq = load_dataset("natural_questions", "default")["train"].select(range(sample_size))
    print("Loading MS MARCO...")
    ms_marco = load_dataset("ms_marco", "v2.1")["train"].select(range(sample_size))
    
    # 加载 API 监测相关数据集
    print("Loading AG News...")
    ag_news = load_dataset("ag_news")["train"].select(range(sample_size))
    print("Loading DBpedia...")
    dbpedia = load_dataset("dbpedia_14")["train"].select(range(sample_size))
    print("Loading GLUE MRPC...")
    mrpc = load_dataset("glue", "mrpc")["train"].select(range(sample_size))
    
    # 对各数据集预处理
    squad = squad.map(lambda x: prepare_multitask_data("squad", x))
    nq = nq.map(lambda x: prepare_multitask_data("natural_questions", x))
    ms_marco = ms_marco.map(lambda x: prepare_multitask_data("ms_marco", x))
    ag_news = ag_news.map(lambda x: prepare_multitask_data("ag_news", x))
    dbpedia = dbpedia.map(lambda x: prepare_multitask_data("dbpedia_14", x))
    mrpc = mrpc.map(lambda x: prepare_multitask_data("mrpc", x))
    
    # 合并所有数据集
    combined = concatenate_datasets([squad, nq, ms_marco, ag_news, dbpedia, mrpc])
    dataset = DatasetDict({"train": combined})
    print(f"Total training samples: {len(dataset['train'])}")
    return dataset


##########################################
# 2. Tokenize 数据集
##########################################

def tokenize_dataset(dataset, tokenizer, max_length=1024):
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length)
    tokenized = dataset["train"].map(tokenize_function, batched=True, remove_columns=["text"])
    return tokenized


##########################################
# 3. 教师模型 Fine-tuning
##########################################

def train_teacher_model(tokenized_dataset, model_name="Qwen/Qwen2.5-0.5B", output_dir="./qwen_teacher", num_train_epochs=1):
    print("Loading teacher model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=2,
        save_steps=500,
        save_total_limit=2,
        fp16=True if torch.cuda.is_available() else False,
        logging_steps=50,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    print("Starting teacher training...")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return model, tokenizer


##########################################
# 4. 构造学生模型（蒸馏）
##########################################

def create_student_model_from_teacher(teacher_model):
    teacher_config = teacher_model.config
    student_config_dict = teacher_config.to_dict()
    original_layers = teacher_config.num_hidden_layers
    # 简单地将隐藏层数减半，至少保留 1 层
    student_config_dict["num_hidden_layers"] = max(1, original_layers // 2)
    student_config = AutoConfig.from_dict(student_config_dict)
    student_model = AutoModelForCausalLM.from_config(student_config)
    print(f"Student model created: {original_layers} -> {student_model.config.num_hidden_layers} layers")
    return student_model


def distillation_training(teacher_model, student_model, tokenized_dataset, tokenizer,
                          num_epochs=1, batch_size=2, alpha=0.5, temperature=2.0, learning_rate=5e-5):
    """
    使用教师模型的软标签（经温度缩放）与学生模型的 NLL 损失进行知识蒸馏训练。
    损失 = alpha * KL_loss + (1 - alpha) * NLL_loss
    """
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    student_model.train()
    optimizer = optim.AdamW(student_model.parameters(), lr=learning_rate)
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)
    
    total_steps = len(dataloader) * num_epochs
    print(f"Starting distillation training for {total_steps} steps...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model.to(device)
    student_model.to(device)
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            # 对于因果 LM，标签即为 input_ids（内部会进行 shift 操作）
            labels = batch["input_ids"].to(device)
            
            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
                teacher_logits = teacher_outputs.logits
            
            student_outputs = student_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            student_logits = student_outputs.logits
            
            student_logits_T = student_logits / temperature
            teacher_logits_T = teacher_logits / temperature
            
            kl_loss = F.kl_div(F.log_softmax(student_logits_T, dim=-1),
                               F.softmax(teacher_logits_T, dim=-1),
                               reduction="batchmean") * (temperature ** 2)
            nll_loss = student_outputs.loss
            loss = alpha * kl_loss + (1 - alpha) * nll_loss
            
            loss.backward()
            optimizer.step()
            
            if step % 50 == 0:
                print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}")
    return student_model


##########################################
# 5. 模型动态量化
##########################################

def quantize_model(model):
    from torch.quantization import quantize_dynamic
    quantized_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    return quantized_model


##########################################
# 6. 推理测试
##########################################

def inference_test(model, tokenizer, device="cpu"):
    model.to(device)
    model.eval()
    test_prompt = ("Task: Question Answering\n"
                   "Question: What is the capital of France?\n"
                   "Context: France is a country in Europe.\n"
                   "Answer:")
    input_ids = tokenizer.encode(test_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=64)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("Test Output:\n", output_text)


##########################################
# 7. 主函数：执行全流程
##########################################

def main():
    # 1. 加载并预处理数据集（包含问答 & API监测任务）
    dataset = load_and_prepare_dataset(sample_size=200)
    
    # 2. 加载教师模型对应的 Tokenizer 并 Tokenize 数据集
    teacher_model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_name, use_fast=False)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, max_length=1024)
    
    # 3. Fine-tune 教师模型
    teacher_out_dir = "./qwen_teacher"
    teacher_model, _ = train_teacher_model(tokenized_dataset, model_name=teacher_model_name, output_dir=teacher_out_dir, num_train_epochs=1)
    
    # 4. 构造学生模型（压缩教师模型）
    student_model = create_student_model_from_teacher(teacher_model)
    
    # 5. 知识蒸馏训练
    print("Starting distillation training for student model...")
    student_model = distillation_training(teacher_model, student_model, tokenized_dataset, tokenizer,
                                          num_epochs=1, batch_size=2, alpha=0.5, temperature=2.0, learning_rate=5e-5)
    
    # 6. 对蒸馏后的学生模型进行动态量化
    print("Performing dynamic quantization on student model...")
    quantized_student = quantize_model(student_model)
    torch.save(quantized_student.state_dict(), "quantized_student_model.pt")
    print("Quantized student model saved as 'quantized_student_model.pt'")
    
    # 7. 保存最终模型和 Tokenizer 到本地
    save_dir = "./qwen_finetuned"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    student_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Fine-tuned model and tokenizer saved to {save_dir}")
    
    # 8. 推理测试（以简单问答为例）
    inference_test(quantized_student, tokenizer, device="cpu")


if __name__ == "__main__":
    main()
