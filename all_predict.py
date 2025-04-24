import os

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers
from datasets import Dataset
from peft import LoraConfig, PeftConfig
from trl import SFTTrainer
from trl import setup_chat_format
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BitsAndBytesConfig,
                          TrainingArguments,
                          pipeline,
                          logging)
from sklearn.metrics import (accuracy_score,
                             classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split

from sklearn.metrics import (accuracy_score,
                             recall_score,
                             precision_score,
                             f1_score)



# checkpoint = 225
# in : 2475
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
)
max_seq_length = 1024  # 2048
def generate_test_prompt(data_point):
    # 인스트럭션 있음
    # return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|> GitHub Issue Report Classifier <|eot_id|><|start_header_id|>user<|end_header_id|> Classify, IN ONLY 1 WORD, the following GitHub issue as 'feature', 'bug', or 'question' based on its title and body:" + str(
    #     data_point["content"]) + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>".strip()

# 인스트럭션 없음
   return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|> GitHub Issue Report Classifier <|eot_id|><|start_header_id|>user<|end_header_id|>"+ str(
        data_point["content"]) + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>".strip()

def predict(X_test, model, tokenizer):
    y_pred = []
    y_answer = []
    for i in tqdm(range(len(X_test))):
        try:
            prompt = X_test.iloc[i]["text"]
            pipe = pipeline(task="text-generation",
                            model=model,
                            tokenizer=tokenizer,
                            max_new_tokens=1,
                            temperature=0.1,
                            # 여기
                            )
            result = pipe(prompt)
            answer = result[0]['generated_text'].split("<|end_header_id|>")[-1]
            y_answer.append(answer)
            if "question" in answer:
                y_pred.append("question")
            elif "feature" in answer:
                y_pred.append("feature")
            elif "bug" in answer:
                y_pred.append("bug")
            else:

                print(answer+" : 없는 레이블?")
                y_pred.append("empty")
                # print(result)
        except Exception as e:
            # print(len(tokenizer(X_test.iloc[i]["text"])['input_ids']))
            y_pred.append("empty")
            y_answer.append('empty')
            # print(e)
            # print('bug')

    return y_pred, y_answer

def evaluate(y_true, y_pred):
    labels = ["bug", "feature", "question"]
    mapping = {'question': 2, 'feature': 1, 'bug': 0}

    def map_func(x):
        return mapping.get(x, 1)

    y_true = np.vectorize(map_func)(y_true)
    y_pred = np.vectorize(map_func)(y_pred)

    # Calculate accuracy
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    print(f'Accuracy: {accuracy:.3f}')

    # Generate accuracy report
    unique_labels = set(y_true)  # Get unique labels

    for label in unique_labels:
        label_indices = [i for i in range(len(y_true))
                         if y_true[i] == label]
        label_y_true = [y_true[i] for i in label_indices]
        label_y_pred = [y_pred[i] for i in label_indices]
        accuracy = accuracy_score(label_y_true, label_y_pred)
        print(f'Accuracy for label {label}: {accuracy:.3f}')

    # Generate classification report
    class_report = classification_report(y_true=y_true, y_pred=y_pred)
    print('\nClassification Report:')

    file_name = "./result/2024_mIx_pIx/" + project_name + "_" + str(checkpoint) + "_new_cm_new.txt"
    with open(file_name, "w") as text_file:
        print(classification_report(y_true, y_pred, digits=4), file=text_file)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1, 2])
    print('\nConfusion Matrix:')

    print(conf_matrix)


def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

def predict_a(project_name, checkpoint,  Clean_Method="CM1"):
    print(project_name)
    test_data = pd.read_csv('../CM1/data/csvdata/' + project_name + '_test.csv')
    print('train data : ../CM1/data/csvdata/' + project_name + '_train.csv')
    print('test data : ../CM1/data/csvdata/' + project_name + '_test.csv')

    X_test = pd.DataFrame(test_data.apply(generate_test_prompt, axis=1), columns=["text"])
    y_true = test_data.label

    # instruction없음
    model_name = "../CM1/model/0204/" + project_name + "-llama-classifier/checkpoint-" + str(checkpoint)
    # 인스트럭션 있음
    # model_name = "../CM1/model/epoch3/" + project_name + "-llama-classifier/checkpoint-" + str(checkpoint)

    print("model 이름: " + model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=compute_dtype,
        quantization_config=bnb_config,
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1
    tokenizer = AutoTokenizer.from_pretrained(model_name, max_seq_length=max_seq_length)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    y_pred, y_answer = predict(X_test, model, tokenizer)
    result = evaluate(y_true, y_pred)

    evaluation = pd.DataFrame({'text': X_test["text"],
                               'y_true': y_true,
                               'y_pred': y_pred,
                               'y_answer': y_answer},
                              )

    evaluation[0:5].to_csv("./result/2024_mIx_pIx/" + project_name + "_" + str(checkpoint) + "_result.csv", index=False)



if __name__ == '__main__':
    # p_list = ['facebook', 'tensorflow', 'microsoft', 'bitcoin', 'opencv',
    #          'flutter', 'kubernetes', 'roslyn', 'typescript', 'dartlang']

    p_list = ['flutter']


    checkpoint = 225
    # checkpoint = 2475

    for project_name in p_list:
        if project_name == "":
            checkpoint = 225
        predict_a(project_name, checkpoint)
