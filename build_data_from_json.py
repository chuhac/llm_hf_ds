from transformers import AutoTokenizer
from datasets import Dataset
import datasets
import argparse
import os
import copy
from tqdm import tqdm


LOSS_IGNORE_INDEX = -100
TRUNCATION_LENGTH = None

def parse_args():
    parser = argparse.ArgumentParser(description="Build Arrow Dataset to disk from json, e.g.: python build_data_from_json.py --tokenizer_file_path ./meta-7b --json_file ./data.json --save_to ./saved_hf_dataset --truncation_length 2048")
    parser.add_argument("--tokenizer_file_path", type=str)
    parser.add_argument("--json_file", type=str)
    parser.add_argument("--save_to", type=str)
    parser.add_argument("--truncation_length", type=int, default=2048)
    parser.add_argument("--is_pretraining", action='store_true')
    parser.add_argument("--prompt_type", '-pt', type=str, default='llama')

    
    
    args = parser.parse_args()
    return args


def batched_preprocess_for_sft(examples, role_word='human'):
    input_ids = []
    labels = []
    attention_mask = []
    # assert 'conversations' in examples.keys(), 'Column `conversations` is required when SFT'
    assert 'conversation' in examples.keys(), 'Column `conversation` is required when SFT'

    conversations = examples["conversation"]
    for one_conversation in conversations:
        one_input_ids = []; one_labels = []; one_attention_mask = []
        for idx, sentence in enumerate(one_conversation):
            sentence_from = sentence["role"].lower()
            if args.prompt_type == 'llama':
                sentence_value = 'Human: \n' + sentence["text"].strip() + '\n\nAssistant: \n' if sentence_from == role_word else sentence["text"].strip()
            elif args.prompt_type == 'baichuan':
                sentence_value = '<reserved_106>' + sentence["text"].strip() + '<reserved_107>' if sentence_from == role_word else sentence["text"].strip()
            sentence_ids = tokenizer.encode(sentence_value, add_special_tokens=False) #do not add bos_token_id
            label = copy.deepcopy(sentence_ids) if sentence_from != role_word else [LOSS_IGNORE_INDEX] * len(sentence_ids)
            
            
            # add eos at every end of assistant sentence
            if sentence_from != role_word:
                sentence_ids += [tokenizer.eos_token_id]#make sure eos_token_id is correct
                label += [tokenizer.eos_token_id]
                
            one_input_ids += sentence_ids
            one_labels += label
            
            one_attention_mask += [1] * len(sentence_ids)

        one_input_ids = one_input_ids[:TRUNCATION_LENGTH-1]
        one_labels = one_labels[:TRUNCATION_LENGTH-1]
        one_attention_mask = one_attention_mask[:TRUNCATION_LENGTH-1]
        

        if all(x == LOSS_IGNORE_INDEX for x in one_labels):
            break
            # one_labels[18:24] = one_input_ids[18:24] #labels can not have all values being -100. 18 and 24 are just random numbers
        input_ids.append(one_input_ids[:]); labels.append(one_labels[:]); attention_mask.append(one_attention_mask[:])
    tokenized_full_prompt = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
    
    return tokenized_full_prompt
    
    
def batched_preprocess_for_pretraining(examples):
    assert 'text' in examples.keys(), 'Column `text` is required when pretraining'
    
    input_ids = []
    text_list = examples['text']
    
    sentence_id_lists = tokenizer(text_list, add_special_tokens=False, padding=False).input_ids
    for sentence_id in sentence_id_lists:
        input_ids += [sentence_id[i * TRUNCATION_LENGTH: (i + 1) * TRUNCATION_LENGTH]
                      for i in range(len(sentence_id) // TRUNCATION_LENGTH)]
    
    labels = copy.deepcopy(input_ids)
    
    tokenized_pretrained = {
        "input_ids": input_ids,
        "labels": labels
    }
    
    return tokenized_pretrained



if __name__ == '__main__':


    args = parse_args()
    
    print(f'Loading data from {args.json_file}...\n')
    
    try:
        dataset = Dataset.from_json(args.json_file)
    except FileNotFoundError:
        print(f'{args.json_file} not found.')
        
    print(f'{dataset.num_rows} rows of data from {args.json_file} loaded')
    
    try:
        if args.prompt_type != 'baichuan':
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_file_path, use_fast=False)
            # tokenizer.pad_token_id = 0 # that is <unk>, initial llama has no <pad>
            # tokenizer.bos_token_id = 1
            # tokenizer.eos_token_id = 2
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_file_path, use_fast=False, trust_remote_code=True)
            
        # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_file_path, use_fast=False)
        

    except:
        print(f'{args.tokenizer_file_path} not found or not a path to a huggingface pretrained tokenizer')
        
    
    if not os.path.exists(args.save_to):
        try:
            os.makedirs(args.save_to)
        except:
            print(f'Exception occurred when creating directory: {args.save_to}')
            
        
    tokenizer.model_max_length = args.truncation_length
    TRUNCATION_LENGTH = args.truncation_length
    
    print(f'Dataset of {dataset.num_rows} data processing...')
    
    if args.is_pretraining:
        tokenized_dataset = dataset.map(batched_preprocess_for_pretraining, batched=True, batch_size=128, num_proc=8, remove_columns=dataset.column_names)
    else:
        tokenized_dataset = dataset.map(batched_preprocess_for_sft, batched=True, batch_size=1, num_proc=8, remove_columns=dataset.column_names)
        
    if (not args.is_pretraining) and dataset.num_rows > tokenized_dataset.num_rows:
        print(f'{dataset.num_rows - tokenized_dataset.num_rows} out of {dataset.num_rows} conversations skipped because they are all human murmurs')
    
    print(f'Dataset saving... Number of rows: {tokenized_dataset.num_rows}')
    tokenized_dataset.save_to_disk(args.save_to)
    print(f'Dataset successfully saved to {args.save_to}')
