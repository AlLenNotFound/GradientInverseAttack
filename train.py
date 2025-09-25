import argparse
import numpy as np
import torch
import peft
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, AdamW, get_scheduler
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

np.random.seed(100)
torch.manual_seed(100)
device = 'cuda'

def save_model(model, save_directory, train_method):
    print('SAVING', flush=True)

    # Ensure the directory exists
    #if not os.path.exists(os.path.dirname(save_directory)):
        #os.makedirs(os.path.dirname(save_directory))

    # Save the model
    try:
        if train_method != 'lora':
            model.save_pretrained(save_directory)
        else:
            torch.save(model.state_dict(), save_directory)
        print("Model saved successfully.", flush=True)
        print(os.listdir(save_directory), flush=True)
    except Exception as e:
        print(f"Error saving model: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['cola', 'sst2', 'rte', 'rotten_tomatoes'], default='cola')
    parser.add_argument('--save_every', type=int, default=5000)
    parser.add_argument('--noise', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--pct_mask', type=float, default=None)
    parser.add_argument('--model_path', type=str, default='bert-base-uncased')
    parser.add_argument('--train_method', type=str, default='full', choices=['full', 'lora'])
    parser.add_argument('--lora_r', type=int, default=None)  
    parser.add_argument('--models_cache', type=str, default='./models_cache')
    args = parser.parse_args()

    seq_key = 'text' if args.dataset == 'rotten_tomatoes' else 'sentence'
    num_labels = 2
    
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=num_labels, cache_dir='./models_cache').to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, cache_dir=args.models_cache)
    
    # Configure tokenizer and model
    if tokenizer.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        
    # Load LoRA model if applicable
    if args.train_method == 'lora':
        lora_cfg = peft.LoraConfig(r=args.lora_r,target_modules=['q_proj'])
        model = peft.LoraModel(model, lora_cfg, "default")
        
    tokenizer.model_max_length = 512
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if args.dataset == 'cola':
        metric = load_metric('matthews_correlation')
        train_metric = load_metric('matthews_correlation')
    else:
        metric = load_metric('./train_utils/accuracy.py')
        train_metric = load_metric('./train_utils/accuracy.py')

    def tokenize_function(examples):
        return tokenizer(examples[seq_key], truncation=True)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    if args.dataset in ['cola', 'sst2', 'rte']:
        datasets = load_dataset('glue', args.dataset)
    else:
        datasets = load_dataset(args.dataset)
    
    tokenized_datasets = datasets.map(tokenize_function, batched=True)
    if args.dataset == 'cola' or args.dataset == 'sst2':
        tokenized_datasets = tokenized_datasets.remove_columns(['idx', 'sentence'])
    elif args.dataset == 'rotten_tomatoes':
        tokenized_datasets = tokenized_datasets.remove_columns(['text'])
    elif args.dataset == 'rte':
        tokenized_datasets = tokenized_datasets.remove_columns(['idx', 'sentence1', 'sentence2'])
    else:
        assert False
    tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')
    tokenized_datasets.set_format('torch')

    train_dataset = tokenized_datasets['train']
    eval_dataset = tokenized_datasets['validation']
    
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=data_collator)
    eval_loader = DataLoader(eval_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=data_collator)

    opt = AdamW(model.parameters(), lr=5e-5)

    num_training_steps = args.num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        'linear',
        optimizer=opt,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(range(num_training_steps))
    model.train()
    n_steps = 0
    train_loss = 0
    
    # Run training loop
    for epoch in range(args.num_epochs):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            train_metric.add_batch(predictions=predictions, references=batch['labels'])
            
            loss = outputs.loss
            train_loss += loss.item()
            loss.backward()

            if args.pct_mask is not None:
                for param in model.parameters():
                    grad_mask = (torch.rand(param.grad.shape).to(device) > args.pct_mask).float()
                    param.grad.data = param.grad.data * grad_mask

            if args.noise is not None:
                for param in model.parameters():
                    param.grad.data = param.grad.data + torch.randn(param.grad.shape).to(device) * args.noise

            opt.step()
            lr_scheduler.step()
            opt.zero_grad()
            progress_bar.update(1)

            n_steps += 1
            if n_steps % args.save_every == 0:
                save_model(model, f'./finetune/{args.train_method}_{n_steps}.pt', args.train_method)
                print('metric train: ', train_metric.compute())
                print('loss train: ', train_loss/n_steps)
                train_loss = 0.0

        model.eval()
        
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            metric.add_batch(predictions=predictions, references=batch['labels'])
            
        with open(f'finetune/{args.dataset}_metric.txt', 'w') as fou:
            #print('metric eval: ', metric.compute(), file=fou)
            print('metric eval: ', metric.compute())
    print('END')
    save_model(model, f'./finetune/{args.train_method}_{n_steps}.pt', args.train_method)
    

if __name__ == '__main__':
    main()
