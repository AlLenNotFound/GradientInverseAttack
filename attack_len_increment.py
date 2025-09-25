from attack import reconstruct, args
from datasets import load_metric
from utils.data import TextDataset
from utils.models import ModelWrapper
import torch
import time

def main():
    device = torch.device("cuda")
    metric = load_metric('rouge', cache_dir='./models_cache')
    dataset = TextDataset("cuda", "glnmario/ECHR", "val", 100, args.batch_size, './models_cache')

    model_wrapper = ModelWrapper(args)

    print('\n\nAttacking..\n', flush=True)
    predictions, references = [], []
    
    t_input_start = time.time()

    tokenizer = model_wrapper.tokenizer
    
    tokenized_samples = [tokenizer(dataset[i][0], return_tensors='pt', padding=True, truncation=True, max_length=tokenizer.model_max_length)['input_ids'] for i in range(4, 14)]
    for l in range(args.start_input, args.end_input, 4):
        accuracy=0
        for i in range(10):
            print(f'Running input #{(l-args.start_input)//4*10+i} of {(args.end_input - args.start_input)//4*10}.', flush=True)
            if args.neptune:
                args.neptune['logs/curr_input'].log(l)
            new_sample = model_wrapper.tokenizer.batch_decode(tokenized_samples[i][:, :l])
        
            prediction, reference = reconstruct(args, device, (new_sample, dataset[i][1]), metric, model_wrapper)
            predictions += prediction
            references += reference

            print(f'Done with input #{(l-args.start_input)//4*10+i} of {(args.end_input - args.start_input)//4*10}.', flush=True)
            print('reference: ', flush=True)
            for seq in reference:
                print('========================')
                print(seq)
                print('========================')

            print('prediction: ', flush=True)
            for seq in prediction:
                print('========================')
                print(seq)
            print('========================')
            tokenized_pred = model_wrapper.tokenizer(prediction, return_tensors='pt', padding=True, truncation=True, max_length=tokenizer.model_max_length)['input_ids']
            tokenized_ref = model_wrapper.tokenizer(reference, return_tensors='pt', padding=True, truncation=True, max_length=tokenizer.model_max_length)['input_ids']
            total_correct = 0
            for i in range(args.batch_size):
                total_correct+=(tokenized_pred[i, :min(len(tokenized_pred[i]), len(tokenized_ref[i]))] == tokenized_ref[i, :min(len(tokenized_pred[i]), len(tokenized_ref[i]))]).int().sum()
            print(f'Sample accuracy: {(total_correct/tokenized_ref.numel()*100).item():.2f}')
            if args.neptune:
                args.neptune['logs/sample_acc'].log(total_correct/tokenized_ref.numel())
            accuracy += total_correct/tokenized_ref.numel()
        print('Sample set accuracy', (accuracy*10).item())
        if args.neptune:
            args.neptune['logs/accuracy'].log(accuracy/10)
    print('Done with all.', flush=True)
    if args.neptune:
        args.neptune['logs/curr_input'].log(args.n_inputs)

if __name__ == '__main__':
    main()
