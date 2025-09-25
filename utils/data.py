from datasets import load_dataset
import numpy as np
import torch 

class TextDataset:
    def __init__(self, device, dataset, split, n_inputs, batch_size, cache_dir=None):
        print("load_dataset with glue, name:", dataset, "split:", split)

        if split == 'val':
            split_name = 'validation'
        elif split == 'test':
            split_name = 'test'
        else:
            raise ValueError(f"Unknown split value: {split}")

        seq_keys = {
            'cola': 'sentence',
            'sst2': 'sentence',
            'rte': 'sentence1',
            'rotten_tomatoes': 'text',
            'glnmario/ECHR': 'text',
            'stanfordnlp/imdb': 'text'
        }
        seq_key = seq_keys[dataset]

        print(f"Final split passed to load_dataset: {split_name}")
        if dataset in ['cola', 'sst2', 'rte']:
            # full = load_dataset('glue', dataset, cache_dir=cache_dir)['train']
            split_name = 'validation' if split == 'val' else split
            full = load_dataset(path='glue', name=dataset, split=split_name, cache_dir=cache_dir)
        elif dataset == 'glnmario/ECHR':
            full = load_dataset('csv', data_files = ['models_cache/datasets--glnmario--ECHR/ECHR_Dataset.csv'], cache_dir=cache_dir)['train']
        else:
            full = load_dataset(dataset)['train']
            
        idxs = list(range(len(full)))
        np.random.shuffle(idxs)
        #if dataset == 'cola':
        #    import pdb; pdb.set_trace()
        #    assert idxs[0] == 2310 # with seed 101

        n_samples = n_inputs * batch_size

        if split == 'test':
            assert n_samples <= 1000
            idxs = idxs[:n_samples]
        elif split == 'val':
            idxs = idxs[1000:] # first 1000 saved for testing
            
            final_idxs = []
            while len(final_idxs) < n_samples:
                zipped = [(idx, len(full[idx][seq_key])) for idx in idxs]
                zipped = sorted(zipped, key=lambda x: x[1])
                chunk_sz = max(len(zipped) // n_samples, 1) 
                
                l = min(len(zipped), n_samples - len(final_idxs))
                for i in range(l):
                    tmp = chunk_sz*i + np.random.randint(0, chunk_sz)
                    final_idxs.append(zipped[tmp][0])
                np.random.shuffle(idxs) 
                
            np.random.shuffle(final_idxs) 
            idxs = final_idxs
        
        # Slice
        self.seqs = []
        self.labels = []
        for i in range(n_inputs):
            seqs = []
            for j in range(batch_size):
                seqs.append(full[idxs[i*batch_size+j]][seq_key])
            if dataset != 'glnmario/ECHR':
                labels = torch.tensor([full[idxs[i*batch_size : (i+1)*batch_size]]['label']], device=device)
            else:
                labels = torch.tensor([full[idxs[i*batch_size : (i+1)*batch_size]]['binary_judgement']], device=device)
            self.seqs.append(seqs)
            self.labels.append(labels)
        assert len(self.seqs) == n_inputs
        assert len(self.labels) == n_inputs


    def __getitem__(self, idx):
        return (self.seqs[idx], self.labels[idx])
