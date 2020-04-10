import os
import pickle as pkl
import random
import numpy as np
from collections import Iterable

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler, BatchSampler

class Continuum(Dataset):
    '''
    The set of tuples containing a sample, a task and a target (x_i,t_i,y_i)
    is defined to be a "continuum of data" in the paper.
    '''
    def __init__(self, data, args):
        self.data = data
        self.n_tasks = len(data)
        self.length = sum([self.data[i][2].size(0) for i in range(len(self.data))])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        try:
            ti = np.where((int(idx)<np.cumsum([self.data[i][2].size()
                             for i in range(len(self.data))])))[0][0]
        except:
            raise IndexError("Index out of range")
        refined_idx = int(idx) - np.cumsum([self.data[i][2].size()
                            for i in range(len(self.data))])[ti-1] \
                            if ti > 0 else int(idx)
        return {'x': self.data[ti][1][refined_idx],
                'ti': ti, # task id
                'y': self.data[ti][2][refined_idx]}

class ContinuumIterator:
    '''
    Random permutation sampler for a continuum-based dataset.
    Allows for more configuration than a standard sampler, and ensures the
    batches are consistent w.r.t. the provided tasks
    '''
    def __init__(self, dataset, n_epochs, args):
        self.dataset = dataset
        self.n_tasks = dataset.n_tasks
        self.n_epochs = n_epochs
        task_permutation = range(self.n_tasks)

        if args.shuffle_tasks == 'yes':
            task_permutation = torch.randperm(self.n_tasks).tolist()

        sample_permutations = []
        self.task_offsets = [0]

        for t in range(self.n_tasks):
            N = self.dataset.data[t][2].size(0)
            #if args.samples_per_task <= 0:
            n = N
            #else:
            #    n = min(args.samples_per_task, N)

            self.task_offsets.append(n)
            p = torch.randperm(N)[0:n]
            sample_permutations.append(p)

        self.task_offsets = np.cumsum(self.task_offsets).tolist()

        self.permutation = []

        for t in range(self.n_tasks):
            task_t = task_permutation[t]
            for _ in range(self.n_epochs):
                task_p = [[task_t, i] for i in sample_permutations[task_t]]
                random.shuffle(task_p)
                self.permutation += task_p

        self.length = len(self.permutation)
        self.current = 0
        self.current_task = 0

    def __iter__(self):
        return self

    def __len__(self):
        return self.length

    def next(self):
        return self.__next__()

    def __next__(self):
        current = self.current
        if current >= self.length:
            raise StopIteration
        ti = self.permutation[current][0]
        if ti != self.current_task:
            # If current task doesn't match the current's sample task the task
            # id has already changed. Therefore switch to the next sample.
            self.current += 1
            raise StopIteration
        if current + 1 >= self.length:
            self.current_task = -1
        elif ti != self.permutation[current + 1][0]:
            self.current_task = self.permutation[current + 1][0] # If
            #current task is different than that of a future sample, change the
            #task id
        else:
            self.current += 1
        return self.task_offsets[ti] + \
               self.permutation[current][1]

class ContinuumSampler(Sampler):
    '''
    Creates a new instance of ContinuumIterator on each request
    '''
    def __init__(self, data_source, args, n_epochs = 1):
        self.data_source = data_source
        self.args = args
        self.n_epochs = n_epochs
        self.ci = ContinuumIterator(self.data_source, self.n_epochs, self.args)
        self.length = len(self.ci)
        self.n_tasks = self.ci.n_tasks

    def __iter__(self):
        if self.ci.current < self.ci.length:
            return self.ci
        else:
            return ContinuumIterator(self.data_source, self.n_epochs, self.args)

    def __len__(self):
        return self.length

class ContinuumBatchSampler(BatchSampler):
    def __iter__(self):
        batch = []
        current_batch = 0
        while len(self) > current_batch:
            for idx in self.sampler:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    current_batch += 1
                    batch = []
            if len(batch) > 0 and not self.drop_last:
                current_batch += 1
                yield batch

def load_continuum(problem, graph_sizes, args, data_path = 'data',
                   seed = 1234, num_samples = None, suffix = 'train'):
    tasks = []
    if isinstance(seed,list):
        if len(seed) < len(graph_sizes):
            seed = seed[0]
        else:
            seed = iter(seed)
    for taskid in graph_sizes:
        if isinstance(seed,Iterable):
            s = next(seed)
        else:
            s = seed
        task_file = os.path.join(data_path, problem,
                      problem+str(taskid)+"_"+suffix+"_seed"+str(s)+".pkl")
        with open(task_file, 'rb') as f:
            data = pkl.load(f)
            name = task_file.split("_")
            labels = torch.zeros(len(data))
            if num_samples is not None:
                data = data[:num_samples]
                labels = labels[:num_samples]
            tasks.append([taskid, torch.tensor(data), labels])

    return Continuum(tasks, args)

def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename


def save_dataset(dataset, filename):

    filedir = os.path.split(filename)[0]

    if not os.path.isdir(filedir):
        os.makedirs(filedir)

    with open(check_extension(filename), 'wb') as f:
        pkl.dump(dataset, f, pkl.HIGHEST_PROTOCOL)


def load_dataset(filename):

    with open(check_extension(filename), 'rb') as f:
        return pkl.load(f)
