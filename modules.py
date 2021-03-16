import pandas as pd
from numpy import random
import time

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F


def building_vocabulary(data):
    # TODO: implement!
    WORDSINLIST = []
    for i in list(data['sentence'].str.split()):
        for j in i:
            WORDSINLIST.append(j)


    tempv = set(WORDSINLIST)

    V = pd.DataFrame({'Values': data} for data in tempv)

    # WORDLIST :: List of all words

    # V :: Vocabulary -> unique set of words from WORDLIST

    return WORDSINLIST, V

# WORDSINLIST, V = building_vocabulary(data)


def word_to_one_hot(word, V):
  lis=[]
  for i in V['Values']:
    if i.strip() in word.strip() and word.strip() in i.strip():
      lis.append(1)
    else:
      lis.append(0)
  return lis


# TODO: implement!
def sampling_prob(word, WORDSINLIST):
    COUNT = 0
    COUNT = WORDSINLIST.count(word)

    Z = COUNT / len(WORDSINLIST)
    if Z == 0:
        return 0

    return ((((Z / 0.001) ** .5) + 1) * (0.001 / Z))


def get_target_context(sentence, WORDSINLIST,  window_size=4):
    words = sentence.split()

    count = 0
    for i in words:
        templis = []

        TOTALCOUNT = 0

        j = count - 1

        moves = 0

        while j >= 0 and moves < (window_size / 2):
            # print(sampling_prob(words[j]))
            if sampling_prob(words[j], WORDSINLIST) > random.rand():
                templis.append(WORDSINLIST.index(words[j]))
                TOTALCOUNT += 1
            j -= 1
            moves += 1

        t = count
        templis1 = []

        FLAG = True

        while moves < window_size and FLAG:

            if t + 1 < len(words):
                if sampling_prob(words[t + 1], WORDSINLIST) > random.rand():
                    templis1.append(WORDSINLIST.index(words[t + 1]))
                    moves += 1
                    TOTALCOUNT += 1
            if t + 1 >= len(words):
                FLAG = False
            t += 1

        templis2 = []

        while TOTALCOUNT < window_size and j >= 0:
            if sampling_prob(words[j], WORDSINLIST) > random.rand():
                templis2.append(WORDSINLIST.index(words[j]))
                TOTALCOUNT += 1
            j -= 1

        templis3 = []

        FLAG1 = True
        while TOTALCOUNT < window_size and FLAG1:
            if t + 1 < len(words):
                if sampling_prob(words[t + 1], WORDSINLIST) > random.rand():
                    templis3.append(WORDSINLIST.index(words[t + 1]))
                    TOTALCOUNT += 1
            if t + 1 >= len(words):
                FLAG1 = False
            t += 1

        finallist = templis + templis2 + templis1 + templis3
        yield (words[count], finallist)

        count += 1




def create_currents_contexts(df,WORDSINLIST, V):

  currentlist = []
  contextlist = []
  start_time = time.time()

  print("appending to list started...")
  for sentence in df['sentence']:
    print(sentence)
    gen= get_target_context(sentence,WORDSINLIST, window_size=4)
    for text in range(len(sentence.split())-1):
      current, context = next(gen)
      current_one_hot = word_to_one_hot(current, V)

      current_one_hot = torch.FloatTensor(current_one_hot)

      currentlist.append(current_one_hot)

      while(len(context)< 4):
        context.append(len(V)+1)

      context = torch.tensor(context)
      contextlist.append(context)

  print("appending to list ended.")
  print("--- %s seconds ---" % (time.time() - start_time))
  return currentlist, contextlist

def return_dataloader(currentlist, contextlist, batch_size):
  x_tensor = torch.stack(currentlist)
  y_tensor = torch.stack(contextlist)

  # Define dataset
  train_ds = TensorDataset(x_tensor, y_tensor)

  # Define data loader
  train_dl = DataLoader(train_ds, batch_size, shuffle=True)

  return train_dl


class Word2vec(nn.Module):
    def __init__(self, LENGTH):
      super(Word2vec, self).__init__()
      self.LENGTH = LENGTH
      self.FC1 = nn.Linear(self.LENGTH, embedding_size,0)

     # initialization of weights
      torch.nn.init.xavier_normal_(self.FC1.weight)

      self.FC2 = nn.Linear(embedding_size, self.LENGTH,0)
      # initialization of weights
      torch.nn.init.xavier_normal_(self.FC2.weight)

    def forward(self, one_hot):
      #one_hot = torch.tensor(one_hot)
      x = self.FC1(one_hot)
      y = self.FC2(x)


      m = nn.LogSoftmax(dim=None)
      y = m(y)
      return y

