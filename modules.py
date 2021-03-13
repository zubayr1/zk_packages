def building_vocabulary(data):
    # TODO: implement!
    WORDSINLIST = []
    for i in list(data['sentence'].str.split()):
        for j in i:
            WORDSINLIST.append(j)

    # WORDSINLIST = df['text'].values.tolist()

    # print(WORDSINLIST)
    tempv = set(WORDSINLIST)

    V = pd.DataFrame({'Values': data} for data in tempv)

    # WORDLIST :: List of all words

    # V :: Vocabulary -> unique set of words from WORDLIST

    return WORDSINLIST, V

# WORDSINLIST, V = building_vocabulary(data)


def word_to_one_hot(word):
  lis=[]
  for i in V['Values']:
    if i.strip() in word.strip() and word.strip() in i.strip():
      lis.append(1)
    else:
      lis.append(0)
  return lis