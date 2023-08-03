from common.util import create_contexts_target, preprocess, convert_one_hot, convert_one_hot_old

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
print(corpus)

print(id_to_word)

contexts, target = create_contexts_target(corpus, window_size=1)
print(contexts)
print(target)

vocab_size = len(word_to_id)
target = convert_one_hot(target, vocab_size)
print(target)
contexts = convert_one_hot(contexts, vocab_size)
print(contexts)
