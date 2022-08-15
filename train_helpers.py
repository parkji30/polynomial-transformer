import random
import re
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

def create_vocabs(file='/train.txt'):
    """Splits the equations into input and target.
    
    Args:
        file (String): training file name.
        
    Returns:
        List, List, List: Vocabulary and Processed
                          Factor, Expand pairs.
    """
    data = open(file, 'r')

    pairs = []
    input_vocab = set()
    target_vocab = set()

    for line in data:
        input_text, target_text = line.strip().split('=')
            
        input_text = re.findall(r"sin|cos|tan|\d+|\w|\(|\)|\+|-|\*+", input_text.strip().lower())
        target_text = re.findall(r"sin|cos|tan|\d+|\w|\(|\)|\+|-|\*+", target_text.strip().lower())
        
        for char in input_text:
            if char not in input_vocab:
                input_vocab.add(char)
        for char in target_text:
            if char not in target_vocab:
                target_vocab.add(char)

        input_text = " ".join(input_text)
        target_text = '[start] ' + " ".join(target_text) + ' [end]'
        pairs.append((input_text, target_text))

    input_vocab = list(input_vocab)
    target_vocab = list(target_vocab) + ['[start]', '[end]']

    pairs.append((input_text, target_text))
    
    return input_vocab, target_vocab, pairs


def get_training_pairs(text_pairs, test_pair_percent=0.01):
    '''
    Splits the pairs into train, val, test pairs
    
    Args:
        text_pairs ((String, String)): factor and expand strings.
        test_pair_percent (float): percentage size of test set 

    Returns:
        List, List, List: Lists containing train, val, test 
                             data.
    '''
    random.shuffle(text_pairs)
    num_val_samples = int(test_pair_percent * len(text_pairs))
    num_train_samples = len(text_pairs) - 2 * num_val_samples
    train_pairs = text_pairs[:num_train_samples]
    val_pairs = text_pairs[num_train_samples : num_train_samples + num_val_samples]
    test_pairs = text_pairs[num_train_samples + num_val_samples :]

    print(f"{len(text_pairs)} total pairs")
    print(f"{len(train_pairs)} training pairs")
    print(f"{len(val_pairs)} validation pairs")
    print(f"{len(test_pairs)} test pairs")
    
    return train_pairs, val_pairs, test_pairs


def create_TextVectorizations(model_params, input_vocab, target_vocab):
    """Create TextVectorization objects for tokenizing our factors and 
    expansions.

    Args:
        model_params (dict): dictionary with model parameters
        input_vocab (list): vocabulary of factor
        target_vocab (list): vocabulary of expand

    Returns:
        Object1, Object2: returns the input and output TextVectorization
                          objects.
    """
    factor_vocab_size = model_params['FACTOR_VOCAB_SIZE']
    expanded_vocab_size = model_params['EXPANDED_VOCAB_SIZE']
    seq_length = model_params['MAX_SEQUENCE_LENGTH']
    
    factor_vectorization = TextVectorization(
        max_tokens = factor_vocab_size,
        output_mode = "int",
        output_sequence_length = seq_length,
        standardize = None
    )
     
    expand_vectorization = TextVectorization(
        max_tokens = expanded_vocab_size,
        output_mode = "int",
        output_sequence_length = seq_length + 1,
        standardize = None
    )
    
    factor_vectorization.set_vocabulary(input_vocab)
    expand_vectorization.set_vocabulary(target_vocab)
    
    return factor_vectorization, expand_vectorization
