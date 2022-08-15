import sys
import numpy as np
from typing import Tuple
import os
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

# These are our installed packages
from tqdm import tqdm
import re
import pickle
import keras_nlp
from keras.models import load_model
from tensorflow.keras.layers import TextVectorization

MAX_SEQUENCE_LENGTH = 29
TRAIN_URL = "https://scale-static-assets.s3-us-west-2.amazonaws.com/ml-interview/expand/train.txt"


def load_file(file_path: str) -> Tuple[Tuple[str], Tuple[str]]:
    """ A helper functions that loads the file into a tuple of strings

    :param file_path: path to the data file
    :return factors: (LHS) inputs to the model
            expansions: (RHS) group truth
    """
    data = open(file_path, "r").readlines()
    factors, expansions = zip(*[line.strip().split("=") for line in data])
    return factors, expansions


def score(true_expansion: str, pred_expansion: str) -> int:
    """ the scoring function - this is how the model will be evaluated

    :param true_expansion: group truth string
    :param pred_expansion: predicted string
    :return:
    """
    return int(true_expansion == pred_expansion)


# --------- START OF IMPLEMENT THIS --------- #
def predict(factors: str, 
            model: object, 
            f_vector: object, 
            e_vector: object, 
            e_lookup: object, 
            seq_length: int = 29):
    """
    Prediction function for guessing the expansion of factors.
    The input comes as a string which is then processed as a list of 
    individual tokens and fed into the pre-trained model to be 
    decoded into the expansion prediction.
    
    :param factors: factors that will be expanded
    :param model: pre-trained model use for guessing.
    :param f_vector: vectorization object for factors
    :param e_vector: vectorization object for expands
    :param e_lookup: lookup table for tokens
    :param seq_length: maximum length in data.
    :return: String
    """
    def text_standardize(text: str):
        '''
        Normalizes the text to display cleaned up output.
        
        Ex: 
            x * * 2 --> x**2
            
        :param text: Guessed Expansion
        :return: String
        '''
        text = (
            text.replace("[pad]", "")
            .replace("[start]", "")
            .replace("[end]", "")
            .strip()    
        )  
        text = text.replace(" ", "")
        return text
    
    # preprocess data to split everything.
    factors = re.findall(r"sin|cos|tan|\d+|\w|\(|\)|\+|-|\*+", factors.strip().lower())
    factors = " ".join(factors)
        
    tokenized_input_sentence = f_vector([factors])
    decoded_sentence = "[start]"
    
    for i in range(seq_length):
        tokenized_target_sentence = e_vector([decoded_sentence])[:, :-1]
        predictions = model([tokenized_input_sentence, tokenized_target_sentence])

        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = e_lookup[sampled_token_index]
        decoded_sentence += ' ' + sampled_token

        if sampled_token == "[end]":
            break
    
    expansion = text_standardize(decoded_sentence)
    return expansion
# --------- END OF IMPLEMENT THIS --------- #


def main(filepath: str):    
    # Loading the the appropriate Vocabulary and Lookup index.
    expand_index_lookup = pickle.load(open(os.path.join(__location__, 'best_model/expand_lookup_index.pickle'), 'rb'))

    from_disk = pickle.load(open(os.path.join(__location__, "best_model/factor_vectorization.pickle"), "rb"))
    factor_vectorization = TextVectorization.from_config(from_disk['config'])
    factor_vectorization.set_weights(from_disk['weights'])

    from_disk = pickle.load(open(os.path.join(__location__, "best_model/expand_vectorization.pickle"), "rb"))
    expand_vectorization = TextVectorization.from_config(from_disk['config'])
    expand_vectorization.set_weights(from_disk['weights'])
    
    # Loads a pretrained model to do the Guessing. 
    # This model Achieved ~ 97% when I ran it on train.txt
    polynomial_transformer = load_model(os.path.join(__location__, 
                                                     "best_model/polynomial_Transformers_heads-8_embed_dim-256_inter_dim-512_best_model.h5"), 
                            custom_objects = {'TokenAndPositionEmbedding': keras_nlp.layers.TokenAndPositionEmbedding,
                                            'TransformerEncoder': keras_nlp.layers.TransformerEncoder,
                                            'TransformerDecoder': keras_nlp.layers.TransformerDecoder})
    
    factors, expansions = load_file(os.path.join(__location__, filepath))
    
    print()
    print("Predicting Expansions...")
    pred = [predict(factors = f, 
                    model = polynomial_transformer,
                    f_vector = factor_vectorization,
                    e_vector = expand_vectorization,
                    e_lookup = expand_index_lookup,
                    seq_length = 29) for f in tqdm(factors)]
    
    scores = [score(te, pe) for te, pe in zip(expansions, pred)]
    
    print(np.mean(scores))

if __name__ == "__main__":
    main("test.txt" if "-t" in sys.argv else "train.txt")