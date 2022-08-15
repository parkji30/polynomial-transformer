import os
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

from model import create_transformer_seq2seq
from train_helpers import *
from keras.backend import clear_session
from keras import callbacks
from keras.callbacks import ModelCheckpoint
import pickle

if __name__ == '__main__':
    clear_session()
    
    # Path to training text file.
    training_file = os.path.join(__location__, 'train.txt')
    
    # Preprocess Data, create Vocabulary List.
    input_vocab, target_vocab, pairs = create_vocabs(training_file)
    
    # Creating train, test, val sets.
    train_pairs, val_pairs, test_pairs = get_training_pairs(pairs)
    
    # Hyperparameters to build our model.
    # Our transformer seq2seq model has 3 Layers
    hyper_params_dict = {
        'BATCH_SIZE': 64,
        'EPOCHS': 12,
        'MAX_SEQUENCE_LENGTH': 29,
        'FACTOR_VOCAB_SIZE': 90,
        'EXPANDED_VOCAB_SIZE': 650,
        'EMBED_DIM': 256,
        'INTERMEDIATE_DIM': 512,
        'NUM_HEADS': 8,
        'DROPOUT_RATE': 0.1
    }
        
    factor_vectorization, expand_vectorization = create_TextVectorizations(hyper_params_dict, input_vocab, target_vocab)

    def format_dataset(factor, expand):
        """ Format dataset to be matrices using respective Vectorization
        objects.
        """
        factor = factor_vectorization(factor)
        expand = expand_vectorization(expand)
        return (
            {
                "encoder_inputs": factor,
                "decoder_inputs": expand[:, :-1],
            },
            expand[:, 1:],
        )

    def make_dataset(pairs):
        """Generator for preprocessing data to train on GPU.
        """
        factors, expansions = zip(*pairs)
        factors = list(factors)
        expansions = list(expansions)
        dataset = tf.data.Dataset.from_tensor_slices((factors, expansions))
        dataset = dataset.batch(64)
        dataset = dataset.map(format_dataset)
        return dataset.shuffle(2048).prefetch(16).cache()

    train_ds = make_dataset(train_pairs)
    val_ds = make_dataset(val_pairs)    
  
    # Returns a 3 Layer seq2seq model with the defined hyperparameters.
    polynomial_transformer = create_transformer_seq2seq(hyper_params_dict)
    polynomial_transformer.summary()
    
    # Create callbacks for saving models at each epochs
    # I found that 12 epochs returns the best model.
    rlrp = callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                       factor=0.1, 
                                       patience=10, 
                                       min_delta=1E-7)
    
    early_stop = callbacks.EarlyStopping(monitor='val_loss', 
                                         mode='min', 
                                         patience=10, 
                                         verbose=0,
                                        restore_best_weights=True)
    
    filepath = os.path.join(__location__, "saved_models/polynomial_transformers" + \
                          "_heads-" + str(hyper_params_dict['NUM_HEADS']) + \
                          "_embed_dim-" + str(hyper_params_dict['EMBED_DIM']) + \
                          "_inter_dim-" + str(hyper_params_dict['INTERMEDIATE_DIM']) + \
                          "_val_loss-{val_loss:.6f}.h5") 
    
    checkpoint = ModelCheckpoint(filepath, 
                                 monitor = 'val_loss', 
                                 verbose = 1, 
                                 save_best_only = True, 
                                 mode = 'min')
    callbacks_list = [early_stop, rlrp, checkpoint]
    
    polynomial_transformer.compile("rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    polynomial_transformer.fit(train_ds, epochs=hyper_params_dict['EPOCHS'], validation_data=val_ds, callbacks=callbacks_list)
    
    # Save Best Model into saved_models folder
    print("Saving Best Model.")
    polynomial_transformer.save(os.path.join(__location__, "saved_model/best_model.h5"))
    polynomial_transformer.save_weights(os.path.join(__location__, "saved_model/best_weights.h5"))
    
    # # Save Vocabulary into saved_models folder
    expand_vocab = expand_vectorization.get_vocabulary()
    expand_index_lookup = dict(zip(range(len(expand_vocab)), expand_vocab))
    
    
    print("Dumping Vocabularies and Index as pickles.")
    # Dump the lookup table into saved_models folder
    pickle.dump(expand_index_lookup, 
                open(os.path.join(__location__, 'saved_model/expand_lookup_index.pickle'), 'wb'))
    
    # Dump the factor vectorizer into saved_models folder
    pickle.dump({'config': factor_vectorization.get_config(),
                'weights': factor_vectorization.get_weights()}, 
                open(os.path.join(__location__, "saved_model/factor_vectorization.pickle"), "wb"))

    # Dump the expand vectorizer into saved_models folder
    pickle.dump({'config': expand_vectorization.get_config(),
                'weights': expand_vectorization.get_weights()},
                open(os.path.join(__location__, "saved_model/expand_vectorization.pickle"), "wb"))
    
    print("Training completed. Files saved in saved_models folder.")