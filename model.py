from tensorflow import keras
import keras_nlp

def create_transformer_seq2seq(model_params):
    """ A Transformer seq2seq model with a positional embedding 
    with 2 additional layers for both the encoder and decoder. 

    Args:
        model_params (dict): model hyperparameters

    Returns:
        Object: The Transformer Model.
    """
    factorized_vocab_size = model_params['FACTOR_VOCAB_SIZE']         
    expanded_vocab_size = model_params['EXPANDED_VOCAB_SIZE']
    inter_dim = model_params['INTERMEDIATE_DIM']
    embed_dim = model_params['EMBED_DIM']
    attention_heads = model_params['NUM_HEADS']
    seq_length = model_params['MAX_SEQUENCE_LENGTH']
    dropout_rate = model_params["DROPOUT_RATE"]

    # Encoder Layers
    encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")

    x = keras_nlp.layers.TokenAndPositionEmbedding(
        vocabulary_size =  factorized_vocab_size,
        sequence_length = seq_length,
        embedding_dim = embed_dim,
        mask_zero = True,
    )(encoder_inputs)

    x = keras_nlp.layers.TransformerEncoder(intermediate_dim = inter_dim, 
                                            num_heads = attention_heads,
                                            dropout = dropout_rate)(inputs=x)

    encoder_outputs = keras_nlp.layers.TransformerEncoder(intermediate_dim = inter_dim, 
                                                        num_heads = attention_heads)(inputs = x)

    encoder = keras.Model(encoder_inputs, encoder_outputs)


    # Decoder Layers
    decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
    encoded_seq_inputs = keras.Input(shape=(None, embed_dim), name="decoder_state_inputs")

    x = keras_nlp.layers.TokenAndPositionEmbedding(
        vocabulary_size = expanded_vocab_size,
        sequence_length = seq_length,
        embedding_dim = embed_dim,
        mask_zero = True,
    )(decoder_inputs)

    x = keras_nlp.layers.TransformerDecoder(
        intermediate_dim = inter_dim, 
        num_heads = attention_heads,
        dropout = dropout_rate
    )(x)

    x = keras_nlp.layers.TransformerDecoder(
        intermediate_dim = inter_dim, 
        num_heads = attention_heads
    )(decoder_sequence = x, 
    encoder_sequence = encoded_seq_inputs)


    x = keras.layers.Dropout(dropout_rate)(x)

    decoder_outputs = keras.layers.Dense(expanded_vocab_size, 
                                        activation = "softmax")(x)
    decoder = keras.Model(
        [
            decoder_inputs,
            encoded_seq_inputs,
        ],
        decoder_outputs,
    )

    decoder_outputs = decoder([decoder_inputs, encoder_outputs])

    # The End to End model.
    transformer = keras.Model(
        [encoder_inputs, decoder_inputs],
        decoder_outputs,
        name="transformer",
    )

    return transformer