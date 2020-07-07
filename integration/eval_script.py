import tensorflow as tf
import tensorflow_addons as tfa
import os
import io
import numpy as np
import re
import unicodedata
import shutil
import itertools
import pickle

def preprocess_sentence(s):
    s = unicode_to_ascii(s.lower().strip())
    s = s.lower().strip()
    
    s = re.sub(r"([?.!,¿])", r" \1 ", s)
    s = re.sub(r'[" "]+', " ", s)
    
    s = re.sub(r"[^a-zA-Z?.!,¿]+", " ", s)

    s = s.rstrip().strip()
   
    s = '<start> ' + s + ' <end>'
    return s
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

BATCH_SIZE = 128
embedding_dims = 200
rnn_units = 512
dense_units = 512
Dtype = tf.float32 
Tx=Ty=24  


with open('X_tokenizer.pickle', 'rb') as handle:
    X_tokenizer=Y_tokenizer=pickle.load(handle)
input_vocab_size = len(X_tokenizer.word_index)+1  
output_vocab_size = len(Y_tokenizer.word_index)+ 1


embedding_dim = 200
num_words= input_vocab_size
embedding_matrix = np.zeros((num_words, embedding_dim)) 
with open('embedding_matrix.pickle', 'rb') as handle:
    embedding_matrix=pickle.load(handle)

#ENCODER
class EncoderNetwork(tf.keras.Model):
    def __init__(self,input_vocab_size,embedding_dims, rnn_units ):
        super().__init__()
        # self.encoder_embedding = tf.keras.layers.Embedding(input_dim=input_vocab_size,
        #                                                    output_dim=embedding_dims)
        self.encoder_embedding = tf.keras.layers.Embedding(num_words, embedding_dim, input_length=Tx,weights=[embedding_matrix],trainable=False)
        self.encoder_rnnlayer = tf.keras.layers.LSTM(rnn_units,return_sequences=True, 
                                                     return_state=True )
        #self.encoder_rnnlayer = tf.keras.layers.LSTM(rnn_units,return_sequences=True, 
                                                     #return_state=True )
    
#DECODER
class DecoderNetwork(tf.keras.Model):
    def __init__(self,output_vocab_size, embedding_dims, rnn_units):
        super().__init__()
        # self.decoder_embedding = tf.keras.layers.Embedding(input_dim=output_vocab_size,
        #                                                    output_dim=embedding_dim) 
        self.decoder_embedding = tf.keras.layers.Embedding(num_words, embedding_dim, input_length=Tx,weights=[embedding_matrix],trainable=False)

        self.dense_layer = tf.keras.layers.Dense(output_vocab_size)
        self.decoder_rnncell = tf.keras.layers.LSTMCell(rnn_units)
        # self.decoder_rnncell = tf.keras.layers.LSTMCell(rnn_units)
        # Sampler
        self.sampler = tfa.seq2seq.sampler.TrainingSampler()
        # Create attention mechanism with memory = None
        self.attention_mechanism = self.build_attention_mechanism(dense_units,None,BATCH_SIZE*[Tx])
        self.rnn_cell =  self.build_rnn_cell(BATCH_SIZE)
        self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell, sampler= self.sampler,
                                                output_layer=self.dense_layer)

    def build_attention_mechanism(self, units,memory, memory_sequence_length):
        return tfa.seq2seq.LuongAttention(units, memory = memory, 
                                          memory_sequence_length=memory_sequence_length)
        #return tfa.seq2seq.BahdanauAttention(units, memory = memory, memory_sequence_length=memory_sequence_length)

    # wrap decodernn cell  
    def build_rnn_cell(self, batch_size ):
        rnn_cell = tfa.seq2seq.AttentionWrapper(self.decoder_rnncell, self.attention_mechanism,
                                                attention_layer_size=dense_units)
        return rnn_cell
    
    def build_decoder_initial_state(self, batch_size, encoder_state,Dtype):
        decoder_initial_state = self.rnn_cell.get_initial_state(batch_size = batch_size, 
                                                                dtype = Dtype)
        decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state) 
        return decoder_initial_state



encoderNetwork = EncoderNetwork(input_vocab_size,embedding_dims, rnn_units)
decoderNetwork = DecoderNetwork(output_vocab_size,embedding_dims, rnn_units)
optimizer = tf.keras.optimizers.Adam()


def loss_function(y_pred, y):

    sparsecategoricalcrossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                                  reduction='none')
    loss = sparsecategoricalcrossentropy(y_true=y, y_pred=y_pred)
    mask = tf.logical_not(tf.math.equal(y,0))   #output 0 for y=0 else output 1
    mask = tf.cast(mask, dtype=loss.dtype)
    loss = mask* loss
    loss = tf.reduce_mean(loss)
    return loss


checkpoint_dir = 'training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer = optimizer, encoderNetwork = encoderNetwork, 
                                 decoderNetwork = decoderNetwork)
status=checkpoint.restore(tf.train.latest_checkpoint('training_checkpoints')).expect_partial()
decoder_embedding_matrix = tf.train.load_variable(
    'training_checkpoints', 'decoderNetwork/decoder_embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUE')

def responder(input_raw):
    beam_width = 3
    input_lines = [preprocess_sentence(input_raw)]
    input_sequences = [[X_tokenizer.word_index[w] for w in line.split(' ')] for line in input_lines]
    input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences,
                                                                maxlen=Tx, padding='post')
    inp = tf.convert_to_tensor(input_sequences)

    inference_batch_size = 1
    encoder_initial_cell_state = [tf.zeros((inference_batch_size, rnn_units)),
                              tf.zeros((inference_batch_size, rnn_units))]
    encoder_emb_inp = encoderNetwork.encoder_embedding(inp)
    a, a_tx, c_tx = encoderNetwork.encoder_rnnlayer(encoder_emb_inp,
                                                initial_state =encoder_initial_cell_state)

    start_tokens = tf.fill([inference_batch_size],Y_tokenizer.word_index['<start>'])

    end_token = Y_tokenizer.word_index['<end>']

    decoder_input = tf.expand_dims([Y_tokenizer.word_index['<start>']]* inference_batch_size,1)
    decoder_emb_inp = decoderNetwork.decoder_embedding(decoder_input)

    encoder_memory = tfa.seq2seq.tile_batch(a, beam_width)
    decoderNetwork.attention_mechanism.setup_memory(encoder_memory)

    decoder_initial_state = decoderNetwork.rnn_cell.get_initial_state(batch_size = inference_batch_size* beam_width,dtype = Dtype)
    encoder_state = tfa.seq2seq.tile_batch([a_tx, c_tx], multiplier=beam_width)
    decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state) 

    decoder_instance = tfa.seq2seq.BeamSearchDecoder(decoderNetwork.rnn_cell,beam_width=beam_width,
                                                 output_layer=decoderNetwork.dense_layer)

    maximum_iterations = tf.round(tf.reduce_max(Tx) * 2)


    (first_finished, first_inputs,first_state) = decoder_instance.initialize(decoder_embedding_matrix,
                             start_tokens = start_tokens,
                             end_token=end_token,
                             initial_state = decoder_initial_state)
    inputs = first_inputs
    state = first_state  
    predictions = np.empty((inference_batch_size, beam_width,0), dtype = np.int32)
    beam_scores =  np.empty((inference_batch_size, beam_width,0), dtype = np.float32)                                                                            
    for j in range(maximum_iterations):
        beam_search_outputs, next_state, next_inputs, finished = decoder_instance.step(j,inputs,state)
        inputs = next_inputs
        state = next_state
        outputs = np.expand_dims(beam_search_outputs.predicted_ids,axis = -1)
        scores = np.expand_dims(beam_search_outputs.scores,axis = -1)
        predictions = np.append(predictions, outputs, axis = -1)
        beam_scores = np.append(beam_scores, scores, axis = -1)                                                                         
    print(input_raw)
    print("---------------------------------------------")
    output_beams_per_sample = predictions[0,:,:]
    score_beams_per_sample = beam_scores[0,:,:]
    best_response=""
    least_score=1000
    for beam, score in zip(output_beams_per_sample,score_beams_per_sample) :
        seq = list(itertools.takewhile( lambda index: index !=2, beam))
        score_indexes = np.arange(len(seq))
        beam_score = score[score_indexes].sum()
        response = " ".join( [Y_tokenizer.index_word[w] for w in seq])
        print(response, " beam score: ", beam_score)
        if beam_score<least_score:
            least_score=beam_score
            best_response= response
    return best_response    

import speech_recognition as sr  

def speechInput(): 
    r = sr.Recognizer()  
    with sr.Microphone() as source:  
        print("Please wait. Calibrating microphone...")  
        r.adjust_for_ambient_noise(source, duration=1)  
        print("Say something!")  
        audio = r.listen(source)   
    try: 
        speech= r.recognize_google(audio)
        print("You said: '" + speech + "'")
        return speech  
    except sr.UnknownValueError:  
        print("I could not understand audio :(")  
    except sr.RequestError as e:  
        print("Recog error; {0}".format(e))  

from gtts import gTTS
from time import sleep
import os
import pyglet

def speakResponse(response):
    tts = gTTS(text=response, lang='en')
    filename = '/tmp/temp.mp3'
    tts.save(filename)

    music = pyglet.media.load(filename, streaming=False)
    music.play()

    sleep(music.duration) 
    os.remove(filename) 

while input("Continue?(y/n)") is 'y':
    speech=speechInput()
    response=responder(speech)
    speakResponse(response)



