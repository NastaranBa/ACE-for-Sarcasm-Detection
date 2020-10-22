import pickle
import pprint
import json
import re
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
from tqdm import tqdm
from os import listdir
from os.path import isfile, join, exists
import os
import requests
import sys
from glob import glob
from itertools import chain
import time
import math
import numpy as np
import urllib.request
!pip install namegenerator
import namegenerator
!pip install bert-tensorflow
import bert
import ipywidgets as widgets
from bert import run_classifier
from bert import optimization
from bert import tokenization
from bert import modeling

#Settings and Defaults#
#GCS Bucket
#A GS bucket is a "gs://bucket" (where bucket is your bucket) location. If the bucket is not present under your GCS project ID, a bucket will be created with this code.

#tf_checkpoint_root
#This is the root location where new fine-tuned and trained BERT model checkpoints will be located.

#data_loc
#Where the source data for this notebook will be downloaded.

#data_cache
#Location for temporary files created for use by the TPU and other procedures.
##


#@title Settings and defaults
gcs_bucket = '<bucket name here>' #@param {type:"string"}
tf_checkpoint_root = 'tf_checkpoint_root' #@param {type:"string"}
#tf_hub_cache = 'tf_hub_cache' #@param {type:"string"}
data_loc = 'data_loc' #@param {type:"string"}
data_cache = 'data_cache' #@param {type:"string"}
project_id = '<input project id>' #@param {type:"string"}

access_api = '<input access api>' #@param {type:"string"}
access_api_private = '<input access api private>' #@param {type:"string"}


BUCKET_LOC = f'gs://{gcs_bucket}/'
TF_CHECKPOINT_ROOT = f'gs://{gcs_bucket}/{tf_checkpoint_root}'
DATA_DIR = f'gs://{gcs_bucket}/{data_loc}'
#GS_TF_HUB_CACHE = f'gs://{gcs_bucket}/{tf_hub_cache}'
DATA_CACHE = f'gs://{gcs_bucket}/{data_cache}'


# Now credentials are set for all future sessions on this TPU.
!gcloud config set project {project_id}

from google.colab import auth
auth.authenticate_user()

assert 'COLAB_TPU_ADDR' in os.environ, 'ERROR: Not connected to a TPU runtime; please see the first cell in this notebook for instructions!'
TPU_ADDRESS = 'grpc://' + os.environ['COLAB_TPU_ADDR']
print('TPU address is', TPU_ADDRESS)
with tf.Session(TPU_ADDRESS) as session:
  print('TPU devices:')
  pprint.pprint(session.list_devices())

  # Upload credentials to TPU.
  with open('/content/adc.json', 'r') as f:
    auth_info = json.load(f)
  tf.contrib.cloud.configure_gcs(session, credentials=auth_info)

print("All defaults successfully established")

label_list = [0,1]
use_TPU = True
random_seed = 42

# Create any buckets or bucket folders needed for code execution.

bucket_status = !gsutil ls {BUCKET_LOC}
if "AccessDeniedException" in bucket_status:
    !gsutil mb -p {project_id} {BUCKET_LOC}
    print(f'Created {BUCKET_LOC}')
else:
    print(f'{BUCKET_LOC} already exists.')
    
if not tf.io.gfile.exists(TF_CHECKPOINT_ROOT):
    print(f'Created {TF_CHECKPOINT_ROOT}')
    tf.gfile.MakeDirs(TF_CHECKPOINT_ROOT)
else:
    print(f'{TF_CHECKPOINT_ROOT} already exists.')
        
if not tf.io.gfile.exists(DATA_DIR):
    print(f'Created {DATA_DIR}')
    tf.gfile.MakeDirs(DATA_DIR)
else:
    print(f'{DATA_DIR} already exists.')
          
if not tf.io.gfile.exists(DATA_CACHE):
    print(f'Created {DATA_CACHE}')
    tf.gfile.MakeDirs(DATA_CACHE)
else:
    print(f'{DATA_CACHE} already exists.')

bert_model_urls = ["https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip",
                   "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip",
                   "https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip",
                   "https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip"]

# If there are no sub-folders located in your tensorflow checkpoint root,
# this code will create one root to start with

list_of_tf_roots = tf.io.gfile.listdir(TF_CHECKPOINT_ROOT)
    
if len(list_of_tf_roots) == 0:
    print("There are no models present in your tf checkpoint root. \n Create a new model using this name or a name of your choice:")
    cur_name = namegenerator.gen()
    text_input = input(f'Enter a model name, or <ENTER> for {cur_name}:')
    if text_input is '':
        text_input=cur_name
    print(f'Creating folder for {text_input}')
    tf.gfile.MakeDirs(TF_CHECKPOINT_ROOT + '/' + text_input)
    print("Folder created")
	
	print("Select a bert model to use")
bert_model_choice = widgets.Dropdown(options=bert_model_urls, value="https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip")
bert_model_choice

print("Select a location for the tensorflow model.")
tf_checkpoint_sub_dir = widgets.Dropdown(options=tf.io.gfile.listdir(TF_CHECKPOINT_ROOT), value=tf.io.gfile.listdir(TF_CHECKPOINT_ROOT)[0])
tf_checkpoint_sub_dir

print(f'Do you want to download the BERT model into {tf_checkpoint_sub_dir.value}?')
should_download = widgets.Dropdown(options=['Yes','No'], value='Yes')
should_download
def create_and_populate_checkpoint_folder(subfolder_name, bert_model_name_with_zip):
    try:
        
        # Check to see if there is a trained model in this folder, in this case,
        # halt and ask the user what to do.
        
        if tf.gfile.Exists(TF_CHECKPOINT_ROOT + '/' + subfolder_name +  'checkpoint'):
            print("There appears to be an existing checkpoint in this sub-folder.")
            print("Do you want to wipe this subfolder and insert a new BERT step 0 checkpoint?")
            answer = input("Type \'yes\' and <Enter> if you want to wipe. Type any other character to not wipe.'")
            if answer == "yes":
                !gsutil rm {TF_CHECKPOINT_ROOT + '/' + subfolder_name + '*'}
            else:
                print("Ending function early.")
                return True

        
        bert_file_name = bert_model_name_with_zip.split('/')[-1]
        print(f'Retrieving {bert_file_name}')

        # Check to see if this file has already been downloaded. If so,
        # skip the download step

        if not exists(bert_model_choice.value):
            urllib.request.urlretrieve(bert_model_choice.value, bert_file_name)
            print(f'{bert_file_name} successfully downloaded')
        else:
            print(f'{bert_file_name} already downloaded')
        # Re-unzip the bert model, just in case something funky happened
        !unzip -o {bert_file_name}
        !cd {bert_file_name.split('.')[0]};gsutil cp * {TF_CHECKPOINT_ROOT + '/' + subfolder_name}

        return True
    except Exception as e:
              print(e)
              return False
# If you chose to download the bert model, the following function will
# download and unzip the model in the designated location

if should_download.value == "Yes":
    create_and_populate_checkpoint_folder(subfolder_name = tf_checkpoint_sub_dir.value,
                                         bert_model_name_with_zip = bert_model_choice.value)
										 
										 
def clean_tweet_text(in_text):
    
    try:
        
        # remove URLS
        # regex from https://regexr.com/36fcc
        url_re = re.compile(r'(http|ftp|https)://([\w+?\.\w+])+([a-zA-Z0-9\~\!\@\#\$\%\^\&\*\(\)_\-\=\+\\\/\?\.\:\;\'\,]*)?')
        in_text = url_re.sub("",in_text)

        # remove hashtags
        hashtag_re = re.compile(r'(#\w+)')
        in_text = hashtag_re.sub("",in_text)

        # remove screen name references
        screen_name_re = re.compile(r'(@\w+)')
        in_text = screen_name_re.sub("", in_text)

        # remove RTs
        in_text = in_text.replace("RT","")

        # strip text of extra spaces , and keep one space between each word
        in_text = " ".join(in_text.split())

        # strip text of emjoi
        # https://stackoverflow.com/questions/51217909/removing-all-emojis-from-text
        emoji_re = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U0001F1F2-\U0001F1F4"  # Macau flag
            u"\U0001F1E6-\U0001F1FF"  # flags
            u"\U0001F600-\U0001F64F"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U0001F1F2"
            u"\U0001F1F4"
            u"\U0001F620"
            u"\u200d"
            u"\u2640-\u2642"
            "]+", flags=re.UNICODE)

        in_text = emoji_re.sub("",in_text)
    
    except:
        in_text = ""
    return in_text
	
# Importing our BERT-specific functions

import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
from bert import modeling

# Adopted from https://github.com/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb

DATA_COLUMN = 'text'
LABEL_COLUMN = 'label'

train_InputExamples = train_labeled_set.apply(lambda x: bert.run_classifier.InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this example
                                                                   text_a = x[DATA_COLUMN], 
                                                                   text_b = None, 
                                                                   label = x[LABEL_COLUMN]), axis = 1)

test_InputExamples = test_labeled_set.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
                                                                   text_a = x[DATA_COLUMN], 
                                                                   text_b = None, 
                                                                   label = x[LABEL_COLUMN]), axis = 1)

def create_tokenizer():
  
  """
  Reviews the type of BERT module selected earlier, 
  and create the appropriate tokenizer.
  
  A tokenizer is required for inputing text into BERT
  """
    
  do_lower_case = False
  if 'uncased' in bert_model_choice.value:
        cased_option = False
  else:
        cased_option = True
        
  vocab_file = TF_CHECKPOINT_ROOT + '/' + tf_checkpoint_sub_dir.value + 'vocab.txt'
  
  with tf.Graph().as_default():      
      return bert.tokenization.FullTokenizer(
          vocab_file=vocab_file, do_lower_case=do_lower_case)

tokenizer = create_tokenizer()

# We'll set sequences to be at most 256 tokens long
# Note, this is not 256 words, but 256 tokens
# obviously, for tweets, this should be sufficient
MAX_SEQ_LENGTH = 256

# Convert our train and test features to InputFeatures that BERT understands.
# Save the results as a TFRecord, which will be copied to Google Cloud Storage,
# so, that the TPU can read the TFRecord directly

train_features = bert.run_classifier.file_based_convert_examples_to_features(train_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer,'train_features.TFRecord')
test_features = bert.run_classifier.file_based_convert_examples_to_features(test_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer,'test_features.TFRecord')

# Copy the generated TFRecords to the data_cache, so that the TPU may directly access

!gsutil cp train_features.TFRecord {DATA_CACHE}/train_features.TFRecord
!gsutil cp test_features.TFRecord {DATA_CACHE}/test_features.TFRecord

# Adopted from https://github.com/google-research/bert/blob/ffbda2a1aafe530525212d13194cc84d92ed0313/run_classifier.py#L574

def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # Since we are classifying sentences, we are using pooled, not
  # token output.
  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, logits, probabilities)
# Adopted from https://github.com/google-research/bert/blob/ffbda2a1aafe530525212d13194cc84d92ed0313/run_classifier.py#L619

def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, logits, probabilities) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(per_example_loss, label_ids, logits):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(label_ids, predictions)
        loss = tf.metrics.mean(values=per_example_loss)
        f1_score = tf.contrib.metrics.f1_score(label_ids,predictions)
        auc = tf.metrics.auc(label_ids,predictions)
        recall = tf.metrics.recall(label_ids,predictions)
        precision = tf.metrics.precision(label_ids,predictions) 
        true_pos = tf.metrics.true_positives(label_ids,predictions)
        true_neg = tf.metrics.true_negatives(label_ids,predictions)   
        false_pos = tf.metrics.false_positives(label_ids,predictions)  
        false_neg = tf.metrics.false_negatives(label_ids,predictions)

        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
            "eval_f1_score": f1_score,
            "eval_auc": auc,
            "eval_recall": recall,
            "eval_precision": precision,
            "eval_true_pos": true_pos,
            "eval_true_neg": true_neg,
            "eval_false_pos": false_pos,
            "eval_false_neg": false_neg,
        }

      eval_metrics = (metric_fn,
                      [per_example_loss, label_ids, logits])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities},
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn
  
 # These default settings are recommended. You can change based on the experiment.

TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32
PREDICT_BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 6.0
# Warmup is a period of time where hte learning rate 
# is small and gradually increases--usually helps training.
WARMUP_PROPORTION = 0.1
# Model configs
SAVE_CHECKPOINTS_STEPS = 5000
SAVE_SUMMARY_STEPS = 2500
NUM_TPU_CORES = 8
OUTPUT_DIR = TF_CHECKPOINT_ROOT + '/' + tf_checkpoint_sub_dir.value
ITERATIONS_PER_LOOP = 1000
MAX_SEQ_LENGTH= 256

# Adopted from https://github.com/google-research/bert/blob/bee6030e31e42a9394ac567da170a89a98d2062f/run_classifier_with_tfhub.py#L180

# Create an object to represent the TPU cluster
tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(TPU_ADDRESS)

# Create our configuration file
run_config = tf.contrib.tpu.RunConfig(
    cluster=tpu_cluster_resolver,
    model_dir=OUTPUT_DIR,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
    tpu_config=tf.contrib.tpu.TPUConfig(
        iterations_per_loop=ITERATIONS_PER_LOOP,
        num_shards=NUM_TPU_CORES,
        per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))
		
# Compute # train and warmup steps from batch size
num_train_steps = int(len(train_InputExamples) / TRAIN_BATCH_SIZE * NUM_TRAIN_EPOCHS)
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)
print(f'Number of training steps is {num_train_steps}, and number of warmup steps is {num_warmup_steps}')

# Adopted from https://github.com/google-research/bert/blob/bee6030e31e42a9394ac567da170a89a98d2062f/run_classifier_with_tfhub.py#L89
# Generate a model from our model builder factory  
    
model_fn = model_fn_builder(
    bert_config=  modeling.BertConfig.from_json_file(TF_CHECKPOINT_ROOT + '/' + tf_checkpoint_sub_dir.value + 'bert_config.json'),
    num_labels=len(label_list),
    init_checkpoint=TF_CHECKPOINT_ROOT + '/' + tf_checkpoint_sub_dir.value + 'bert_model.ckpt',
    learning_rate=2e-5,
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps,
    use_tpu=True,
    use_one_hot_embeddings=True)
	
# Create the estimator
    
estimator = tf.contrib.tpu.TPUEstimator(
use_tpu=True,
model_fn=model_fn,
config=run_config,
train_batch_size=TRAIN_BATCH_SIZE,
eval_batch_size=EVAL_BATCH_SIZE,
predict_batch_size=PREDICT_BATCH_SIZE)

# Create an input function for training, using the TFRecord we generated

train_input_fn = bert.run_classifier.file_based_input_fn_builder(
        input_file=DATA_CACHE + '/train_features.TFRecord',
        seq_length=MAX_SEQ_LENGTH,
        is_training=True,
        drop_remainder=True)
		
print(f'Beginning Training!')
current_time = datetime.now()
estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
print("Training took time ", datetime.now() - current_time)

test_input_fn = bert.run_classifier.file_based_input_fn_builder(
        input_file=DATA_CACHE + '/test_features.TFRecord',
        seq_length=MAX_SEQ_LENGTH,
        is_training=False,
        drop_remainder=True)
		
# The calculation for evaluation steps is separate from the calculation for training steps
eval_steps = int(len(test_InputExamples) / EVAL_BATCH_SIZE)
print(eval_steps)

results_dict = estimator.evaluate(input_fn=test_input_fn, steps=eval_steps)
print(results_dict)


# Adopted from https://github.com/google-research/bert/blob/0a0ea64a3ac1f43ed27d75278b9578708f9febcf/predicting_movie_reviews_with_bert_on_tf_hub.ipynb#L1090

def getPrediction(in_sentences):
  ret = []
  try:
        
      labels = [0, 1]
      input_examples = [run_classifier.InputExample(guid="", text_a = x, text_b = None, label = 0) for x in in_sentences] # here, "" is just a dummy label
      input_features = run_classifier.file_based_convert_examples_to_features(input_examples, label_list, MAX_SEQ_LENGTH, tokenizer,'predict_features.TFRecord')
      !gsutil cp predict_features.TFRecord {DATA_CACHE}/predict_features.TFRecord
      predict_input_fn = run_classifier.file_based_input_fn_builder(input_file=DATA_CACHE + '/predict_features.TFRecord', seq_length=MAX_SEQ_LENGTH, is_training=False, drop_remainder=True)
      predictions = estimator.predict(predict_input_fn)
      #ret = [(sentence, prediction['probabilities'], labels[prediction['labels']]) for sentence, prediction in zip(in_sentences, predictions)]
      for p in predictions:
            ret.append(p)
  except IndexError:
      return(ret)
  return(ret)
  
# Matching batch size
pred_sentences = pred_sentences * 8
preds = getPrediction(pred_sentences)

import numpy as np
pred_probs = [np.argmax(p['probabilities']) for p in preds]
final_predictions = list(zip(pred_sentences, pred_probs))
print(final_predictions)

## Adding Affective features to the training from AFE





print("Select a location for the tensorflow model. Run the function two cells down to create a new folder.")
extra_features_model_folder = widgets.Dropdown(options=tf.io.gfile.listdir(TF_CHECKPOINT_ROOT), value=tf.io.gfile.listdir(TF_CHECKPOINT_ROOT)[0])
extra_features_model_folder

# assign the var
extra_features_model_folder = str(extra_features_model_folder.value[:-1])
print(f'Working from {extra_features_model_folder} as {type(extra_features_model_folder)}')


## **Only run if you want to create a new folder. Otherwise, select a model from above.
import urllib
cur_name = namegenerator.gen()
text_input = input(f'Enter a model name, or <ENTER> for {cur_name}:')
if text_input is '':
    text_input=cur_name
print(f'Creating folder for {text_input}')
tf.gfile.MakeDirs(TF_CHECKPOINT_ROOT + '/' + text_input)
extra_features_model_folder = text_input
bert_file_name = bert_model_choice.value.split('/')[-1]
print(bert_file_name)
urllib.request.urlretrieve(bert_model_choice.value, bert_file_name )
print("downloaded")
!ls
!unzip -o {bert_file_name}
!cd {bert_file_name.split('.')[0]};gsutil cp * {TF_CHECKPOINT_ROOT + '/' + extra_features_model_folder}

# to add our extra features

# From https://github.com/google-research/bert/blob/ffbda2a1aafe530525212d13194cc84d92ed0313/run_classifier.py#L161


class InputExampleExtraFeatures(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None,extra_features=None):
    """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
      extra_features: 1-D numpy array of extra features
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label
    self.extra_features = extra_features
	
# From https://github.com/google-research/bert/blob/ffbda2a1aafe530525212d13194cc84d92ed0313/run_classifier.py#L479

def file_based_convert_examples_to_features_with_extra_features(
    examples, label_list, max_seq_length, tokenizer, output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example_extra_features(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f
    
    def create_float_feature(values):
      f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
      return f
    

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_int_feature([feature.label_id])

    # Adding extra feature here
    features["extra_features"] = create_float_feature(feature.extra_features)
    features["is_real_example"] = create_int_feature(
        [int(feature.is_real_example)])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()
  
# From https://github.com/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb

DATA_COLUMN = 'text'
LABEL_COLUMN = 'label'

train_InputExamples_extra_features = train_extra_features.apply(lambda x: InputExampleExtraFeatures(guid=None, # Globally unique ID for bookkeeping, unused in this example
                                                                   text_a = x[DATA_COLUMN], 
                                                                   text_b = None, 
                                                                   label = x[LABEL_COLUMN], # adding extra features
                                                                   extra_features = x['extra_features']), axis = 1)

test_InputExamples_extra_features = test_extra_features.apply(lambda x: InputExampleExtraFeatures(guid=None, 
                                                                   text_a = x[DATA_COLUMN], 
                                                                   text_b = None, 
                                                                   label = x[LABEL_COLUMN], # adding extra features
                                                                   extra_features = x['extra_features']), axis = 1)
																   
# From https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L161

import collections

class PaddingInputExample_ExtraFeatures(object):
    pass

class InputFeatures_ExtraFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               extra_features,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.extra_features = extra_features
    self.is_real_example = is_real_example

def convert_single_example_extra_features(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample_ExtraFeatures):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        label_id=0,
        # if this is padding, insert 3 zeros for extra features
        extra_features=[0] * num_of_extra_features,
        is_real_example=False)

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  label_id = label_map[example.label]
  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label: %s (id = %d)" % (example.label, label_id))
    tf.logging.info("extra_features: %s" % " ".join([str(x) for x in example.extra_features]))
    

  feature = InputFeatures_ExtraFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      extra_features=example.extra_features,
      is_real_example=True)
  return feature
  
MAX_SEQ_LENGTH = 256
label_list = [0,1]
train_features = file_based_convert_examples_to_features_with_extra_features(train_InputExamples_extra_features, label_list, MAX_SEQ_LENGTH, tokenizer,'train_extra_features.TFRecord')
test_features = file_based_convert_examples_to_features_with_extra_features(test_InputExamples_extra_features, label_list, MAX_SEQ_LENGTH, tokenizer,'test_extra_features.TFRecord')

!gsutil cp train_extra_features.TFRecord {DATA_CACHE}/train_extra_features.TFRecord
!gsutil cp test_extra_features.TFRecord {DATA_CACHE}/test_extra_features.TFRecord
# Uncomment these lines if you want to copy from data_cache to your local
# 
#!gsutil cp {DATA_CACHE}/train_extra_features.TFRecord train_extra_features.TFRecord 
#!gsutil cp {DATA_CACHE}/test_extra_features.TFRecord test_extra_features.TFRecord
#

# Adopted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L574

def create_model_extra_features(bert_config, is_training, input_ids, input_mask, segment_ids,extra_features,
                 labels, num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)


  output_layer = model.get_pooled_output()
  # Here, we make alterations to add the extra features
  output_layer_extra_features = tf.concat([output_layer,tf.convert_to_tensor(extra_features, dtype=tf.float32)],axis=1)  
    
  hidden_size = output_layer_extra_features.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer_extra_features = tf.nn.dropout(output_layer_extra_features, keep_prob=0.9)

    logits = tf.matmul(output_layer_extra_features, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, logits, probabilities)
	
#Adopted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L509

def file_based_input_fn_builder_extra_features(input_file, seq_length, is_training,
                                drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([], tf.int64),
      "extra_features": tf.FixedLenFeature([num_of_extra_features], tf.float32), #Adding extra features
      "is_real_example": tf.FixedLenFeature([], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t
    
    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn
  
# Adopted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L619

def model_fn_builder_extra_features(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    extra_features = features["extra_features"] # Adding extra features

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, logits, probabilities) = create_model_extra_features(
        bert_config, is_training, input_ids, input_mask, segment_ids, extra_features, label_ids,
        num_labels, use_one_hot_embeddings)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(per_example_loss, label_ids, logits):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(label_ids, predictions)
        loss = tf.metrics.mean(values=per_example_loss)
        f1_score = tf.contrib.metrics.f1_score(label_ids,predictions)
        auc = tf.metrics.auc(label_ids,predictions)
        recall = tf.metrics.recall(label_ids,predictions)
        precision = tf.metrics.precision(label_ids,predictions) 
        true_pos = tf.metrics.true_positives(label_ids,predictions)
        true_neg = tf.metrics.true_negatives(label_ids,predictions)   
        false_pos = tf.metrics.false_positives(label_ids,predictions)  
        false_neg = tf.metrics.false_negatives(label_ids,predictions)

        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
            "eval_f1_score": f1_score,
            "eval_auc": auc,
            "eval_recall": recall,
            "eval_precision": precision,
            "eval_true_pos": true_pos,
            "eval_true_neg": true_neg,
            "eval_false_pos": false_pos,
            "eval_false_neg": false_neg,
        }

      eval_metrics = (metric_fn,
                      [per_example_loss, label_ids, logits])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities},
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn
  
# Compute # train and warmup steps from batch size
num_train_steps = int(len(train_InputExamples_extra_features) / TRAIN_BATCH_SIZE * 6)
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)
print(f'Number of training steps is {num_train_steps}, and number of warmup steps is {num_warmup_steps}')

model_fn_extra_features = model_fn_builder_extra_features(
  bert_config=  modeling.BertConfig.from_json_file(TF_CHECKPOINT_ROOT + '/' + extra_features_model_folder + '/bert_config.json'),
  num_labels=len(label_list),
  init_checkpoint=TF_CHECKPOINT_ROOT + '/' + extra_features_model_folder + '/bert_model.ckpt',
  learning_rate=2e-5,
  num_train_steps=num_train_steps,
  num_warmup_steps=num_warmup_steps,
  use_tpu=True,
  use_one_hot_embeddings=True)
  
run_config_extra_features = tf.contrib.tpu.RunConfig(
    cluster=tpu_cluster_resolver,
    model_dir=TF_CHECKPOINT_ROOT + '/' + extra_features_model_folder,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
    tpu_config=tf.contrib.tpu.TPUConfig(
        iterations_per_loop=ITERATIONS_PER_LOOP,
        num_shards=NUM_TPU_CORES,
        per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))
		
estimator_extra_features = tf.contrib.tpu.TPUEstimator(
use_tpu=True,
model_fn=model_fn_extra_features,
config=run_config_extra_features,
train_batch_size=TRAIN_BATCH_SIZE,
eval_batch_size=EVAL_BATCH_SIZE,
predict_batch_size=PREDICT_BATCH_SIZE)

# Create an input function for training. drop_remainder = True for using TPUs.

train_input_fn_extra_features = file_based_input_fn_builder_extra_features(
        input_file=DATA_CACHE + '/train_extra_features.TFRecord',
        seq_length=MAX_SEQ_LENGTH,
        is_training=True,
        drop_remainder=True)
		
print(f'Beginning Training!')
current_time = datetime.now()
estimator_extra_features.train(input_fn=train_input_fn_extra_features, max_steps=num_train_steps)
print("Training took time ", datetime.now() - current_time)

test_input_fn_extra_features = file_based_input_fn_builder_extra_features(
        input_file=DATA_CACHE + '/test_extra_features.TFRecord',
        seq_length=MAX_SEQ_LENGTH,
        is_training=False,
        drop_remainder=True)
		
eval_steps = int(len(test_InputExamples_extra_features) / EVAL_BATCH_SIZE)
print(eval_steps)

results_dict_extra_features = estimator_extra_features.evaluate(input_fn=test_input_fn_extra_features, steps=eval_steps)

print(results_dict_extra_features)

