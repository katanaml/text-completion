import tensorflow as tf
from transformers import TFDistilBertForMaskedLM


class DistilbertConverter(object):
    def __init__(self):
        pass

    def call(self):
        # Doesn't work with AutoModelForMaskedLM
        distilbert = TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

        concrete_function = tf.function(distilbert.call).get_concrete_function(
            [tf.TensorSpec([None, 384], tf.int32, name="input_ids"),
             tf.TensorSpec([None, 384], tf.int32, name="attention_mask")])

        tf.saved_model.save(distilbert,
                            'saved_model/distilbert-base-uncased',
                            signatures=concrete_function)

        # saved_model_cli show --dir saved_model/distilbert-base-uncased --tag_set serve --signature_def serving_default
