import torch

from data import Data
from model import Models


class Pipeline:
    def __init__(self, model: Models, data: Data):
        self.model = model.eval()
        self.data = data

    def run(self, sentences):
        self.data.batch_encode_sentence(sentences, inplace=True)
        dataloader = self.data.to_dataloader()
        output = []
        with torch.no_grad():
            for batch in dataloader:
                model_output = self.model(
                    batch.input_ids, batch.word_begin_index, batch.labels
                )
                batch_sentences = [
                    self.data.sentences[i] for i in batch.sentence_index.tolist()
                ]
                # output dict prediction and scores of shape batch_size, depth, num_word)
                # batch_sentences (batch_size)
                out = self.data.batch_decode_prediction(
                    **model_output, sentences=batch_sentences
                )
                output.extend(out)
        return output
