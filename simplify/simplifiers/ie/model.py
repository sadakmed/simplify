import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from data import dot_dict
from utils import get_weight, read_config_file


class IEModel(nn.Module):
    def __init__(self, config):
        self.config = config

    def forward(self, ):
        ...

class IEConfig:
    def __init__(self,
                 backbone_model:'bert-large-cased',
                 max_depth:int=3,
                 num_iterative_layers:int=2,
                 dropout_rate:float=0.0,
                 label_hidden_size:int=300,
                 label_size:int=100,
                 num_labels:int=6,
                 max_word_length:int=100,
                 additional_special_tokens:List[str]=[ "[unused1]", "[unused2]", "[unused3]"],
                 **kwargs):

        self.backbone_model=   backbone_model
        self.max_depth=   max_depth
        self.num_iterative_layers=   num_iterative_layers
        self.dropout_rate=   dropout_rate
        self.label_hidden_size=   label_hidden_size
        self.label_size=   label_size
        self.num_labels=   num_labels
        self.max_word_length=   max_word_length
        self.additional_special_tokens=   additional_special_tokens
        self.kwargs = kwargs

    @classmethod
    def load_config(cls, config_file):
        config = read_config_file(config_file)
        return cls(**config)

    def save_config(self, file_path):
        with open(file_path, 'w') as f:
            json.dump(self.config, f)

config = {

    "max_depth": 3,
    "model_path": "bert-large-cased",
    "num_iterative_layers": 2,
    "dropout_rate": 0.0,
    "label_size":100,
    "label_hidden_size":300,
    "num_labels": 6,
    "additional_special_tokens": [ "[unused1]", "[unused2]", "[unused3]"],
    "max_word_length": 100,
}

class Models:
    def __init__(self, config):
        self.config = dot_dict(config)
        self.config.num_labels = self.config.num_labels or 6
        self.config.label_size = self.config.label_size or 100
        self.config.label_hidden_size = self.config.label_hidden_size or 300

        self.config.num_iterative_layers = self.config.num_iterative_layers or 2
        self.config.drop_rate = self.config.drop_rate or 0.0

    def save_pretrained(self, path):
        torch.save({"config": dict(self.config), "state_dict": self.state_dict()}, path)

    @classmethod
    def load_pretrained(cls, path, device="cpu"):
        path_file = get_weight(path)
        config_and_state_dict = torch.load(path_file, map_location=device)
        model = cls(config_and_state_dict["config"])
        model.load_state_dict(config_and_state_dict["state_dict"])
        return model

    def is_valid_extraction(self, prediction):
        return any([(1 in p and 2 in p) for p in prediction])


class BackboneModel(nn.Module, Models):
    def __init__(self, config):
        nn.Module.__init__(self)
        Models.__init__(self, config)

        model_config = AutoConfig.from_pretrained(self.config.model_path)
        self.base_model = AutoModel.from_config(model_config)
        self.hidden_size = self.base_model.config.hidden_size

        if self.config.num_iterative_layers == 0:
            self.iterative_layers = []
        else:
            self.iterative_layers = self.base_model.encoder.layer[
                -self.config.num_iterative_layers :
            ]
            self.base_model.encoder.layer = self.base_model.encoder.layer[
                : -self.config.num_iterative_layers
            ]
        self.dropout = nn.Dropout(self.config.drop_rate)

        self.label_embedding = nn.Embedding(self.config.label_size, self.hidden_size)
        self.linear1 = nn.Linear(self.hidden_size, self.config.label_hidden_size)
        self.linear2 = nn.Linear(self.config.label_hidden_size, self.config.num_labels)

    def forward(self, input_ids, word_begin_index, labels=None):
        hidden_states = self.base_model(input_ids)["last_hidden_state"]
        prediction_in_depth = []
        word_scores_in_depth = []
        for d in range(self.config.depth):
            for layer in self.iterative_layers:
                hidden_states = layer(hidden_states)[0]
            hidden_states = self.dropout(hidden_states)
            un_word_begin_index = word_begin_index.unsqueeze(2).repeat(
                1, 1, self.hidden_size
            )
            word_hidden_states = torch.gather(
                hidden_states, dim=1, index=un_word_begin_index
            )

            if d != 0:
                index_words = torch.argmax(word_scores, dim=-1)
                label_embeddings = self.label_embedding(index_words)
                word_hidden_states = word_hidden_states + label_embeddings

            word_hidden_states = self.linear1(word_hidden_states)
            word_scores = self.linear2(word_hidden_states)

            prediction = torch.argmax(word_scores, dim=-1)

            word_scores_in_depth.append(word_scores)
            prediction_in_depth.append(prediction)

            if not self.is_valid_extraction(prediction):
                break
        # word_scores = depth [batch,num_words]
        word_scores_in_depth = torch.stack(word_scores_in_depth, dim=1)

        scores = [
            self.calculate_score(word_scores, label)
            for word_scores, label in zip(word_scores_in_depth, labels)
        ]
        output = dict()
        output["scores"] = torch.stack(scores, dim=0)
        output["predictions"] = torch.stack(prediction_in_depth, dim=1)
        return output

    def calculate_score(self, word_scores, labels):
        max_log_probs, predictions = torch.log_softmax(word_scores, dim=-1).max(dim=-1)
        mask_labels = (labels[0, :] != -100).float()
        mask_predictions = (predictions != 0).float() * mask_labels
        log_prob = (max_log_probs * mask_predictions) / (
            mask_predictions.sum(dim=1) + 1
        ).unsqueeze(-1)
        scores = torch.exp(log_prob.sum(dim=1))
        return scores


class ModelForConjunction(nn.Module, Models):
    def __init__(self, config):
        nn.Module.__init__(self)
        Models.__init__(self, config)

        self.config.depth = 3
        self.model = BackboneModel(self.config)

    def forward(self, input_ids, word_begin_index, labels=None):
        return self.model(input_ids, word_begin_index, labels)
