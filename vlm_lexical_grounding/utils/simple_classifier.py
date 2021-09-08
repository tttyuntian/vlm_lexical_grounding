import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"


class MultipleChoiceClassifier(nn.Module):
    def __init__(
        self, 
        embedder, 
        hidden_dropout=.2, 
        num_choices=2
    ):
        super(MultipleChoiceClassifier, self).__init__()
                
        self.embedder = embedder
        self.dropout = nn.Dropout(hidden_dropout)
        self.classifier = nn.Linear(embedder.config.hidden_size, 1)

    def forward(
        self, 
        input_ids, 
        attention_mask, 
        token_type_ids, 
        labels=None, 
        is_challenge=False
    ):
        num_choices = input_ids.shape[1]
        
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        
        outputs = self.embedder(
            input_ids=input_ids,
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids
        )
        
        pooled_output = outputs[1]
        logits = self.classifier(self.dropout(pooled_output))
        logits = logits.view(-1, num_choices)
        
        loss = None
        if labels is not None:
            if not is_challenge:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)
            else:
                labels_onehot = torch.zeros(labels.size(0), num_choices).to(device)
                labels_onehot.scatter_(1, labels.view(-1, 1), 1)
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels_onehot)
        
        return logits, loss


class MultipleChoiceProbingHead(nn.Module):
    def __init__(
            self, 
            input_size, 
            hidden_size, 
            cls_type="mlp", 
            num_choices=2, 
            num_heads=12, 
            intermediate_size=3072, 
            hidden_dropout=0.1, 
            activation="gelu"
    ):
        super(MultipleChoiceProbingHead, self).__init__()
        if cls_type == "linear":
            classifier = nn.Linear(input_size, 1)
        elif cls_type == "mlp":
            classifier = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.Tanh(),
                nn.LayerNorm(hidden_size),
                nn.Dropout(hidden_dropout),
                nn.Linear(hidden_size, 1)
            )
        elif cls_type == "transformer":
            self.encoder_layer = nn.TransformerEncoderLayer(
                input_size, num_heads, intermediate_size, hidden_dropout, activation
            )
            classifier = nn.Linear(input_size, 1)
        else:
            raise ValueError("Classifier type %s not found" % cls_type)
        self.cls_type = cls_type
        self.classifier = classifier
    
    def forward(self, inputs, labels=None, is_challenge=False):
        num_choices = inputs.shape[1]
        
        outputs = inputs.view(-1, inputs.size(-1))
        if self.cls_type == "transformer":
            outputs = self.encoder_layer(inputs)
        logits = self.classifier(outputs)
        logits = logits.view(-1, num_choices)
        
        loss = None
        if labels is not None:
            if not is_challenge:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)
            else:
                labels_onehot = torch.zeros(inputs.size(0), inputs.size(1)).to(device)
                labels_onehot.scatter_(1, labels.view(-1, 1), 1)
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels_onehot)
        
        return logits, loss


class AdjectiveProbingHead(nn.Module):
    def __init__(
        self, 
        input_size, 
        output_size, 
        hidden_size, 
        cls_type="mlp", 
        num_choices=None, 
        num_heads=12, 
        intermediate_size=3072, 
        hidden_dropout=0.1, 
        activation="gelu"
    ):
        super(AdjectiveProbingHead, self).__init__()
        if cls_type == "linear":
            classifier = nn.Linear(input_size, output_size)
        elif cls_type == "mlp":
            classifier = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.Tanh(),
                nn.LayerNorm(hidden_size),
                nn.Dropout(hidden_dropout),
                nn.Linear(hidden_size, output_size)
            )
        elif cls_type == "transformer":
            self.encoder_layer = nn.TransformerEncoderLayer(
                input_size, num_heads, intermediate_size, hidden_dropout, activation
            )
            classifier = nn.Linear(input_size, 1)
        else:
            raise ValueError("Classifier type %s not found" % cls_type)
        self.cls_type = cls_type
        self.classifier = classifier
        self.output_size = output_size
    
    def forward(self, inputs, labels=None):        
        outputs = inputs.view(-1, inputs.size(-1))
        if self.cls_type == "transformer":
            outputs = self.encoder_layer(inputs)
        logits = self.classifier(outputs)
        logits = logits.view(-1, self.output_size)
        
        loss = None
        if labels is not None:        
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            
        return logits, loss
