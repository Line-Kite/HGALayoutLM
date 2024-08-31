import math
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    TokenClassifierOutput,
)

from model.basemodel import LayoutLMv3Model
from model.configuration_graphlayoutlm import GraphLayoutLMConfig


class GraphLayoutLMPreTrainedModel(PreTrainedModel):
    config_class = GraphLayoutLMConfig
    base_model_prefix = "graphlayoutlm"

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class GraphAttentionLayer(nn.Module):
    def __init__(self,config):
        super(GraphAttentionLayer, self).__init__()
        self.num_attention_heads = int(config.num_attention_heads/2)
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.final = nn.Linear(config.hidden_size, self.all_head_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self,
        seq_inputs,
        graph_mask,
    ):
        mixed_query_layer = self.query(seq_inputs)

        key_layer = self.transpose_for_scores(self.key(seq_inputs))
        value_layer = self.transpose_for_scores(self.value(seq_inputs))
        query_layer = self.transpose_for_scores(mixed_query_layer)


        attention_scores = torch.matmul(query_layer , key_layer.transpose(-1, -2))/ math.sqrt(self.attention_head_size)
        
        mask = -9e15 * torch.ones(graph_mask.shape).to(graph_mask.device)

        mask = mask.masked_fill(graph_mask, value=torch.tensor(0).to(graph_mask.device))

        attention_scores = attention_scores+mask.unsqueeze(1).repeat(1,self.num_attention_heads,1,1)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = self.final(context_layer)

        return outputs
    
class SubLayerConnection(nn.Module):
    def __init__(self,config):
        super(SubLayerConnection,self).__init__()
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-05)
        self.dropout=nn.Dropout(p=config.hidden_dropout_prob)
        self.size=config.hidden_size

    def forward(self,x,graph_mask,sublayer):
        return x+self.dropout(sublayer(self.norm(x),graph_mask))
    

class GraphLayoutLM(GraphLayoutLMPreTrainedModel):
    def __init__(self,config):
        super().__init__(config)
        self.model_base=LayoutLMv3Model(config)
        self.graph_attention_layer=GraphAttentionLayer(config)
        self.sublayer=SubLayerConnection(config)
        self.init_weights()


    def forward(
        self,
        input_ids=None,
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        images=None,
        valid_span=None,
        graph_mask=None,
    ):
        outputs = self.model_base(
            input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            images=images,
            valid_span=valid_span,
        )
        sequence_output=self.sublayer(outputs[0],graph_mask,self.graph_attention_layer)

        if not return_dict:
            return (sequence_output, outputs[1])

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=outputs.pooler_output,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )
        
class HGAHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_labels=config.num_labels
        self.inner_dim=64
        self.dense = nn.Linear(config.hidden_size, self.num_labels * self.inner_dim * 2)
        self.RoPE=True
        # classifier_dropout = (
        #     config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        # )
        # self.dropout = nn.Dropout(classifier_dropout)
        # self.dense0 = nn.Linear(config.hidden_size, config.hidden_size)
        # self.d=0.05
    
    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim, device):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(device)
        return embeddings
    
    def create_box_ids(self, boxes,seq_len):
        box_ids=torch.zeros((boxes.size()[0],seq_len),device=boxes.device)
        for b,b_box in enumerate(boxes):
            i=0
            count=0
            for j,box in enumerate(b_box):
                if box.tolist()!=b_box[i].tolist():
                    box_ids[b,i:j]=count
                    count+=1
                    i=j
        return box_ids
    
    def box_position_embedding(self, batch_size, seq_len, output_dim, boxes):
        position_ids = self.create_box_ids(boxes,seq_len).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float).to(position_ids.device)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(position_ids.device)
        return embeddings

    def forward(self, x, attention_mask,boxes):
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        batch_size = x.size()[0]
        seq_len = x.size()[1]
        
        # x = self.dropout(x)

        outputs = self.dense(x)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        # outputs:(batch_size, seq_len, ent_type_size, inner_dim*2)
        outputs = torch.stack(outputs, dim=-2)
        # qw,kw:(batch_size, seq_len, ent_type_size, inner_dim)
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]

        if self.RoPE:
            pos_emb=self.box_position_embedding(batch_size, seq_len, self.inner_dim,boxes)
            # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
    
        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)

        # padding mask
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.num_labels, seq_len, seq_len)
        # pad_mask = pad_mask_v&pad_mask_h
        logits = logits * pad_mask - (1 - pad_mask) * 1e12

        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), -1)
        
        logits = logits - (mask) * 1e12

        return logits / self.inner_dim ** 0.5


class HGALayoutLMForTokenClassification(GraphLayoutLMPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.graphlayoutlm = GraphLayoutLM(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.head = HGAHead(config)

        self.init_weights()
        
    def focus_loss(self, input, dim):
        p=torch.sum(torch.exp(input),dim=dim)
        return torch.log(p)
        
    def banlanced_hyperedge_loss(self, y_true, y_pred, b):
        y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
        y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
        y_pred_pos = (y_pred - (1 - y_true) * 1e12)  # mask the pred outputs of neg classes
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = self.focus_loss(y_pred_neg, -1)*(1-b)
        pos_loss = self.focus_loss(y_pred_pos, -1)*(1+b)

        return (neg_loss + pos_loss).mean()


    def forward(
        self,
        input_ids=None,
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        valid_span=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        images=None,
        graph_mask=None,
        matrix_labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.graphlayoutlm(
            input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            images=images,
            valid_span=valid_span,
            graph_mask=graph_mask,
        )

        sequence_output = outputs[0]

        # sequence_output = self.dropout(sequence_output)
        logits = self.head(sequence_output, attention_mask, bbox)

        def loss_fun(y_pred, y_true):
            batch_size, ent_type_size = y_pred.shape[:2]
            y_true = y_true.reshape(batch_size * ent_type_size, -1)
            y_pred = y_pred.reshape(batch_size * ent_type_size, -1)
            loss = self.banlanced_hyperedge_loss(y_true, y_pred, 0.4 if self.num_labels>10 else 0)
            return loss
        
        loss = None
        if matrix_labels is not None:
            hyperedges = torch.zeros(matrix_labels.shape).to(matrix_labels.device)
            hyperedges = hyperedges.masked_fill(matrix_labels, value=torch.tensor(1).to(matrix_labels.device))
            loss = loss_fun(logits, hyperedges)
        
        batch_size=sequence_output.size()[0]
        seq_len=sequence_output.size()[1]
        # logits=logits-1e12*create_mask(bbox,seq_len,self.num_labels)
        final_logits=torch.zeros((batch_size,seq_len,2*self.num_labels+1),device=logits.device)
        final_logits[:,:,-1]=0.8
        label_matrix=torch.where(torch.sigmoid(logits)>0.5,logits,torch.zeros_like(logits))
        label_matrix=torch.max(label_matrix,dim=1)
        pro_matrix=label_matrix.values
        label_matrix=label_matrix.indices
        zeros_line=torch.zeros((seq_len),device=logits.device)
        d_p=0.2/(seq_len+1)
        # label_record=-torch.ones((batch_size,seq_len),device=logits.device)
        for i,(matrix_i,labels_i) in enumerate(zip(pro_matrix,label_matrix)):
            for j,(line,labels_line) in enumerate(zip(matrix_i,labels_i)):
                if line.equal(zeros_line):
                    continue
                else:
                    end=torch.max(line,dim=0).indices.item()
                    # label_record[i,j:end]=labels_line[end].item()
                    bio_label=labels_line[end].item()*2
                    final_logits[i,j,bio_label]=0.8+(j+1)*d_p
                    final_logits[i,j+1:end+1,bio_label+1]=0.8+(j+1)*d_p
        logits = final_logits
        # print(labels)
        # exit(0)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        
