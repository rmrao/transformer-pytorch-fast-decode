import torch
import torch.nn as nn
import torch.nn.functional as F
from timeit import default_timer as timer


class TransformerDecoderLayerWithFastDecode(nn.TransformerDecoderLayer):  # type: ignore

    def fast_decode(self, tgt, memory, cache,
                    memory_mask=None, memory_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, cache, cache)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        if hasattr(self, "activation"):
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        else:  # for backward compatibility
            tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class TransformerDecoderWithFastDecode(nn.TransformerDecoder):  # type: ignore

    def fast_decode(self, tgt, memory, src_mask=None,
                    memory_mask=None, src_key_padding_mask=None,
                    memory_key_padding_mask=None,
                    tgt_decode_fn=None, tgt_encode_fn=None, stop_fn=None,
                    max_decode_len=100):
        embeddings = make_empty_tensor(memory.device, memory.dtype)
        cache = [make_empty_tensor(memory.device, memory.dtype)
                 for _ in range(self.num_layers)]
        tgt_curr = tgt
        for seqlen in range(max_decode_len):
            if tgt_encode_fn is not None:
                tgt_curr = tgt_encode_fn(tgt_curr)
            for i, layer in enumerate(self.layers):
                cache[i] = torch.cat((cache[i], tgt_curr), 0)
                tgt_curr = layer.fast_decode(
                    tgt_curr, memory, cache[i],
                    memory_mask=memory_mask,
                    memory_key_padding_mask=memory_key_padding_mask)
            if self.norm:
                tgt_curr = self.norm(tgt_curr)
            embeddings = torch.cat((embeddings, tgt_curr), 0)
            if tgt_decode_fn is not None:
                tgt_curr = tgt_decode_fn(tgt_curr)
            tgt = torch.cat((tgt, tgt_curr), 0)

            if stop_fn is not None and stop_fn(tgt):
                break

        return tgt, embeddings


class TransformerWithFastDecode(nn.Transformer):  # type: ignore

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", custom_encoder=None, custom_decoder=None):
        super().__init__()

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout, activation)
            encoder_norm = nn.LayerNorm(d_model)
            self.encoder = nn.TransformerEncoder(
                encoder_layer, num_encoder_layers, encoder_norm)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayerWithFastDecode(
                d_model, nhead, dim_feedforward, dropout, activation)
            decoder_norm = nn.LayerNorm(d_model)
            self.decoder = TransformerDecoderWithFastDecode(
                decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def fast_decode(self, src, tgt, src_mask=None,
                    memory_mask=None, src_key_padding_mask=None,
                    memory_key_padding_mask=None,
                    tgt_decode_fn=None, tgt_encode_fn=None, stop_fn=None,
                    max_decode_len=100):

        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return self.decoder.fast_decode(
            tgt, memory, memory_mask=memory_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_decode_fn=tgt_decode_fn, tgt_encode_fn=tgt_encode_fn,
            stop_fn=stop_fn, max_decode_len=max_decode_len)


def make_empty_tensor(device: torch.device = torch.device('cpu'),
                      dtype: torch.dtype = torch.float32):
    tensor = torch.Tensor().to(device=device, dtype=dtype)
    return tensor


def decode(model, src, tgt_start, src_mask=None,
           memory_mask=None, src_key_padding_mask=None,
           memory_key_padding_mask=None,
           tgt_decode_fn=None, tgt_encode_fn=None, stop_fn=None,
           max_decode_len=100):
    tgt = tgt_start
    for i in range(max_decode_len):
        if tgt_encode_fn is not None:
            tgt_embed = tgt_encode_fn(tgt)
        else:
            tgt_embed = tgt

        tgt_mask = get_causal_mask(tgt_embed, 0)
        embeddings = model(
            src, tgt_embed, src_mask=src_mask, tgt_mask=tgt_mask,
            memory_mask=memory_mask, src_key_padding_mask=src_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask)
        if tgt_decode_fn is not None:
            tgt_curr = tgt_decode_fn(embeddings[i].unsqueeze(0))
            tgt = torch.cat((tgt, tgt_curr))
        else:
            tgt = torch.cat((tgt, embeddings[i].unsqueeze(0)))

        if stop_fn is not None and stop_fn(tgt):
            break
    return tgt, embeddings


def layer_fast_decode(layer, tgt, memory, cache, memory_mask=None,
                      memory_key_padding_mask=None):
    tgt2 = layer.self_attn(tgt, cache, cache)[0]
    tgt = tgt + layer.dropout1(tgt2)
    tgt = layer.norm1(tgt)
    tgt2 = layer.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                key_padding_mask=memory_key_padding_mask)[0]
    tgt = tgt + layer.dropout2(tgt2)
    tgt = layer.norm2(tgt)
    if hasattr(layer, "activation"):
        tgt2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(tgt))))
    else:  # for backward compatibility
        tgt2 = layer.linear2(layer.dropout(F.relu(layer.linear1(tgt))))
    tgt = tgt + layer.dropout3(tgt2)
    tgt = layer.norm3(tgt)
    return tgt


def fast_decode(model, src, tgt_start, src_mask=None,
                memory_mask=None, src_key_padding_mask=None,
                memory_key_padding_mask=None,
                tgt_decode_fn=None, tgt_encode_fn=None, stop_fn=None,
                max_decode_len=100):
    tgt = make_empty_tensor(src.device, tgt_start.dtype)
    embeddings = make_empty_tensor(src_embed.device, src_embed.dtype)
    memory = model.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
    cache = [make_empty_tensor(src.device, src.dtype)
             for _ in range(model.decoder.num_layers)]
    tgt_curr = tgt_start
    for _ in range(max_decode_len):
        if tgt_encode_fn is not None:
            tgt_curr = tgt_encode_fn(tgt_curr)
        for i, layer in enumerate(model.decoder.layers):
            cache[i] = torch.cat((cache[i], tgt_curr), 0)
            tgt_curr = layer_fast_decode(
                layer,
                tgt_curr, memory, cache[i],
                memory_mask=memory_mask,
                memory_key_padding_mask=memory_key_padding_mask)
        if model.decoder.norm:
            tgt_curr = model.decoder.norm(tgt_curr)
        embeddings = torch.cat((embeddings, tgt_curr), 0)
        if tgt_decode_fn is not None:
            tgt_curr = tgt_decode_fn(tgt_curr)
        tgt = torch.cat((tgt, tgt_curr), 0)

        if stop_fn is not None and stop_fn(tgt):
            break

    return tgt, embeddings


def fast_decode_class(
        model, src, tgt_start, src_mask=None,
        memory_mask=None, src_key_padding_mask=None,
        memory_key_padding_mask=None,
        tgt_decode_fn=None, tgt_encode_fn=None, stop_fn=None,
        max_decode_len=100):
    return model.fast_decode(
        src, tgt_start,
        src_mask=src_mask, src_key_padding_mask=src_key_padding_mask,
        memory_mask=memory_mask, memory_key_padding_mask=memory_key_padding_mask,
        tgt_decode_fn=tgt_decode_fn, tgt_encode_fn=tgt_encode_fn,
        stop_fn=stop_fn, max_decode_len=max_decode_len)


def get_causal_mask(sequence, axis=1):
    seqlen = sequence.size(axis)
    mask = torch.triu(torch.ones(
        seqlen, seqlen,
        dtype=sequence.dtype,
        device=sequence.device), 1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


def time(func):
    start = timer()
    func()
    end = timer()
    return end - start


# decoder_layer = TransformerDecoderLayerWithFastDecode(
    # D_MODEL, NHEAD, 4 * D_MODEL, DROPOUT, ACTIVATION)
# decoder_norm = nn.LayerNorm(D_MODEL)
# decoder = TransformerDecoderWithFastDecode(
    # decoder_layer, NUM_LAYERS, decoder_norm)  # type: ignore


BATCH_SIZE = 32
VOCAB_SIZE = 10
D_MODEL = 512
NHEAD = 8
NUM_LAYERS = 12
DROPOUT = 0
ACTIVATION = 'relu'

model = TransformerWithFastDecode(
    d_model=D_MODEL,
    # custom_decoder=decoder,
    num_encoder_layers=NUM_LAYERS,
    num_decoder_layers=NUM_LAYERS,
    dim_feedforward=4 * D_MODEL,
    dropout=DROPOUT,
    activation=ACTIVATION).cuda()

embedding = nn.Embedding(VOCAB_SIZE, D_MODEL).cuda()
src = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, 50), device='cuda')
tgt = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, 10), device='cuda')
src[0, 30:] = 0
src[1, 45:] = 0
tgt[0, 4:] = 0
src_key_padding_mask = (src == 0)
src = src.transpose(0, 1)
tgt = tgt.transpose(0, 1)
src_embed = embedding(src)
tgt_embed = embedding(tgt)
tgt_mask = get_causal_mask(tgt_embed, axis=0)


def forward_normal():
    return model(src_embed, tgt_embed, tgt_mask=tgt_mask,
                 src_key_padding_mask=src_key_padding_mask,
                 memory_key_padding_mask=src_key_padding_mask)


def forward_decode():
    tgt_iter = (tgt_seq.unsqueeze(0) for tgt_seq in tgt)

    def decode_fn(_):
        try:
            return next(tgt_iter)
        except StopIteration:
            return tgt[0].unsqueeze(0)

    return decode(model, src_embed, next(tgt_iter),
                  src_key_padding_mask=src_key_padding_mask,
                  memory_key_padding_mask=src_key_padding_mask,
                  tgt_decode_fn=decode_fn,
                  tgt_encode_fn=embedding,
                  max_decode_len=tgt.size(0))


def forward_fast_decode():
    tgt_iter = (tgt_seq.unsqueeze(0) for tgt_seq in tgt)

    def decode_fn(_):
        try:
            return next(tgt_iter)
        except StopIteration:
            return tgt[0].unsqueeze(0)

    return fast_decode_class(
        model, src_embed, next(tgt_iter),
        src_key_padding_mask=src_key_padding_mask,
        memory_key_padding_mask=src_key_padding_mask,
        tgt_decode_fn=decode_fn,
        tgt_encode_fn=embedding,
        max_decode_len=tgt.size(0))


def test_fast_decode():
    normal_out = forward_normal()
    _, decode_out = forward_fast_decode()

    assert torch.all(torch.isclose(normal_out, decode_out, atol=1e-5))


def time_fast_decode():
    time_normal = min(time(forward_normal) for _ in range(10))
    time_decode = min(time(forward_decode) for _ in range(10))
    time_fast_decode = min(time(forward_fast_decode) for _ in range(10))

    print("Time Normal:", time_normal)
    print("Time Decode:", time_decode)
    print("Time Fast:", time_fast_decode)


if __name__ == '__main__':
    test_fast_decode()
    time_fast_decode()
