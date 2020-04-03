# transformer-pytorch-fast-decode
Implements fast decoding using the built in `nn.Transformer` in PyTorch. Both an object-oriented and a functional version are available, the functional version will allow you to simply pass in an `nn.Transformer`. Testing + Timing code are also available.

Given `batch_size=32, d_model=512, n_layers=12, src_len=50, tgt_len=10`, results in a speedup of 2x on a Titan Xp GPU.
