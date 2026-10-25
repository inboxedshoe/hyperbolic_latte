bci_encoder = {"inception_channels": 10,
               "curvature": 1,
               "learnable": True,
               "windows": 1,
               "sub_rank": 5,
               "intermediate": 1024,
               "lora_lr": 1e-1,
               "decoder_rank": 64}


cha_encoder = {"inception_channels": 10,
               "curvature": 3,
               "learnable": True,
               "windows": 4,
               "sub_rank": 5,
               "intermediate": 1024,
               "lora_lr": 1e-1,
               "decoder_rank": 64}

bnci2015001_encoder = {**bci_encoder}

DEFAULT_MODEL_CONFIGS = {                         "bcicha": [cha_encoder],
                         "bcic2a": [bci_encoder],
                         "bcic2b": [bci_encoder],
                         "bnci2015001": [bnci2015001_encoder]}

# multi-kernel CNN
_tcformer_shared = {
    "F1": 32,
    "temp_kernel_lengths": (20, 32, 64),
    "d_group": 16,
    "D": 2,
    "pool_length_1": 8,
    "pool_length_2": 7,
    "dropout": 0.4,
    "use_group_attn": True,
}

# num_subjects + 1 adapter slots are allocated.
bci_tcformer_preprocessor    = {"n_channels": 22, "num_subjects": 9,  **_tcformer_shared}
cha_tcformer_preprocessor    = {"n_channels": 56, "num_subjects": 16, **_tcformer_shared}
bcic2b_tcformer_preprocessor = {"n_channels": 3,  "num_subjects": 9,  **_tcformer_shared}
bnci2015001_tcformer_preprocessor = {"n_channels": 13, "num_subjects": 12, **_tcformer_shared}

# d_model = d_group * len(temp_kernel_lengths) = 16 * 3 = 48
DEFAULT_TCFORMER_PREPROCESSOR_CONFIGS = {
    "bcic2a": [bci_tcformer_preprocessor,    48],
    "bcicha": [cha_tcformer_preprocessor,    48],
    "bcic2b": [bcic2b_tcformer_preprocessor, 48],
    "bnci2015001": [bnci2015001_tcformer_preprocessor, 48],
}
