import torch
import torch.nn as nn

from latte.geometry.layers.linear import (LORENTZ_FC_TYPES,
                                          LorentzFullyConnectedLora,
                                          LorentzFullyConnectedBoostLora)
from latte.geometry.layers.misc import GeodesicLoRAAdapter, HypLoRAAdapter
from latte.geometry.manifold import CustomLorentz
from latte.models.processor import TCFormerPreprocessorLora
from latte.models import encoder as encoders
from latte.models import decoders
from latte.models.utils_h import ToHyperbolic
from latte.training.helpers import slice_time_series
from latte.models.configs import DEFAULT_MODEL_CONFIGS, DEFAULT_TCFORMER_PREPROCESSOR_CONFIGS


class preDecoder(nn.Module):
    """Projection followed by the subject adapter: fc(x) then adapter(x, subject)."""

    def __init__(self, l1, l2):
        super().__init__()
        self.l1 = l1
        self.l2 = l2

    def forward(self, x, sub):
        x = self.l1(x)
        x = self.l2(x, sub)
        return x


class TCFormerDeviationHeadLora(nn.Module):
    """LAtte.

    lora_type (the subject adapter after the projection)
        "boost"     LB-Adapter, composed Lorentz boosts
        "linear"    Euclidean LoRA on the space coordinates
        "geodesic"  low-rank tangent update transported and exp-mapped
        "hyplora"   direct Lorentz low-rank transformation
    decoder_proj
        "learned"   the shared projection inside the adapter layer is an
                    ordinary trainable linear map on the space coordinates.
        "frozen"    unit-norm random rows, frozen: a random projection.
        "none"      no dimensionality reduction (identity, frozen).
    proto_init      "lift" or "wrapped_normal"; see decoders.py.
    proto_std       std of the prototype draw (None: the default of each init).
    head_center     "batch": the head embeddings of each batch are centred on their
                    mean (tangent space at the origin) before the prototypes, in
                    training and at test alike.
                    "none": no centring. See center().
    reference, block_norm, pool_type
                    see encoder.py and blocks.py.
    """

    def __init__(self,
                 manifold=None,
                 n_classes=3,
                 dataset="bcic2a",
                 learn_k=False,
                 features=256 / 4,
                 conv_type="base",
                 batch_type="gyro",
                 pool_type="dirty",
                 pool_tau=1.0,
                 fc_type="base",
                 dropout=0,
                 windows=None,
                 lora_lr=None,
                 lora_type="boost",
                 bias=False,
                 proc_lora_lr=1.0,
                 preprocessor_adaptive=False,
                 encoder_adaptive=False,
                 bn_kwargs=None,
                 dw_lora=False,
                 reference="block",
                 block_norm="encoder",
                 decoder_proj="learned",
                 proto_init="lift",
                 proto_std=None,
                 head_center="none"):
        super().__init__()
        if lora_type not in ("boost", "linear", "geodesic", "hyplora"):
            raise ValueError(f"unknown lora_type {lora_type!r}")
        if decoder_proj not in ("learned", "frozen", "none"):
            raise ValueError(f"decoder_proj must be learned/frozen/none, got {decoder_proj!r}")
        if head_center not in ("none", "batch"):
            raise ValueError(f"head_center must be none/batch, got {head_center!r}")

        model_configs = DEFAULT_MODEL_CONFIGS[dataset]
        inception_channels = model_configs[0]['inception_channels']
        rank = model_configs[0]['sub_rank']
        self.intermediate = model_configs[0]['intermediate']
        self.decoder_rank = model_configs[0]['decoder_rank']
        self.lora_lr = lora_lr
        self.lora_type = lora_type
        self.decoder_proj = decoder_proj
        self.proto_init = proto_init
        self.proto_std = proto_std
        self.head_center = head_center
        self.bias = bias
        self.fc_type = fc_type
        self.windows = model_configs[0]['windows'] if windows is None else windows
        self.n_classes = n_classes

        features = int(features)
        self.manifold = CustomLorentz(k=1, learnable=learn_k) if manifold is None else manifold

        preprocessor_cfg, processor_output = DEFAULT_TCFORMER_PREPROCESSOR_CONFIGS[dataset]
        self.num_subs = preprocessor_cfg['num_subjects']
        self.processor = TCFormerPreprocessorLora(
            rank=rank, proc_lora_lr=proc_lora_lr,
            adaptive_bn=preprocessor_adaptive, dw_lora=dw_lora, **preprocessor_cfg,
        )

        self.encoder = encoders.BaselineDeviationEncoder(self.manifold,
                                                         in_channel=processor_output,
                                                         features=features,
                                                         kernel_sizes=(9, 19, 39),
                                                         inception_channels=inception_channels,
                                                         conv_type=conv_type,
                                                         batch_type=batch_type,
                                                         pool_type=pool_type,
                                                         pool_tau=pool_tau,
                                                         dropout=dropout,
                                                         fc_type=fc_type,
                                                         bias=bias,
                                                         num_subjects=self.num_subs,
                                                         adaptive_bn=encoder_adaptive,
                                                         bn_kwargs=bn_kwargs,
                                                         reference=reference,
                                                         block_norm=block_norm)

        self.to_hyperbolic = ToHyperbolic(self.manifold, norm=False, tangent_based=False)
        self.first_run = True

    # ------------------------------------------------------------------ build
    def get_decoder(self, x, x_sub=None):
        """Build the head. Its input width depends on the data, so this runs on
        the first batch."""
        self.first_run = False

        flattened = self._embed_to_head(x, x_sub)
        flattened = self.manifold.rescale_to_max(flattened)

        in_dim = flattened.shape[-1]
        out_dim = in_dim if self.decoder_proj == "none" else self.intermediate + 1

        self.fc = nn.Sequential()
        match self.lora_type:
            case "boost":
                self.lora = LorentzFullyConnectedBoostLora(
                    self.manifold, in_dim, out_dim,
                    num_subjects=self.num_subs, rank=self.decoder_rank, alpha=self.lora_lr)
            case "linear":
                self.lora = LorentzFullyConnectedLora(
                    self.manifold, in_dim, out_dim,
                    num_subjects=self.num_subs, rank=self.decoder_rank, lora_alpha=self.lora_lr)
            case "geodesic" | "hyplora":
                self.fc = LORENTZ_FC_TYPES[self.fc_type](manifold=self.manifold,
                                                         in_features=in_dim,
                                                         out_features=out_dim,
                                                         bias=self.bias)
                adapter = GeodesicLoRAAdapter if self.lora_type == "geodesic" else HypLoRAAdapter
                self.lora = adapter(self.manifold, out_dim, rank=self.decoder_rank,
                                    num_subjects=self.num_subs, lora_scaling=self.lora_lr)

        if self.decoder_proj != "learned":
            self._fix_projection()

        self.pre_decoder = preDecoder(self.fc, self.lora).to(device=x.device)
        self.decoder = decoders.LorentzPrototypeDecoder(
            self.manifold, out_dim - 1, self.n_classes, init=self.proto_init, std=self.proto_std
        ).to(device=x.device)

    def _projection_params(self):
        """The shared (subject-independent) projection of the pre-decoder."""
        if self.lora_type in ("boost", "linear"):
            return list(self.lora.weight.parameters())
        return list(self.fc.parameters())

    @torch.no_grad()
    def _fix_projection(self):
        if self.lora_type in ("boost", "linear"):
            w = self.lora.weight
            if self.decoder_proj == "none":
                w.weight.copy_(torch.eye(*w.weight.shape))
            else:
                # W ~ U(-sigma, sigma), each row normalised to unit length
                nn.init.uniform_(w.weight, -self.lora.init_std, self.lora.init_std)
                w.weight.div_(w.weight.norm(dim=1, keepdim=True).clamp_min(1e-8))
            if w.bias is not None:
                w.bias.zero_()
        elif self.decoder_proj == "none":
            raise ValueError("decoder_proj='none' is only implemented for lora_type boost/linear")
        for p in self._projection_params():
            p.requires_grad = False

    def set_trainable(self, learn_decoder=True, learn_predecoder=True, learn_lora=True):
        """Apply the --learn_* flags. Call after get_decoder().

        learn_lora=False freezes every subject adapter (processor LoRA and the
        head adapter). They are initialised to the identity, so the model then
        runs without subject adaptation.
        """
        # The manifold (curvature k) is shared by every layer; its trainability
        # is set by --learnable_k alone, so skip it here.
        for n, p in self.decoder.named_parameters():
            if "manifold" not in n:
                p.requires_grad = learn_decoder
        for n, p in self.pre_decoder.named_parameters():
            if "manifold" not in n:
                p.requires_grad = learn_predecoder
        if not learn_lora:
            for name, p in self.named_parameters():
                if name.startswith("processor.") and (name.endswith(".Q") or name.endswith(".R")):
                    p.requires_grad = False
            shared = {id(p) for p in self._projection_params()}
            for n, p in self.lora.named_parameters():
                if id(p) not in shared and "manifold" not in n:
                    p.requires_grad = False
        if self.decoder_proj != "learned":
            for p in self._projection_params():
                p.requires_grad = False

    # ---------------------------------------------------------------- forward
    def _sliced_subs(self, x_sub):
        """Repeat subject IDs to align with the post-slice batch (B -> B*windows)."""
        if x_sub is None:
            return None
        return x_sub.view(-1).repeat_interleave(self.windows)

    def _embed_to_head(self, x, x_sub):
        processed = self.processor(x, x_sub)
        processed = self.to_hyperbolic(processed)
        processed = slice_time_series(processed, self.windows)
        embeddings = self.encoder(processed, self._sliced_subs(x_sub))
        embeddings = embeddings.reshape(x.shape[0], self.windows, -1,
                                        embeddings.shape[-1]).permute(0, 2, 1, 3)
        embeddings = self.manifold.direct_concat(embeddings)
        return self.manifold.centroid(embeddings)

    def embed(self, x, x_sub=None):
        """Point on the manifold that the prototype decoder classifies."""
        z = self.pre_decoder(self._embed_to_head(x, x_sub), x_sub)
        return self.manifold.rescale_to_max(z)

    def center(self, z):
        """head_center="batch": centre the head embeddings of the current batch on their mean,
        taken in the tangent space at the origin, so the prototypes see a batch whose mean sits
        at the origin."""
        if getattr(self, "head_center", "none") != "batch":
            return z
        v = self.manifold.logmap0(z)[..., 1:]
        v = v - v.mean(0, keepdim=True)
        return self.manifold.expmap0(torch.nn.functional.pad(v, (1, 0)))

    def forward(self, x, x_sub=None):
        if self.first_run:
            with torch.no_grad():
                self.get_decoder(x, x_sub)
        return self.decoder(self.center(self.embed(x, x_sub)))
