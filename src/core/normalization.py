import torch


class Normalization:
    def __init__(self, norm_type: str, *args, **kwargs):
        self.norm_type = norm_type

    @torch.inference_mode()
    def __call__(self, vec):
        if self.norm_type == "sign":
            return torch.sign(vec)
        elif self.norm_type == "l2":
            return vec / vec.norm(p=2, dim=(1, 2, 3), keepdim=True)
        elif self.norm_type == "l1":
            return vec / vec.norm(p=1, dim=(1, 2, 3), keepdim=True)
        elif self.norm_type == "linf":
            return vec / vec.norm(p=torch.inf, dim=(1, 2, 3), keepdim=True)
        elif not self.norm_type:
            return vec
        else:
            raise NotImplementedError(f"{self.norm_type} is not implemented.")
