# Modified based on the DEQ and MDEQ repo.
# Identifiers will be add once the code is made public.

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.functional import normalize


class VariationalHidDropout2d(nn.Module):
    def __init__(self, dropout=0.0, spatial=True):
        """
        Hidden-to-hidden (VD-based) dropout that applies the same mask at every pixel and every layer
        :param dropout: The dropout rate (0 means no dropout is applied)
        """
        super(VariationalHidDropout2d, self).__init__()
        self.dropout = dropout
        self.mask = None
        self.spatial = spatial

    def reset_mask(self, x):
        dropout = self.dropout
        spatial = self.spatial

        # x has dimension (N, C, H, W)
        if spatial:
            m = torch.zeros_like(x[:,:,:1,:1]).bernoulli_(1 - dropout)
        else:
            m = torch.zeros_like(x).bernoulli_(1 - dropout)
        mask = m.requires_grad_(False) / (1 - dropout)
        self.mask = mask
        return mask

    def forward(self, x):
        if not self.training or self.dropout == 0:
            return x
        assert self.mask is not None, "You need to reset mask before using VariationalHidDropout"
        return self.mask.expand_as(x) * x 

    
class VariationalHidDropout2dList(nn.Module):
    def __init__(self, dropout=0.0, spatial=True):
        """
        Hidden-to-hidden (VD-based) dropout that applies the same mask at every pixel and every layer
        :param dropout: The dropout rate (0 means no dropout is applied)
        """
        super(VariationalHidDropout2dList, self).__init__()
        self.dropout = dropout
        self.mask = None
        self.spatial = spatial

    def reset_mask(self, xs):
        dropout = self.dropout
        spatial = self.spatial
        
        self.mask = []
        for x in xs:
            # x has dimension (N, C, H, W)
            if spatial:
                m = torch.zeros_like(x[:,:,:1,:1]).bernoulli_(1 - dropout)
            else:
                m = torch.zeros_like(x).bernoulli_(1 - dropout)
            mask = m.requires_grad_(False) / (1 - dropout)
            self.mask.append(mask)
        return self.mask

    def forward(self, xs):
        if not self.training or self.dropout == 0:
            return xs
        assert self.mask is not None and len(self.mask) > 0, "You need to reset mask before using VariationalHidDropoutList"
        return [self.mask[i].expand_as(x) * x for i, x in enumerate(xs)]
    
    
def _norm(p, dim):
    """Computes the norm over all dimensions except dim"""
    if dim is None:
        return p.norm()
    elif dim == 0:
        output_size = (p.size(0),) + (1,) * (p.dim() - 1)
        return p.contiguous().view(p.size(0), -1).norm(dim=1).view(*output_size)
    elif dim == p.dim() - 1:
        output_size = (1,) * (p.dim() - 1) + (p.size(-1),)
        return p.contiguous().view(-1, p.size(-1)).norm(dim=0).view(*output_size)
    else:
        return _norm(p.transpose(0, dim), 0).transpose(0, dim)


class WeightNorm(object):
    def __init__(self, names, dim):
        """
        Weight normalization module
        :param names: The list of weight names to apply weightnorm on
        :param dim: The dimension of the weights to be normalized
        """
        self.names = names
        self.dim = dim

    def compute_weight(self, module, name):
        g = getattr(module, name + '_g')
        v = getattr(module, name + '_v')
        return v * (g / _norm(v, self.dim))

    @staticmethod
    def apply(module, names, dim):
        fn = WeightNorm(names, dim)

        for name in names:
            weight = getattr(module, name)

            # remove w from parameter list
            del module._parameters[name]

            # add g and v as new parameters and express w as g/||v|| * v
            module.register_parameter(name + '_g', Parameter(_norm(weight, dim).data))
            module.register_parameter(name + '_v', Parameter(weight.data))
            setattr(module, name, fn.compute_weight(module, name))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)
        return fn

    def remove(self, module):
        for name in self.names:
            weight = self.compute_weight(module, name)
            delattr(module, name)
            del module._parameters[name + '_g']
            del module._parameters[name + '_v']
            module.register_parameter(name, Parameter(weight.data))

    def reset(self, module):
        for name in self.names:
            setattr(module, name, self.compute_weight(module, name))

    def __call__(self, module, inputs):
        # Typically, every time the module is called we need to recompute the weight. However,
        # in the case of TrellisNet, the same weight is shared across layers, and we can save
        # a lot of intermediate memory by just recomputing once (at the beginning of first call).
        pass


def weight_norm(module, names, dim=0):
    fn = WeightNorm.apply(module, names, dim)
    return module, fn


def compute_spectral_norm(weight_mat, iterations=10):
    h, w = weight_mat.size()
    u = normalize(torch.randn(h), dim=0, eps=1e-12)
    v = normalize(torch.randn(w), dim=0, eps=1e-12)
    with torch.no_grad():
        for _ in range(iterations):
            v = normalize(torch.mv(weight_mat.t(), u), dim=0, eps=1e-12, out=v)
            u = normalize(torch.mv(weight_mat, v), dim=0, eps=1e-12, out=u)
    sigma = torch.dot(u, torch.mv(weight_mat, v))
    return sigma


class WeightSpectralNorm(object):

    _version: int = 1
    
    def __init__(self, names, n_power_iterations=1, dim=0, eps=1e-12, norm_range=1):
        """
        Weight spectral normalization module
        :param names: The list of weight names to apply weightnorm on
        """
        self.names = names
        self.n_power_iterations = n_power_iterations
        self.dim = dim
        self.eps = eps
        self.norm_range = norm_range

    def reshape_weight_to_matrix(self, weight):
        weight_mat = weight
        if self.dim != 0:
            weight_mat = weight_mat.permute(self.dim, *[d for d in range(weight_mat.dim()) if d != self.dim])
        height = weight_mat.size(0)
        return weight_mat.reshape(height, -1)

    def compute_weight(self, module, name):
        g = getattr(module, name + '_g')
        w = getattr(module, name + '_w')
        u = getattr(module, name + '_u')
        v = getattr(module, name + '_v')
        w_mat = self.reshape_weight_to_matrix(w)

        # restrict the spectral norm
        g = torch.clamp(g, -self.norm_range, self.norm_range)

        # spectral normalization for w
        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                v = normalize(torch.mv(w_mat.t(), u), dim=0, eps=self.eps, out=v)
                u = normalize(torch.mv(w_mat, v), dim=0, eps=self.eps, out=u)
            if self.n_power_iterations > 0:
                u = u.clone(memory_format=torch.contiguous_format)
                v = v.clone(memory_format=torch.contiguous_format)
        sigma = torch.dot(u, torch.mv(w_mat, v))
        w = w / sigma

        return g * w

    @staticmethod
    def apply(module, names, n_power_iterations=1, dim=0, eps=1e-12, norm_range=1.):
        fn = WeightSpectralNorm(names, n_power_iterations, dim, eps, norm_range)

        for name in names:
            weight = getattr(module, name)

            with torch.no_grad():
                weight_mat = fn.reshape_weight_to_matrix(weight)
                h, w = weight_mat.size()
                u = normalize(weight.new_empty(h).normal_(0, 1), dim=0, eps=fn.eps)
                v = normalize(weight.new_empty(w).normal_(0, 1), dim=0, eps=fn.eps)
                # first compute the spectral norm of weight
                sigma = compute_spectral_norm(weight_mat, iterations=10)

            # remove w from parameter list
            del module._parameters[name]

            # add g and w as new parameters and express w as g / ||w|| * w
            module.register_parameter(name + '_g', Parameter(sigma.data))
            module.register_parameter(name + '_w', Parameter(weight.data))
            module.register_buffer(name + '_u', u)
            module.register_buffer(name + '_v', v)
            setattr(module, name, fn.compute_weight(module, name))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)
        module._register_state_dict_hook(SpectralNormStateDictHook(fn))
        module._register_load_state_dict_pre_hook(SpectralNormLoadStateDictPreHook(fn))
        return fn

    def remove(self, module):
        for name in self.names:
            with torch.no_grad():
                weight = self.compute_weight(module, name)
            delattr(module, name)
            del module._parameters[name + '_g']
            del module._parameters[name + '_w']
            del module._parameters[name + '_u']
            del module._parameters[name + '_v']
            module.register_parameter(name, Parameter(weight.data))

    def reset(self, module):
        for name in self.names:
            setattr(module, name, self.compute_weight(module, name))

    def __call__(self, module, inputs):
        pass

    def _solve_v_and_rescale(self, weight_mat, u, target_sigma):
        v = torch.chain_matmul(weight_mat.t().mm(weight_mat).pinverse(), weight_mat.t(), u.unsqueeze(1)).squeeze(1)
        return v.mul_(target_sigma / torch.dot(u, torch.mv(weight_mat, v)))


class SpectralNormLoadStateDictPreHook:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        fn = self.fn
        for name in fn.names:
            version = local_metadata.get('weight_spectral_norm', {}).get(name + '.version', None)
            if version is None or version < 1:
                weight_key = prefix + name
                if version is None and all(weight_key + s in state_dict for s in ('_g', '_w', '_u', '_v')) and weight_key not in state_dict:
                    return
                has_missing_keys = False
                for suffix in ('_w', '_g', '', '_u'):
                    key = weight_key + suffix
                    if key not in state_dict:
                        has_missing_keys = True
                        if strict:
                            missing_keys.append(key)
                if has_missing_keys:
                    return
                with torch.no_grad():
                    weight_orig = state_dict[weight_key + '_w']
                    weight = state_dict.pop(weight_key)
                    sigma = (weight_orig / weight).mean()
                    weight_mat = fn.reshape_weight_to_matrix(weight_orig)
                    u = state_dict[weight_key + '_u']
                    v = fn._solve_v_and_rescale(weight_mat, u, sigma)
                    state_dict[weight_key + '_v'] = v


class SpectralNormStateDictHook:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, module, state_dict, prefix, local_metadata):
        if 'weight_spectral_norm' not in local_metadata:
            local_metadata['weight_spectral_norm'] = {}
        for name in self.fn.names:
            key = name + '.version'
            if key in local_metadata['weight_spectral_norm']:
                raise RuntimeError("Unexpected key in metadata['weight_spectral_norm']: {}".format(key))
            local_metadata['weight_spectral_norm'][key] = self.fn._version


def weight_spectral_norm(module, names, n_power_iterations=1, dim=0, eps=1e-12, norm_range=1.):
    fn = WeightSpectralNorm.apply(module, names, n_power_iterations, dim, eps, norm_range)
    return module, fn
