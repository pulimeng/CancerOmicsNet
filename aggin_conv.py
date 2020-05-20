import torch
from torch.nn import Parameter
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, softmax


class AGGINConv(MessagePassing):
    r"""Graph attentional propagation layer from the
    `"Attention-based Graph Neural Network for Semi-Supervised Learning"
    <https://arxiv.org/abs/1803.03735>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{P} \mathbf{X},

    where the propagation matrix :math:`\mathbf{P}` is computed as

    .. math::
        P_{i,j} = \frac{\exp( \beta \cdot \cos(\mathbf{x}_i, \mathbf{x}_j))}
        {\sum_{k \in \mathcal{N}(i)\cup \{ i \}} \exp( \beta \cdot
        \cos(\mathbf{x}_i, \mathbf{x}_k))}

    with trainable parameter :math:`\beta`.

    Args:
        requires_grad (bool, optional): If set to :obj:`False`, :math:`\beta`
            will not be trainable. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, nn, eps=0, train_eps=False, requires_grad=True, dropout=0, **kwargs):
        super(AGGINConv, self).__init__(aggr='add', **kwargs)
        
        self.nn = nn
        self.initial_eps = eps
        self.requires_grad = requires_grad
        self.dropout = dropout
        
        if train_eps:
            self.eps = Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

        if requires_grad:
            self.beta = Parameter(torch.Tensor(1))
        else:
            self.register_buffer('beta', torch.ones(1))
            
        self.reset_parameters()

    def reset_parameters(self):
        for net in self.nn:
            if isinstance(net, Linear):
                net.reset_parameters()
        self.eps.data.fill_(self.initial_eps)
        if self.requires_grad:
            self.beta.data.fill_(1)

    def forward(self, x, edge_index):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_index, _ = remove_self_loops(edge_index)

        x_norm = F.normalize(x, p=2, dim=-1)
        
        out = self.nn((1 + self.eps) * x + self.propagate(edge_index, x=x, x_norm=x_norm, num_nodes=x.size(0)))
        return out

    def message(self, edge_index_i, x_j, x_norm_i, x_norm_j, num_nodes):
        # Compute attention coefficients.
        beta = self.beta if self.requires_grad else self._buffers['beta']
        # alpha = beta * (1 - (x_norm_i * x_norm_j).sum(dim=-1))
        alpha = beta * (x_norm_i * x_norm_j).sum(dim=-1)
        alpha = softmax(alpha, edge_index_i, num_nodes)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        # print('max {:4f}, mean: {:.4f} var: {:.4f}'.format(alpha.max(), alpha.mean(), alpha.std()))
        return x_j * alpha.view(-1, 1)

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)
