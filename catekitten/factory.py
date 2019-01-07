from catekitten.network import YoonKimCNN, YoonKimCNNv2, YoonKimCNNv3, BidirectionalCNN
from catekitten.han import HAN


NETWORKS = {
    'YoonKimCNN': YoonKimCNN,
    'YoonKimCNNv2': YoonKimCNNv2,
    'YoonKimCNNv3': YoonKimCNNv3,
    'BidirectionalCNN': BidirectionalCNN,
    'HAN': HAN,
}