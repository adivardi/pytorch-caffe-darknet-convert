import torch
from collections import OrderedDict
import onnx2pytorch


def load_conv(buf, start, conv_model):
    num_w = conv_model.weight.numel()
    num_b = conv_model.bias.numel()
    # conv_model.bias.data.copy_(torch.from_numpy(buf[start:start+num_b]));   start = start + num_b
    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_w])); start = start + num_w
    return start

def save_conv(fp, conv_model):
    # if conv_model.bias.is_cuda:
    #     convert2cpu(conv_model.bias.data).numpy().tofile(fp)
    #     convert2cpu(conv_model.weight.data).numpy().tofile(fp)
    # else:
    if conv_model.bias:
        conv_model.bias.data.numpy().tofile(fp)
    conv_model.weight.data.numpy().tofile(fp)

def save_conv_bn(fp, conv_model, bn_model: onnx2pytorch.operations.batchnorm.BatchNormUnsafe):
    # BatchNormUnsafe inherits from torch.nn.modules.batchnorm

    # if bn_model.bias.is_cuda:
    #     convert2cpu(bn_model.bias.data).numpy().tofile(fp)
    #     convert2cpu(bn_model.weight.data).numpy().tofile(fp)
    #     convert2cpu(bn_model.running_mean).numpy().tofile(fp)
    #     convert2cpu(bn_model.running_var).numpy().tofile(fp)
    #     convert2cpu(conv_model.weight.data).numpy().tofile(fp)
    # else:
    bn_model.bias.data.numpy().tofile(fp)
    bn_model.weight.data.numpy().tofile(fp)
    bn_model.running_mean.numpy().tofile(fp)
    bn_model.running_var.numpy().tofile(fp)
    if conv_model.bias:
        conv_model.bias.data.numpy().tofile(fp)
    conv_model.weight.data.numpy().tofile(fp)

