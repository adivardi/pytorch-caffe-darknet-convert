import torch
import onnx2pytorch
from cfg_onnx2pytorch import save_conv, save_conv_bn
from cfg import save_fc
from IPython import embed

def save_custom_weights(model, filename):
    print("-------------------")
    fp = open(filename, 'wb')
    header = torch.IntTensor([0,0,0,0])
    header.numpy().tofile(fp)
    layers_types = set()

    modules_list = list(model._modules.items())
    modules_done = [False] * len(modules_list)
    for index, (key, layer) in enumerate(modules_list):
        layers_types.add(type(layer))
        # print(f"{key}  : {type(layer)}  :  {layer}")
        print(f"\n{key}  : {type(layer)} ")

        if modules_done[index] == True:
            print(f"skipped layer done {key}")
            continue
        if type(layer) == torch.nn.Conv2d:
            next_key, next_layer = modules_list[index + 1]
            if type(next_layer) == onnx2pytorch.operations.batchnorm.BatchNormWrapper:
                # BatchNormWrapper.bnu => onnx2pytorch.operations.batchnorm.BatchNormUnsafe
                save_conv_bn(fp, layer, next_layer.bnu)
                modules_done[index + 1] = True
            else:
                save_conv(fp, layer)

        elif type(layer) == torch.nn.modules.activation.ReLU:
            print(f"Skipped RELU module. previous layer is {type(modules_list[index -1][1])}")
            pass
            # if next_next_key, next_next_layer = modules_list[index + 1]
        elif type(layer) == torch.nn.Linear:
            save_fc(fp, layer)
        elif type(layer) == onnx2pytorch.operations.add.Add:
            if layer.input_indices:
                print(f"Found ADD module with input indices of {layer.input_indices} - not supported!")
                exit(-1)
            else:
                print("Skipped ADD module")
        elif type(layer) == onnx2pytorch.operations.pad.Pad:    # implements torch.nn.functional.pad
            if any(layer.padding):  # check for non 0 padding
                print(f"Found PAD module with input indices of {layer.padding} - not supported!")
                exit(-1)
            else:
                print(f"Skipped PAD module")
        elif type(layer) == torch.nn.modules.pooling.AvgPool2d:
            print("Skipped AvgPool2D module, since no weights")
        elif type(layer) == onnx2pytorch.operations.shape.Shape:
            print("Skipped Shape module")
        elif type(layer) == onnx2pytorch.operations.constant.Constant:
            print("Skipped Constant module")
        elif type(layer) == onnx2pytorch.operations.gather.Gather:
            print("Skipped Gather module")
        elif type(layer) == onnx2pytorch.operations.unsqueeze.Unsqueeze:
            # TODO maybe need to add [unsqueeze] tag to darknet cfg to add torch.unsqueeze(data, dim=dim)
            print("Skipped Unsqueeze module")
        elif type(layer) == onnx2pytorch.operations.reshape.Reshape:
            print("Skipped Reshape module")
        else:
            print(f"Missing layer {type(layer)}")
            exit(-1)

        modules_done[index] = True

    print(f"layers_types: {layers_types}")
    print(f"modules_done: {modules_done}")


"""
{
--- <class 'onnx2pytorch.operations.batchnorm.BatchNormWrapper'>,
--- <class 'onnx2pytorch.operations.pad.Pad'>,
??? <class 'onnx2pytorch.operations.unsqueeze.Unsqueeze'>,
<class 'onnx2pytorch.operations.reshape.Reshape'>,
??? <class 'onnx2pytorch.operations.shape.Shape'>,
??? <class 'onnx2pytorch.operations.add.Add'>,
--- <class 'torch.nn.modules.linear.Linear'>,
--- <class 'torch.nn.modules.activation.ReLU'>,
--- <class 'torch.nn.modules.pooling.AvgPool2d'>,
??? <class 'onnx2pytorch.operations.constant.Constant'>,
--- <class 'torch.nn.modules.conv.Conv2d'>,
??? <class 'onnx2pytorch.operations.gather.Gather'>
}
"""

model_path = '/home/adi/Projects/traffic_lights/traffic_lights_detection/models/autoware2_traffic_light_classifier_mobilenetv2_model.pt'
result_path = '/home/adi/Projects/traffic_lights/traffic_lights_detection/models/autoware2_traffic_light_classifier_mobilenetv2_model_darknet.weights'
device = 'cpu'  # don't use cuda for conversion


model = torch.load(model_path)
if (next(model.parameters()).is_cuda):
    model.to('cpu')

print(next(model.parameters()).is_cuda)

# if (model.device != 'cpu'):
#     exit(-1)
print(model)
# print(model._modules)
print(f"convert pytorch model to darknet, save to {result_path}")
save_custom_weights(model, result_path)

print("Done")
