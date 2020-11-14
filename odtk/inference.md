# 推理

我们提供两种方法来推断使用`odtk`：

- 使用训练好的模型（FP32或FP16精度）进行PyTorch推理
- 将训练好的pytorch模型导出到TensorRT以优化推理（FP32，FP16或INT8精度）

`odtk infer`将在所有可用GPU上运行分布式推理。使用PyTorch时，默认行为是以混合精度运行推理。使用TensorRT引擎进行推理时使用的精度将对应于将模型导出到TensorRT时选择的精度（请参见下面的[TensorRT部分](https://github.com/BensonBlack/retinanet-examples/blob/master/INFERENCE.md#exporting-trained-pytorch-model-to-tensorrt)）。

**注意**：HW是否支持[NVIDIA Tensor Core](https://www.nvidia.com/en-us/data-center/tensorcore/)等快速FP16和INT8精度取决于您的GPU架构：Volta或更新的GPU支持FP16和INT8，而Pascal GPU可以支持FP16或INT8。

## PyTorch推断

在COCO 2017上评估经过训练的PyTorch检测模型（混合精度）：

```
odtk infer model.pth --images=/data/coco/val2017 --annotations=instances_val2017.json --batch 8
```

**注意**：`--batch N`指定用于推理的*全局*批处理大小。每个GPU的批处理大小为`N // num_gpus`。

评估期间使用全精度（FP32）：

```
odtk infer model.pth --images=/data/coco/val2017 --annotations=instances_val2017.json --full-precision
```

以较小的输入图像尺寸评估PyTorch检测模型：

```
odtk infer model.pth --images=/data/coco/val2017 --annotations=instances_val2017.json  --resize 400 --max-size 640
```

在这里，`resize`只要长边的长度不大于`max-size`，输入图像的短边将被调整为，否则，输入图像的长边将被调整为`max-size`。

**注意**：为获得最佳准确性，建议在首选的导出尺寸下训练模型。

使用您自己的数据集运行推理：

```
odtk infer model.pth --images=/data/your_images --output=detections.json
```

## 将训练有素的PyTorch模型导出到TensorRT

`odtk`提供了一个简单的工作流程来优化经过训练的PyTorch模型以使用TensorRT进行推理部署。将PyTorch模型导出到[ONNX](https://github.com/onnx/onnx)，然后由TensorRT使用并优化ONNX模型。要了解有关TensorRT优化的更多信息，请参见此处：[https](https://developer.nvidia.com/tensorrt) : [//developer.nvidia.com/tensorrt](https://developer.nvidia.com/tensorrt)

**注意**：当使用TensorRT优化模型时，输出是可用于部署的TensorRT引擎（.plan文件）。该TensorRT引擎具有在导出过程中指定的几个固定属性。

- 输入图像大小：TensorRT引擎仅支持固定的输入大小。
- 精度：TensorRT支持FP32，FP16或INT8精度。
- 目标GPU：TensorRT优化与执行优化的系统上的GPU类型相关。它们不能在不同类型的GPU之间转移。换句话说，如果您打算在Tensor T4 GPU上部署TensorRT引擎，则必须在具有T4 GPU的系统上运行优化。

将训练有素的PyTorch检测模型导出到TensorRT的工作流程非常简单：

```
odtk export model.pth model_fp16.plan --size 1280
```

这将创建一个针对TensorRT引擎进行优化的批次大小1，使用1280x1280的输入大小。默认情况下，引擎将被创建为以FP16精度运行。

使用非平方输入大小导出模型以使用全精度：

```
odtk export model.pth model_fp32.plan --full-precision --size 800 1280
```

为了将TensorRT与INT8精度配合使用，您需要提供用于重新调整网络规模的校准图像（代表运行时将看到的图像）。

```
odtk export model.pth model_int8.plan --int8 --calibration-images /data/val/ --calibration-batches 10 --calibration-table model_calibration_table
```

这将从中随机选择20张图像`/data/val/`以校准网络以达到INT8精度。校准结果将保存到该结果中`model_calibration_table`，可用于为该模型创建后续的INT8引擎，而无需重新校准。

为先前校准的模型构建INT8引擎：

```
odtk export model.pth model_int8.plan --int8 --calibration-table model_calibration_table
```

## 使用TensorRT在NVIDIA Jetson AGX Xavier上进行部署

我们提供了一条使用TensorRT将训练后的模型部署到[NVIDIA Jetson AGX Xavier](https://developer.nvidia.com/embedded/buy/jetson-agx-xavier-devkit)等嵌入式平台上的途径，在该平台上PyTorch尚不可用。

您将需要将训练有素的PyTorch模型导出到主机系统上的ONNX表示形式，并将生成的ONNX模型复制到Jetson AGX Xavier：

```
odtk export model.pth model.onnx --size 800 1280
```

请参阅有关使用示例cppapi代码构建TensorRT引擎并在此处运行推理的其他文档：[cppapi示例代码](https://github.com/BensonBlack/retinanet-examples/blob/master/extras/cppapi/README.md)

## 旋转检测

*旋转ODTK*允许用户训练和推断图像中的旋转边界框。

### 推理

一个示例命令：

```
odtk infer model.pth --images /data/val --annotations /data/val_rotated.json --output /data/detections.json \ 
    --resize 768 --rotated-bbox
```

### 导出

可以导出旋转的边界框模型以通过使用axisaligned命令添加来创建TensorRT引擎`--rotated-bbox`。