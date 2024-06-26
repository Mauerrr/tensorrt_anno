#!/usr/bin/env python3
# Template auto-generated by polygraphy [v0.44.2] on 07/19/23 at 20:33:32
# Generation Command: /home/kalfazed/packages/TensorRT-8.5.1.7/bin/polygraphy template trt-network sample-cbr.onnx -o sample-cbr-template.py
# Creates a TensorRT Network using the Network API.
from polygraphy import mod
from polygraphy import func
from polygraphy.backend.trt import NetworkFromOnnxPath
trt = mod.lazy_import('tensorrt')

# Loaders
parse_network_from_onnx = NetworkFromOnnxPath('/home/kalfazed/Code/deep_learning/inference/tensorrt-starter/chapter3-tensorrt-basics-and-onnx/3.8-polygraphy-basics/models/sample-cbr.onnx')

@func.extend(parse_network_from_onnx)
def load_network(builder, network, parser):
    pass # TODO: Set up the network here. This function should not return anything.
