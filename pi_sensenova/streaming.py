"""Safe synchronous block streaming for packed tensor subclasses.

The upstream offloader mutates Parameter.data and can strand packed storage on
CPU. Here model-owned modules use their own _apply implementation, and exact
CPU tensors are retained for restoration. No quantized tensor is unpacked,
re-quantized or merged. No extra CPU copy is made for each sampling step.
"""
import torch


class BlockStream:
    def __init__(self, model, target="cuda"):
        self.model = model
        self.target = target
        self.saved = []
        self.handles = []

    @staticmethod
    @torch.inference_mode(False)
    def capture(module):
        entries = []
        for child in module.modules():
            for name, tensor in child._parameters.items():
                if tensor is not None:
                    entries.append((child, name, tensor.detach(), True))
            for name, tensor in child._buffers.items():
                if tensor is not None:
                    entries.append((child, name, tensor.detach(), False))
        return entries

    @staticmethod
    @torch.inference_mode(False)
    def restore(entries):
        for module, name, tensor, parameter in entries:
            if parameter:
                module._parameters[name] = torch.nn.Parameter(tensor, requires_grad=False)
            else:
                module._buffers[name] = tensor

    def __enter__(self):
        layers = self.model.language_model.model.layers
        self.saved = self.capture(self.model)
        layer_ids = {id(child) for layer in layers for child in layer.modules()}
        try:
            for module in self.model.modules():
                if id(module) not in layer_ids:
                    module._apply(lambda tensor: tensor.to(self.target), recurse=False)
            for layer in layers:
                saved = self.capture(layer)
                def before(module, args):
                    with torch.inference_mode(False):
                        module.to(self.target)
                def after(module, args, result, saved=saved):
                    self.restore(saved)
                self.handles.append(layer.register_forward_pre_hook(before))
                self.handles.append(layer.register_forward_hook(after, always_call=True))
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self.model

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.restore(self.saved)
        self.saved.clear()
