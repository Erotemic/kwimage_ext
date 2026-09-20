"""GPU NMS is not provided by the Rust rewrite."""

def gpu_nms(*args, **kwargs):
    raise NotImplementedError("GPU NMS is out of scope for the Rust backend")
