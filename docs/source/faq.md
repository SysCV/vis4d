# Troubleshooting

## File limits
```bash
RuntimeError: received 0 items of ancdata
```

Please check `ulimit -n` on your machine and if it is of value 1024 or lower, try increasing it to 2048 or 4096. See https://github.com/pytorch/pytorch/issues/973 for further info.


## CPU out of memory

```bash
OSError: [Errno 12] Cannot allocate memory
```

Try setting `workers_per_gpu` to a lower value (usually 1/2 of `samples_per_gpu` is sufficient).

## Memory leakage during training

You might experience RAM keep increasing during training. Besides lower the workers of data loader, it might be also related to the `databackend`. If you're using `HDF5Backend` or `ZipBackend`, you can close the backend in every `__getitem__` call to prevent memory leakage.
