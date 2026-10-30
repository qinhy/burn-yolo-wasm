"""Train exactly three Linear layers; export little-endian FP32 and parity fixtures."""
import argparse
import gzip
import hashlib
import json
import urllib.request
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent

class TokenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Linear(49, 16)
        self.hidden = nn.Linear(256, 64)
        self.head = nn.Linear(64, 10)

    def forward(self, x):
        # Row-major 4x4 grid of nonoverlapping 7x7 patches, no convolution.
        x = x.reshape(-1, 4, 7, 4, 7).permute(0, 1, 3, 2, 4).reshape(-1, 16, 49)
        x = self.patch(x).relu().flatten(1)
        return self.head(self.hidden(x).relu())

def dataset(cache, split):
    prefix = 'train' if split == 'train' else 't10k'
    arrays = []
    for kind, offset in [('images-idx3-ubyte', 16), ('labels-idx1-ubyte', 8)]:
        name = f'{prefix}-{kind}.gz'
        path = cache / name
        if not path.exists():
            urllib.request.urlretrieve('https://storage.googleapis.com/cvdf-datasets/mnist/' + name, path)
        raw = gzip.decompress(path.read_bytes())
        arrays.append(torch.from_numpy(np.frombuffer(raw, dtype=np.uint8, offset=offset).copy()))
    return arrays[0].reshape(-1, 784).float() / 255, arrays[1].long()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    if args.epochs < 1 or args.threads < 1:
        p.error('--epochs and --threads must be positive')
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    cache = ROOT / 'data'
    cache.mkdir(exist_ok=True)
    out = ROOT / 'artifacts'
    out.mkdir(exist_ok=True)
    x, y = dataset(cache, 'train')
    tx, ty = dataset(cache, 'test')
    model = TokenMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = []
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(x))
        loss_sum = 0.0
        for ids in order.split(256):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x[ids]), y[ids])
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(ids)
        model.eval()
        with torch.no_grad():
            logits = model(tx)
            accuracy = (logits.argmax(1) == ty).float().mean().item()
        history.append(dict(epoch=epoch+1, loss=loss_sum/len(x), test_accuracy=accuracy))
        print(history[-1], flush=True)
    # Each linear weight is output-major, then its bias. No header or padding.
    weights = b''.join(t.detach().numpy().astype('<f4').tobytes() for layer in [model.patch, model.hidden, model.head] for t in [layer.weight, layer.bias])
    (out / 'weights.bin').write_bytes(weights)
    # 100 held-out examples, original bytes and reference logits; never training data.
    fixtures = [dict(label=int(ty[i]), pixels=(tx[i]*255).round().byte().tolist(), logits=logits[i].tolist()) for i in range(100)]
    (out / 'fixtures.json').write_text(json.dumps(fixtures, separators=(',', ':')))
    report = dict(seed=42, epochs=args.epochs, torch_version=torch.__version__, architecture='784 -> 16x49 tokens -> Linear(49,16)+ReLU -> flatten(256) -> Linear(256,64)+ReLU -> Linear(64,10)', parameters=sum(t.numel() for t in model.parameters()), weights_bytes=len(weights), sha256=hashlib.sha256(weights).hexdigest(), test_examples=len(tx), history=history)
    (out / 'training.json').write_text(json.dumps(report, indent=2)+'\n')

if __name__ == '__main__':
    main()
