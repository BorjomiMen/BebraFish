"""Copy checkpoints from Drive/local paths without changing their contents."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--unet', required=True)
    p.add_argument('--diffusion', required=True)
    p.add_argument('--root', default=str(Path(__file__).resolve().parents[1]))
    a = p.parse_args()
    import torch
    root = Path(a.root)
    entries = []
    for kind, source, expected, target in (
        ('unetpp',Path(a.unet),'naomi-unetpp-fullframe-v1',root/'weights/unetpp/best.pt'),
        ('diffusion',Path(a.diffusion),'neural-naomi-static-vdiffusion-v1',None)):
        state = torch.load(source,map_location='cpu',weights_only=True)
        if state.get('format') != expected:
            raise ValueError(f'{source}: incompatible format {state.get("format")}')
        step = state.get('step')
        if kind == 'diffusion':
            target = root/'weights/diffusion'/f'step_{step:06d}.pt'
        target.parent.mkdir(parents=True,exist_ok=True)
        if source.resolve() != target.resolve():
            if target.exists():
                stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
                backup = target.with_name(target.name+'.'+stamp+'.previous')
                shutil.copy2(target,backup)
            shutil.copy2(source,target)
        entries.append(dict(kind=kind,path=str(target.relative_to(root)),format=expected,
                            step=step,epoch=state.get('epoch'),bytes=target.stat().st_size,
                            sha256=digest(target)))
        del state
    manifest=root/'weights/imported.json'
    manifest.write_text(json.dumps(entries,ensure_ascii=False,indent=2),encoding='utf-8')
    print(manifest.read_text(encoding='utf-8'))

if __name__ == '__main__':
    main()
