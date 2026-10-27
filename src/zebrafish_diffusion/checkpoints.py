"""Select a local diffusion checkpoint without loading model tensors."""
from pathlib import Path
import re

def latest_diffusion(folder='weights/diffusion'):
    candidates=[]
    for path in Path(folder).glob('step_*.pt'):
        match=re.fullmatch(r'step_(\d+)\.pt',path.name)
        if match and path.is_file():
            candidates.append((int(match.group(1)),path))
    return max(candidates,key=lambda item:item[0])[1] if candidates else None
