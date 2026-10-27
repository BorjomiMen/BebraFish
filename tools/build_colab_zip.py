"""Build a portable ZIP. --include-weights adds local checkpoints."""
from pathlib import Path
import argparse
import zipfile

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--include-weights',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    target=root/'dist'/('diffusion_zebrafish_full.zip' if args.include_weights else 'diffusion_zebrafish_code.zip')
    target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in sorted(root.rglob('*')):
            if not p.is_file():
                continue
            rel=p.relative_to(root)
            if rel.parts[0] in ('dist','outputs','.git','.venv') or '__pycache__' in rel.parts:
                continue
            if rel.parts[0]=='data' and 'examples' not in rel.parts:
                continue
            if p.suffix in ('.pt','.pth') and not args.include_weights:
                continue
            z.write(p,rel.as_posix())
    print(target)

if __name__=='__main__':
    main()
