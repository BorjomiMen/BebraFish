"""Command line interface for preparation, training and inference."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

def main(argv=None):
    parser = argparse.ArgumentParser(prog='zebrafish-diffusion')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare-diffusion','prepare-unet'):
        p = sub.add_parser(name)
        p.add_argument('--source', required=True)
        p.add_argument('--output', required=True)
        p.add_argument('--image-key', default=None)
        if name == 'prepare-unet':
            p.add_argument('--mask-key', default=None)
        else:
            p.add_argument('--split-method', choices=['auto','groups','temporal'], default='auto')
            p.add_argument('--groups-csv', default=None)
            p.add_argument('--val-fraction', type=float, default=.15)
            p.add_argument('--test-fraction', type=float, default=.15)
            p.add_argument('--seed', type=int, default=42)
            p.add_argument('--gap', type=int, default=32)
    p = sub.add_parser('generate')
    p.add_argument('--diffusion', default='weights/diffusion/step_003000.pt')
    p.add_argument('--unet', default='weights/unetpp/best.pt')
    p.add_argument('--seed', type=int, default=20331027)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--count', type=int, default=1)
    p.add_argument('--output', default='outputs/generated')
    p.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    p.add_argument('--threads', type=int, default=None)
    p.add_argument('--threshold', type=float, default=None)
    p.add_argument('--tta', action='store_true')
    p.add_argument('--no-amp', action='store_true')
    p = sub.add_parser('segment')
    p.add_argument('--input', required=True)
    p.add_argument('--unet', default='weights/unetpp/best.pt')
    p.add_argument('--output', default='outputs/segmentation')
    p.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    p.add_argument('--threshold', type=float, default=None)
    p.add_argument('--tta', action='store_true')
    p.add_argument('--no-amp', action='store_true')
    sub.add_parser('train-unet', add_help=False)
    sub.add_parser('train-diffusion', add_help=False)
    import sys
    raw = list(sys.argv[1:] if argv is None else argv)
    if '--config' in raw:
        index = raw.index('--config')
        if index+1 >= len(raw):
            parser.error('--config requires a JSON path')
        config = json.loads(Path(raw[index+1]).read_text(encoding='utf-8'))
        mapping = {'generate': {'diffusion_checkpoint':'diffusion', 'unet_checkpoint':'unet', 'sample_steps':'steps'}}.get(raw[0], {})
        defaults = []
        for key,value in config.items():
            if value is None or value is False:
                continue
            defaults.append('--'+mapping.get(key,key).replace('_','-'))
            if value is not True:
                defaults.extend(map(str,value if isinstance(value,list) else [value]))
        raw = [raw[0],*defaults,*raw[1:index],*raw[index+2:]]
    # Delegate training argument parsers without losing --help or checkpoint args.
    if raw and raw[0] == 'train-diffusion':
        from . import diffusion
        args = diffusion.parse_args(['--mode','train',*raw[1:]])
        print(diffusion.train(args))
        return
    if raw and raw[0] == 'train-unet':
        from . import unetpp
        previous = sys.argv
        try:
            sys.argv = ['train-unet',*raw[1:]]
            unetpp.main()
        finally:
            sys.argv = previous
        return
    args = parser.parse_args(raw)
    if args.command == 'prepare-diffusion':
        from .data import prepare_diffusion
        prepare_diffusion(args.source,args.output,args.image_key,args.split_method,args.groups_csv,
                          args.val_fraction,args.test_fraction,args.seed,args.gap)
    elif args.command == 'prepare-unet':
        from .data import prepare_segmentation
        prepare_segmentation(args.source,args.output,args.image_key,args.mask_key)
    elif args.command == 'generate':
        from .api import Pipeline
        if args.count < 1:
            parser.error('count must be positive')
        pipeline = Pipeline(args.diffusion,args.unet,args.device,not args.no_amp,args.threshold,args.threads)
        for i in range(args.count):
            result = pipeline.generate(args.seed+i,args.steps,tta=args.tta)
            print(json.dumps(pipeline.save(result,args.output),ensure_ascii=False,indent=2))
    elif args.command == 'segment':
        from .api import segment_file
        print(json.dumps(segment_file(args.input,args.unet,args.output,args.device,args.threshold,
                         args.tta,not args.no_amp),ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
