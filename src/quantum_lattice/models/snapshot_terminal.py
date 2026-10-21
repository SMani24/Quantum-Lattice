"""Small launch/progress menu for the raw-configuration development pilot."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time

from . import snapshot_campaign as campaign


class ProgressLine:
    def __init__(self,stream=None):
        self.stream=stream or sys.stdout
        self.last=0.
        self.stage=None

    def __call__(self,event):
        stage=event.get('stage',event.get('state'))
        now=time.monotonic()
        if not self.stream.isatty() and stage==self.stage and now-self.last<5:return
        self.last=now;self.stage=stage
        if stage=='training':message=f"{event['task']} | epoch {event['epoch']} | best MSE {event['best_mse']:.6f}"
        elif stage=='inverse':message=f"{event['task']} | fitted {event['completed']}/{event['total']} Hamiltonians"
        else:message=f"Preparing configurations | {event.get('completed',0)}/{event.get('total','?')} Hamiltonians"
        self.stream.write(('\r\033[K' if self.stream.isatty() else '')+message+('' if self.stream.isatty() else '\n'));self.stream.flush()

    def finish(self):
        if self.stream.isatty():self.stream.write('\n');self.stream.flush()


def print_status(status,*,stream=None):
    stream=stream or sys.stdout
    stream.write(f"\nCompleted groups: {status['completed']}/{status['total']} (neural models and per-budget inverse groups)\n")
    if not status['tasks']:
        p=status.get('data_progress')
        stream.write(f"Data: {p.get('completed',0)}/{p.get('total','?')} Hamiltonians prepared\n" if p else 'Data not prepared yet. Run/continue prepares it automatically.\n')
        return
    stream.write('Method          Records  Seed        State       Epoch / MSE\n')
    for row in status['tasks']:
        seed=str(row['seed'])[-2:] if row['seed'] is not None else '—'
        detail=f"{row['metrics']['mse']:.6f}" if row['state']=='complete' else str(row['epoch']) if row.get('epoch') is not None else '—'
        if 'inverse_progress' in row:detail=f"{row['inverse_progress']['completed']}/{row['inverse_progress']['total']} fitted"
        stream.write(f"{row['method']:<16}{row['shots']:<9}{seed:<12}{row['state']:<12}{detail}\n")
    stream.write('Results stay in this campaign directory. Validation only; no final test.\n')


def main(argv=None,*,repo_root=None):
    root=Path(repo_root or Path(__file__).resolve().parents[3]).resolve()
    parser=argparse.ArgumentParser(description='L4 raw-configuration pilot: GPU CNNs plus CPU inverse Ising')
    parser.add_argument('--config',default='configs/snapshot_learning.json')
    actions=parser.add_mutually_exclusive_group()
    actions.add_argument('--run',action='store_true',help='Prepare/resume and run every pending task')
    actions.add_argument('--prepare-only',action='store_true',help='Prepare frozen raw bags and task plan; no fitting')
    actions.add_argument('--status',action='store_true',help='Show progress and completed metric summaries')
    parser.add_argument('--workers',type=int,help='CPU inverse-Ising workers; does not change the scientific plan')
    args=parser.parse_args(argv)
    config_path=Path(args.config)
    if not config_path.is_absolute():config_path=root/config_path
    config=campaign.validate_config(json.loads(config_path.read_text()))
    run=campaign._safe_run(root,root/config['output_root'])
    preference=run/'preferences.json'
    workers=args.workers if args.workers is not None else (json.loads(preference.read_text()).get('inverse_workers') if preference.exists() else config['execution']['inverse_workers'])
    if type(workers) is not int or not 1<=workers<=128:parser.error('--workers must be in1..128')
    line=ProgressLine()
    def execute():
        print(f"\nL4 pilot | {config['selection']['train_hamiltonians']} train / {config['selection']['validation_hamiltonians']} validation H | records {config['selection']['shot_counts']}")
        print(f'GPU neural training; {workers} CPU inverse workers. Ctrl+C saves progress and returns here.')
        try:result=campaign.run_campaign(root,config,workers=workers,progress_callback=line)
        finally:line.finish()
        print_status(result)
        print('Files:',run)
    if args.status:
        print_status(campaign.progress(root,config));return 0
    if args.prepare_only:
        stopped={'value':False}
        import signal
        old=signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT,lambda *_:stopped.update(value=True))
        try:campaign.prepare(root,config,should_stop=lambda:stopped['value'],progress=line)
        finally:signal.signal(signal.SIGINT,old);line.finish()
        print_status(campaign.progress(root,config));return 0
    if args.run:
        execute();return 0
    while True:
        print(f'\nSnapshot pilot — L4 | records {config["selection"]["shot_counts"]} | CPU workers {workers}')
        print('1  Run / continue\n2  Progress / results\n3  CPU worker count\n0  Quit')
        try:choice=input('Choose: ').strip()
        except (EOFError,KeyboardInterrupt):print();return 0
        try:
            if choice=='0':return 0
            if choice=='1':execute()
            elif choice=='2':print_status(campaign.progress(root,config))
            elif choice=='3':
                value=input('Inverse-Ising CPU workers [1..128]: ').strip()
                if not value.isdigit() or not 1<=int(value)<=128:print('Enter an integer from1 to128.');continue
                workers=int(value);run.mkdir(parents=True,exist_ok=True);campaign._json(preference,{'inverse_workers':workers})
            else:print('Choose0,1,2 or3.')
        except (ValueError,OSError,RuntimeError) as error:
            line.finish();print(f'\nStopped safely: {error}')
            print('For GPU errors, check nvidia-smi and use this repository’s .venv. Restore pinned settings to resume, or use a new campaign directory.')


if __name__=='__main__':
    raise SystemExit(main())
