#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from hackathon.egoflow.interaction_model import score_expected_dynamics, train_expected_dynamics

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--hand-dir',type=Path,required=True); parser.add_argument('--split',type=Path,required=True); parser.add_argument('--output-dir',type=Path,required=True); parser.add_argument('--hidden-size',type=int,default=32); parser.add_argument('--window',type=int,default=8); parser.add_argument('--max-steps',type=int,default=400); parser.add_argument('--score-episode')
    args=parser.parse_args(); split=json.loads(args.split.read_text()); result=train_expected_dynamics(args.hand_dir,split['train'],args.output_dir,hidden_size=args.hidden_size,window=args.window,max_steps=args.max_steps)
    if args.score_episode: result['score']=score_expected_dynamics(args.hand_dir/f'{args.score_episode}.json',result['checkpoint'],args.output_dir/f'{args.score_episode}.json')
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
