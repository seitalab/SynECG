from argparse import ArgumentParser

from experiment import ExperimentManager
from pretrain import PretrainManager
from eval_mae import PretrainingEvaluationManager

def split_exp_targets(exp_targets):
    if exp_targets.isdigit():
        exp_ids = [int(exp_targets)]
    elif exp_targets.find(",") != -1:
        exp_ids = [int(v) for v in exp_targets.split(",")]
    elif exp_targets.find("-") != -1:
        s_e = exp_targets.split("-")
        s, e = int(s_e[0]), int(s_e[-1])
        exp_ids = [i for i in range(s, e+1)]
    else:
        raise
    return exp_ids

parser = ArgumentParser()

parser.add_argument(
    '--ids', 
    default="9001,9002"
)
parser.add_argument(
    '--device', 
    default="cuda:0"
)
parser.add_argument(
    '--mode', 
    default="e"
)

args = parser.parse_args()

errors = []
ids = split_exp_targets(args.ids)
for exe_id in ids:
    if args.mode == "p":
        executer = PretrainManager(
            int(exe_id),
            args.device,
            debug=False
        )
    elif args.mode == "m":
        executer = PretrainingEvaluationManager(
            int(exe_id),
            args.device,
            debug=False
        )
    else:
        executer = ExperimentManager(
            int(exe_id), 
            args.device,
            debug=False
        )
    try:
        executer.main(single_run=False)
    except:
        errors.append([int(exe_id), args.device])
print("*"*80)
for e in errors:
    print(e)

