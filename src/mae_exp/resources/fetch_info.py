import os
from glob import glob

import yaml

cfg_file = "../../../config.yaml"
with open(cfg_file) as f:
    config = yaml.safe_load(f)["experiment"]

root = config["path"]["processed_data"]
id_range = range(1, 85)

def load_cfg(yaml_id):
    """
    Args:

    Returns:

    """
    loc = os.path.join(
        root, 
        f"clf-exp{yaml_id//100:02d}s",
        f"exp{yaml_id:04d}"
    )
    targets = sorted(glob(loc + "/??????-??????"))
    target = targets[-1]
    yaml_file = glob(target + "/exp????.yaml")[0]
    with open(yaml_file, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg, yaml_file

def get_path(target_model, target_dx, ft_target, target_syn):
    """
    Args:

    Returns:

    """
    for yaml_id in id_range:
        cfg, yaml_loc = load_cfg(yaml_id)

        # Check model.
        model = cfg["modelname"]["param_val"]
        if model != target_model:
            continue

        # Check dx
        dx = cfg["target_dx"]["param_val"]
        if dx != target_dx:
            continue

        # Check ft_target
        ft = cfg["finetune_target"]["param_val"]
        if ft != ft_target:
            continue
        
        pos_ds = cfg["pos_dataset"]["param_val"]
        is_syn = pos_ds.startswith("syn_")
        if is_syn == target_syn:
            print("-"*80)
            path = yaml_loc.replace(root, "")[1:]
            print(os.path.dirname(path))
    print("Done")

if __name__ == "__main__":
    import sys

    target_model = sys.argv[1]
    target_ft = sys.argv[2]
    try:
        target_dx = sys.argv[3]
    except:
        target_dx = None
    target_syn = False
    
    if target_ft == "none":
        target_ft = None
    if target_dx is None:
        target_dx = target_ft.split("_")[-1]

    get_path(target_model, target_dx, target_ft, target_syn)