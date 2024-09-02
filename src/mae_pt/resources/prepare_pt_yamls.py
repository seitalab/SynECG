import os
import sys
from itertools import product

dirnames = {
    "p": "pretrain_yamls",
}

conditions = {

    "p": {
        "template001": 
            {
                "TARGET01": [
                    "syn_ecg-01",
                    "syn_afib-01",
                    "syn_aflt-01",
                    "syn_pvc-01",
                    "syn_wpw-01",
                    "syn_ecg-01//syn_afib-01",
                    "syn_ecg-01//syn_aflt-01",
                    "syn_ecg-01//syn_pvc-01",
                    "syn_ecg-01//syn_wpw-01",
                ],
            }, 
        "template002":
            {
                "TARGET01": [
                    "gen-vae/v01",
                ],                
            }
    }
}
base_id = {
    "p": {
        "template001": 1,
        "template002": 3001,
    },
}

def make_yaml(template, keys, vals):
    for key, val in zip(keys, vals):
        template = template.replace(key, val)
    return template

def main(template_id: int, template_type: str):
    """
    Args:
        template_id (int): 
        template_type (str): 
    Returns:
        None
    """
    assert template_type in ["p", "e", "m"]

    template_id = f"template{template_id:03d}"
    template_dir = dirnames[template_type]
    template_file = f"./templates/{template_dir}/{template_id}.yaml"
    template = open(template_file).read()

    keys = conditions[template_type][template_id].keys()
    val_combs = list(product(*conditions[template_type][template_id].values()))
    for idx, val_comb in enumerate(val_combs):
        exp_yaml = make_yaml(template, keys, val_comb)
        yaml_id = base_id[template_type][template_id] + idx

        dirname = os.path.join(
            ".",
            template_dir,
            f"{template_type}{(yaml_id//100):02d}s"
        )
        os.makedirs(dirname, exist_ok=True)
        if template_type == "e":
            savename = dirname + f"/exp{yaml_id:04d}.yaml"
        elif template_type == "p":
            savename = dirname + f"/pt{yaml_id:04d}.yaml"
        elif template_type == "m":
            savename = dirname + f"/check{yaml_id:04d}.yaml"
        else:
            raise
        
        with open(savename, "w") as f:
            f.write(exp_yaml)
        print(savename)
    print("Done")

if __name__ == "__main__":

    template_id = sys.argv[1]

    template_type = template_id[-1]
    template_id = int(template_id[:-1])
    main(template_id, template_type)
