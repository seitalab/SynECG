import os
import sys
from itertools import product

space = "  "

dx_dataset_comb = {

    "ptbxl-af": {
        "dx": "af",
        "pos": "PTBXL-AFIB",
        "neg": "PTBXL-NORM",
    },
    "ptbxl-pvc": {
        "dx": "pvc",
        "pos": "PTBXL-PVC",
        "neg": "PTBXL-NORM",
    },
    "ptbxl-pac": {
        "dx": "pac",
        "pos": "PTBXL-PAC",
        "neg": "PTBXL-NORM",
    },
    "ptbxl-irbbb": {
        "dx": "irbbb",
        "pos": "PTBXL-IRBBB",
        "neg": "PTBXL-NORM",
    },
    "ptbxl-crbbb": {
        "dx": "crbbb",
        "pos": "PTBXL-CRBBB",
        "neg": "PTBXL-NORM",
    },
    "ptbxl-std": {
        "dx": "std",
        "pos": "PTBXL-STD_",
        "neg": "PTBXL-NORM",
    },
    "ptbxl-asmi": {
        "dx": "asmi",
        "pos": "PTBXL-ASMI",
        "neg": "PTBXL-NORM",
    },
    "ptbxl-imi": {
        "dx": "imi",
        "pos": "PTBXL-IMI",
        "neg": "PTBXL-NORM",
    },

    "ptbxl-lvh": {
        "dx": "lvh",
        "pos": "PTBXL-LVH",
        "neg": "PTBXL-NORM",
    },
    "ptbxl-lafb": {
        "dx": "lafb",
        "pos": "PTBXL-LAFB",
        "neg": "PTBXL-NORM",
    },
    "ptbxl-isc": {
        "dx": "isc",
        "pos": "PTBXL-ISC_",
        "neg": "PTBXL-NORM",
    },
    "ptbxl-iavb": {
        "dx": "iavb",
        "pos": "PTBXL-1AVB",
        "neg": "PTBXL-NORM",
    },
    "ptbxl-abqrs": {
        "dx": "abqrs",
        "pos": "PTBXL-ABQRS",
        "neg": "PTBXL-NORM",
    },
}

dx_set_dict = {
    "dx_set06": [
        "ptbxl-af",
        "ptbxl-pvc",
        "ptbxl-pac",
        "ptbxl-irbbb",
        "ptbxl-crbbb",
        "ptbxl-std",
        "ptbxl-asmi",
        "ptbxl-imi",
        "ptbxl-lvh",
        "ptbxl-lafb",
        "ptbxl-isc",
        "ptbxl-iavb",
        "ptbxl-abqrs",
    ],
}


dgm_pt_models = {
    "vae-gan": [
        "pt3001",
        "pt3002"
    ],
}

conditions = {
    "t0101":
        {
            "template_file": "template0101.yaml",
            "TARGET-PTMODEL": dgm_pt_models["vae-gan"],
            "TARGET-DX": dx_set_dict["dx_set06"],
    }, 
}



base_id = {

    # MAE hps
    "t0101": 8001,
}

def swap_val(template, key, val):
    """
    Args:
        template (str):
        key (str): 
        val (str):         
    Returns:
        template
    """
    pattern_src = f'{key}:\n{space}param_type: fixed\n{space}param_val: XXX'
    pattern_dst= f'{key}:\n{space}param_type: fixed\n{space}param_val: {val}'
    template = template.replace(pattern_src, pattern_dst)
    return template

def make_yaml(template, keys, vals):
    """
    Args:
        template (str):
        keys (str): `TARGET-xxx`
        vals (str): 
    Returns: 
        template (str): 
    """
    for key, val in zip(keys, vals):
        if key == "TARGET-DX":
            data_info = dx_dataset_comb[val]
            assert template.find("TARGET-DS-POS") != -1
            assert template.find("TARGET-DS-NEG") != -1
            template = template.replace("TARGET-DS-POS", data_info["pos"])
            template = template.replace("TARGET-DS-NEG", data_info["neg"])
            template = template.replace("TARGET-DX", data_info["dx"])
        else:
            template = template.replace(key, val)
    return template

def get_key_index(keys, target):
    if target in keys:
        key_index = list(keys).index(target)
    else:
        key_index = None
    return key_index

def main(template_key: int):
    """
    Args:
        template_key (int): 
    Returns:
        None
    """
    template_key = f"t{template_key:04d}"
    template_yaml = conditions[template_key]['template_file']
    template_file = f"./templates/exp_yamls/{template_yaml}"
    template = open(template_file).read()

    template_conditions = conditions[template_key]

    # if "add_hps" in template_conditions:
    #     add_hps = template_conditions["add_hps"]
    #     del template_conditions["add_hps"]
    # else:
    #     add_hps = True

    del template_conditions["template_file"]
    keys = template_conditions.keys()
    val_combs = list(product(*template_conditions.values()))

    # model_key_index = get_key_index(keys, "TARGET-MODELNAME")
    # dx_key_index = get_key_index(keys, "TARGET-DX")
    # pt_key_index = get_key_index(keys, "TARGET-PTMODEL")

    for idx, val_comb in enumerate(val_combs):
        # if model_key_index is not None:
        #     model_template = add_model_info(template, val_comb[model_key_index])
        #     modelname = val_comb[model_key_index]
        # else:
        model_template = template
        # modelname = "mae-" + val_comb[pt_key_index]

        exp_yaml = make_yaml(model_template, keys, val_comb)
        # if add_hps:
        #     exp_yaml = add_hps_info(
        #         exp_yaml, modelname, val_comb[dx_key_index])
        yaml_id = base_id[template_key] + idx

        dirname = os.path.join(
            ".",
            "exp_yamls",
            f"exp{(yaml_id//100):02d}s"
        )
        os.makedirs(dirname, exist_ok=True)
        savename = dirname + f"/exp{yaml_id:04d}.yaml"
        
        assert exp_yaml.find("TARGET") == -1
        with open(savename, "w") as f:
            f.write(exp_yaml)
        print(savename)
    print("Done")

if __name__ == "__main__":
    import sys
    template_key = sys.argv[1]

    template_key = int(template_key)
    main(template_key)
