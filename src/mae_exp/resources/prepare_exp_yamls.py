import os
import sys
from itertools import product

import yaml

setting_file = "./settings.yaml"
with open(setting_file, "r") as f:
    settings = yaml.safe_load(f)

conditions = {

    "t0001": # arch: MAE-trans, prior_training: MAE, data: real
        {
            "template_file": "template0001.yaml",
            "DX-DATA-PAIR": settings["dx_sets"]["dx_set01"],
            "BASEMODEL-INFO": settings["models"]["mae_models"]["set01"],
            "add_hps": False
        }, 

    "t0002": # arch: baselines01, prior_training: none, data: real
        {
            "template_file": "template0001.yaml",
            "DX-DATA-PAIR": settings["dx_sets"]["dx_set01"],
            "BASEMODEL-INFO": settings["models"]["baselines"]["set01"],
            "add_hps": False
        }, 

    "t0003": # arch: MAE-trans, prior_training: MAE, data: syn
        {
            "template_file": "template0002.yaml",
            "DX-DATA-PAIR": settings["dx_sets"]["dx_set02"],
            "BASEMODEL-INFO": settings["models"]["mae_models"]["set01"],
            "add_hps": False
        }, 

    "t0004": # arch: baselines01, prior_training: none, data: syn
        {
            "template_file": "template0002.yaml",
            "DX-DATA-PAIR": settings["dx_sets"]["dx_set02"],
            "BASEMODEL-INFO": settings["models"]["baselines"]["set01"],
            "add_hps": False
        }, 

    "t0005": # arch: baselines02, prior_training: none, data: real
        {
            "template_file": "template0001.yaml",
            "DX-DATA-PAIR": settings["dx_sets"]["dx_set01"],
            "BASEMODEL-INFO": settings["models"]["baselines"]["set02"],
            "add_hps": False
        }, 

    "t0006": # arch: baselines02, prior_training: none, data: syn
        {
            "template_file": "template0002.yaml",
            "DX-DATA-PAIR": settings["dx_sets"]["dx_set02"],
            "BASEMODEL-INFO": settings["models"]["baselines"]["set02"],
            "add_hps": False
        }, 

    "t0007": # arch: MAE-trans, prior_training: synclf, data: real
        {
            "template_file": "template0001.yaml",
            "DX-DATA-PAIR": settings["dx_sets"]["dx_set01"],
            "BASEMODEL-INFO": settings["models"]["mae_models"]["set02"],
            "add_hps": False
        }, 

    "t0008": # arch: baselines01, prior_training: synclf, data: real
        {
            "template_file": "template0001.yaml",
            "DX-DATA-PAIR": settings["dx_sets"]["dx_set01"],
            "BASEMODEL-INFO": settings["models"]["baselines"]["set03"],
            "add_hps": False
        }, 

    "t0009": # arch: baselines02, prior_training: synclf, data: real
        {
            "template_file": "template0001.yaml",
            "DX-DATA-PAIR": settings["dx_sets"]["dx_set01"],
            "BASEMODEL-INFO": settings["models"]["baselines"]["set04"],
            "add_hps": False
        }, 

    "t0010": # arch: MAE-trans, prior_training: MAE (dx), data: real
        {
            "template_file": "template0001.yaml",
            "DX-DATA-PAIR": settings["dx_sets"]["dx_set01"],
            "BASEMODEL-INFO": settings["models"]["mae_models"]["set03"],
            "add_hps": False
        }, 

    "t0011": # arch: MAE-trans, prior_training: MAE (dx), data: real
        {
            "template_file": "template0001.yaml",
            "DX-DATA-PAIR": settings["dx_sets"]["dx_set01"],
            "BASEMODEL-INFO": settings["models"]["mae_models"]["set04"],
            "add_hps": False
        }, 

    "t0012": # arch: MAE-trans, prior_training: MAE (dx), data: syn
        {
            "template_file": "template0002.yaml",
            "DX-DATA-PAIR": settings["dx_sets"]["dx_set02"],
            "BASEMODEL-INFO": settings["models"]["mae_models"]["set03"],
            "add_hps": False
        }, 

    "t0013": # arch: MAE-trans, prior_training: MAE (dx), data: syn
        {
            "template_file": "template0002.yaml",
            "DX-DATA-PAIR": settings["dx_sets"]["dx_set02"],
            "BASEMODEL-INFO": settings["models"]["mae_models"]["set04"],
            "add_hps": False
        }, 

    # Lim data
    "t0014": # arch: MAE-trans, prior_training: MAE (dx), data: real
        {
            "template_file": "template0003.yaml",
            "DX-DATA-PAIR": settings["dx_sets"]["dx_set01"],
            "BASEMODEL-INFO": settings["models"]["mae_models"]["set01"],
            "TARGET-LIMDATA": settings["data_lim_set"]["data_lim01"],
            "add_hps": True
        },

    "t0015": # arch: MAE-trans, prior_training: synclf, data: real
        {
            "template_file": "template0003.yaml",
            "DX-DATA-PAIR": settings["dx_sets"]["dx_set01"],
            "BASEMODEL-INFO": settings["models"]["mae_models"]["set02"],
            "TARGET-LIMDATA": settings["data_lim_set"]["data_lim01"],
            "add_hps": True
        },

    "t0016": # arch: baselines01, prior_training: none, data: real
        {
            "template_file": "template0003.yaml",
            "DX-DATA-PAIR": settings["dx_sets"]["dx_set01"],
            "BASEMODEL-INFO": settings["models"]["baselines"]["set01"],
            "TARGET-LIMDATA": settings["data_lim_set"]["data_lim01"],
            "add_hps": True
        },

    "t0017": # arch: baselines01, prior_training: synclf, data: real
        {
            "template_file": "template0003.yaml",
            "DX-DATA-PAIR": settings["dx_sets"]["dx_set01"],
            "BASEMODEL-INFO": settings["models"]["baselines"]["set03"],
            "TARGET-LIMDATA": settings["data_lim_set"]["data_lim01"],
            "add_hps": True
        },        

    "t0018": # arch: baselines01, prior_training: none, data: real
        {
            "template_file": "template0003.yaml",
            "DX-DATA-PAIR": settings["dx_sets"]["dx_set01"],
            "BASEMODEL-INFO": settings["models"]["baselines"]["set02"],
            "TARGET-LIMDATA": settings["data_lim_set"]["data_lim01"],
            "add_hps": True
        },

    "t0019": # arch: baselines02, prior_training: synclf, data: real
        {
            "template_file": "template0003.yaml",
            "DX-DATA-PAIR": settings["dx_sets"]["dx_set01"],
            "BASEMODEL-INFO": settings["models"]["baselines"]["set04"],
            "TARGET-LIMDATA": settings["data_lim_set"]["data_lim01"],
            "add_hps": True
        },        
    

    # lim pos.
    "t0020": # arch: MAE-trans, prior_training: MAE (dx), data: real
        {
            "template_file": "template0003.yaml",
            "DX-DATA-PAIR": settings["dx_sets"]["dx_set01"],
            "BASEMODEL-INFO": settings["models"]["mae_models"]["set01"],
            "TARGET-LIMDATA": settings["data_lim_set"]["data_lim02"],
            "add_hps": True
        },

    "t0021": # arch: MAE-trans, prior_training: synclf, data: real
        {
            "template_file": "template0003.yaml",
            "DX-DATA-PAIR": settings["dx_sets"]["dx_set01"],
            "BASEMODEL-INFO": settings["models"]["mae_models"]["set02"],
            "TARGET-LIMDATA": settings["data_lim_set"]["data_lim02"],
            "add_hps": True
        },

    "t0022": # arch: baselines01, prior_training: none, data: real
        {
            "template_file": "template0003.yaml",
            "DX-DATA-PAIR": settings["dx_sets"]["dx_set01"],
            "BASEMODEL-INFO": settings["models"]["baselines"]["set01"],
            "TARGET-LIMDATA": settings["data_lim_set"]["data_lim02"],
            "add_hps": True
        },

    "t0023": # arch: baselines01, prior_training: synclf, data: real
        {
            "template_file": "template0003.yaml",
            "DX-DATA-PAIR": settings["dx_sets"]["dx_set01"],
            "BASEMODEL-INFO": settings["models"]["baselines"]["set03"],
            "TARGET-LIMDATA": settings["data_lim_set"]["data_lim02"],
            "add_hps": True
        },      

    "t0024": # arch: baselines01, prior_training: none, data: real
        {
            "template_file": "template0003.yaml",
            "DX-DATA-PAIR": settings["dx_sets"]["dx_set01"],
            "BASEMODEL-INFO": settings["models"]["baselines"]["set02"],
            "TARGET-LIMDATA": settings["data_lim_set"]["data_lim02"],
            "add_hps": True
        },

    "t0025": # arch: baselines02, prior_training: synclf, data: real
        {
            "template_file": "template0003.yaml",
            "DX-DATA-PAIR": settings["dx_sets"]["dx_set01"],
            "BASEMODEL-INFO": settings["models"]["baselines"]["set04"],
            "TARGET-LIMDATA": settings["data_lim_set"]["data_lim02"],
            "add_hps": True
        },        
    

    # Additional models.
    "t0026": # arch: baselines03, prior_training: none, data: real
        {
            "template_file": "template0001.yaml",
            "DX-DATA-PAIR": settings["dx_sets"]["dx_set01"],
            "BASEMODEL-INFO": settings["models"]["baselines"]["set05"],
            "add_hps": False
        }, 

    "t0027": # arch: baselines03, prior_training: none, data: syn
        {
            "template_file": "template0002.yaml",
            "DX-DATA-PAIR": settings["dx_sets"]["dx_set02"],
            "BASEMODEL-INFO": settings["models"]["baselines"]["set05"],
            "add_hps": False
        }, 

    "t0028": # arch: baselines03, prior_training: synclf, data: real
        {
            "template_file": "template0001.yaml",
            "DX-DATA-PAIR": settings["dx_sets"]["dx_set01"],
            "BASEMODEL-INFO": settings["models"]["baselines"]["set06"],
            "add_hps": False
        }, 


    "t0029": # arch: baselines03, prior_training: none, data: real
        {
            "template_file": "template0003.yaml",
            "DX-DATA-PAIR": settings["dx_sets"]["dx_set01"],
            "BASEMODEL-INFO": settings["models"]["baselines"]["set05"],
            "TARGET-LIMDATA": settings["data_lim_set"]["data_lim01"],
            "add_hps": True
        },

    "t0030": # arch: baselines03, prior_training: synclf, data: real
        {
            "template_file": "template0003.yaml",
            "DX-DATA-PAIR": settings["dx_sets"]["dx_set01"],
            "BASEMODEL-INFO": settings["models"]["baselines"]["set06"],
            "TARGET-LIMDATA": settings["data_lim_set"]["data_lim01"],
            "add_hps": True
        },        


    "t0031": # arch: baselines03, prior_training: none, data: real
        {
            "template_file": "template0003.yaml",
            "DX-DATA-PAIR": settings["dx_sets"]["dx_set01"],
            "BASEMODEL-INFO": settings["models"]["baselines"]["set05"],
            "TARGET-LIMDATA": settings["data_lim_set"]["data_lim02"],
            "add_hps": True
        },

    "t0032": # arch: baselines03, prior_training: synclf, data: real
        {
            "template_file": "template0003.yaml",
            "DX-DATA-PAIR": settings["dx_sets"]["dx_set01"],
            "BASEMODEL-INFO": settings["models"]["baselines"]["set06"],
            "TARGET-LIMDATA": settings["data_lim_set"]["data_lim02"],
            "add_hps": True
        },      


    "t9999":
        {
            "template_file": "template0001.yaml",
            "DX-DATA-PAIR": settings["dx_sets"]["dx_set01"],
            "BASEMODEL-INFO": settings["models"]["test"]["test02"],
            "add_hps": False
        },         



    "t9998":
        {
            "template_file": "template0001.yaml",
            "DX-DATA-PAIR": settings["dx_sets"]["dx_set01"],
            "BASEMODEL-INFO": settings["models"]["test"]["test01"],
            "add_hps": False
        },  
}

base_id = {

    # MAE hps
    "t0001": 1,
    "t0002": 5,
    "t0003": 21,
    "t0004": 25,
    "t0005": 41,
    "t0006": 49,

    "t0007": 57,
    "t0008": 61,
    "t0009": 77,

    "t0010": 101,
    "t0011": 105,
    "t0012": 109,
    "t0013": 113,

    # Lim data.
    "t0014": 1001,
    "t0015": 1041,
    "t0016": 1081,
    "t0017": 1241, # -1400
    "t0018": 1401,
    "t0019": 1481,

    # Lim positive data.
    "t0020": 1561,
    "t0021": 1601,
    "t0022": 1641,
    "t0023": 1801,
    "t0024": 1961,
    "t0025": 2041, # -2120

    # Additional four models.
    "t0026": 3001, # -3016
    "t0027": 3017, # -3032
    "t0028": 3033,

    # Lim positive data.
    "t0029": 3101,
    "t0030": 3261,
    "t0031": 3421,
    "t0032": 3581, # -3740

    "t9998": 9909,
    "t9999": 9901,
}

def get_key_index(keys, target):
    if target in keys:
        key_index = list(keys).index(target)
    else:
        key_index = None
    return key_index

class ExpYamlPreparator:

    space = "  "

    def __init__(self, template_key: int):

        template_key = f"t{template_key:04d}"
        template_yaml = conditions[template_key]['template_file']
        template_file = f"./templates/exp_yamls/{template_yaml}"

        self.template = open(template_file).read()
        self.template_key = template_key

    def _insert_val(self, template, key, val):
        """
        Args:
            template (str):
            key (str): 
            val (str):         
        Returns:
            template
        """
        pattern_src = f'{key}:\n{self.space}param_type: fixed\n{self.space}param_val: XXX'
        pattern_dst= f'{key}:\n{self.space}param_type: fixed\n{self.space}param_val: {val}'
        template = template.replace(pattern_src, pattern_dst)
        return template

    def _make_yaml(self, template, keys, vals):
        """
        Args:
            template (str):
            keys (str): `TARGET-xxx`
            vals (str): 
        Returns: 
            template (str): 
        """
        for key, val in zip(keys, vals):
            template = template.replace(key, str(val))
        return template

    def _add_dx_and_data(self, template, dx_info): 
        """
        Args:
            template (str):
            dx_info (str): <dataset>-<dx>
        Returns: 
            template (str): 
        """        
        data_info = settings["dx_dataset_comb"][dx_info]
        assert template.find("TARGET-DS-POS") != -1
        assert template.find("TARGET-DS-NEG") != -1
        template = template.replace("TARGET-DS-POS", data_info["pos"])
        template = template.replace("TARGET-DS-NEG", data_info["neg"])
        template = template.replace("TARGET-DX", data_info["dx"])
        return template

    def _add_ft_target_and_arch(self, template, ft_target):
        """
        Args:
            template (str):
            ft_target (str): <arch>-<prior_training-info>
        Returns: 
            template (str): 
        """
        arch = ft_target.split("-")[0]
        if arch == "mae":
            arch = "mae_base"
        template = template.replace("MODEL-ARCH", arch)

        prior_training_info = ft_target.split("-")[1]
        if prior_training_info.find("none") != -1:
            template = template.replace("TARGET-PTMODEL", "null")
        else:
            template = template.replace("TARGET-PTMODEL", ft_target)
        return template

    def _add_model_info(self, template: str, modelname: str):
        """
        Args:
            template (str):
            modelname (str): <arch>-<prior_training-info>
        Returns:

        """
        arch = modelname.split("-")[0]
        if arch == "mae":
            return template
        
        additional_param_file = os.path.join(
            settings["paths"]["non_mae_templates"]["root"],
            settings["paths"]["non_mae_templates"]["models"][arch]
        )

        with open(additional_param_file, "r") as f:
            additional_params = f.read()
        
        template += "\n" + additional_params

        return template

    def _add_hps_info(self, param_data, modelname, dx):
        """
        Args:
            param_data (str): 
            modelname (str): modelname. e.g. <arch>-<pt>
            dx (str): e.g. `af`
        Returns:

        """
        arch = modelname.split("-")[0]
        model_key = f"{modelname}-{dx}"

        hps_file = os.path.join(
            settings["paths"]["hps"]["root"],
            settings["paths"]["hps"][arch][model_key],
            "ResultTableHPS.csv"
        )
        if not os.path.exists(hps_file):
            print(hps_file)
        assert os.path.exists(hps_file)

        hps_info = (
            "# Hyperparameter search result\n"
            "hps_result:\n"
            f"{self.space}param_type: fixed\n"
            f"{self.space}param_val: {hps_file}"
        )
        param_data = "\n" + hps_info + "\n" + param_data

        return param_data

    def _save_yaml(self, exp_yaml, yaml_id):
        """
        Args:

        Returns:

        """
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
        # print(savename)
        base = exp_yaml.find("modelname")
        start = exp_yaml.find("param_val:", base)
        end = exp_yaml.find("\n", start)
        modelname = exp_yaml[start+10:end].strip()
        print(savename, modelname)

    def _prepare_val_comb(self, template_conditions):
        """
        Args:

        Returns:

        """
        add_hps = template_conditions.pop("add_hps")
        target_dxs = template_conditions.pop("DX-DATA-PAIR")

        model_settings = template_conditions.pop("BASEMODEL-INFO")
        del template_conditions["template_file"]        

        link_dx = model_settings["link_dx"]
        models = model_settings["models"]

        dx_model_comb = []
        if link_dx:
            for dx, model in product(target_dxs, models):
                _model = model + dx.split("-")[1]
                dx_model_comb.append((dx, _model))
            dx_model_comb = tuple(dx_model_comb)
        else:
            dx_model_comb = list(product(target_dxs, models))

        keys = template_conditions.keys()
        val_combs = list(product(*template_conditions.values()))

        return keys, val_combs, dx_model_comb, add_hps

    def main(self):
        """
        Args:
            template_key (int): 
        Returns:
            None
        """

        template_conditions = conditions[self.template_key]
        keys, val_combs, dx_model_comb, add_hps =\
            self._prepare_val_comb(template_conditions)

        for idx, (val_comb, dx_model_comb)\
            in enumerate(product(val_combs, dx_model_comb)):
            # dx_model_comb: [<dataset>-<dx>, <arch>-<prior_training-info>]
            model_template = self._add_model_info(
                self.template, dx_model_comb[1])
            model_template = self._add_dx_and_data(
                model_template, dx_model_comb[0])
            model_template = self._add_ft_target_and_arch(
                model_template, dx_model_comb[1])

            exp_yaml = self._make_yaml(model_template, keys, val_comb)
            if add_hps:
                dx = dx_model_comb[0].split("-")[1]
                exp_yaml = self._add_hps_info(
                    exp_yaml, dx_model_comb[1], dx)
            # Save.
            yaml_id = base_id[self.template_key] + idx
            self._save_yaml(exp_yaml, yaml_id)

        print("Done")

if __name__ == "__main__":
    import sys
    template_key = sys.argv[1]

    prepartor = ExpYamlPreparator(int(template_key))
    prepartor.main()
