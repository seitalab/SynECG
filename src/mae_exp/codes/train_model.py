from typing import Dict

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from optuna import TrialPruned

from codes.models.model import prepare_clf_model
from codes.train_base import BaseTrainer
from codes.supports.monitor import Monitor, EarlyStopper
from codes.supports.utils import aggregator

class ModelTrainer(BaseTrainer):

    def set_model(self) -> None:
        """
        Args:
            None
        Returns:
            None
        """

        model = prepare_clf_model(self.args)
        model = model.to(self.args.device)

        self.model = model

    def set_pretrained_mae(
        self, 
        weight_file: str, 
        freeze: bool=False,
    ):
        """
        Set trained weight to model.
        Args:
            weight_file (str):
            freeze (bool): Freeze parameter or not.
            double_ft (bool): If True, meaning model was 
                1) MAE pretrained
                2) and then Finetuned with synthesized data classification.
        Returns:
            None
        """
        assert (self.model is not None)

        self.model.backbone.to("cpu")

        # Temporal solution.
        state_dict = dict(torch.load(weight_file, map_location="cpu")) # OrderedDict -> dict
        
        old_keys = list(state_dict.keys())
        for key in old_keys:
            new_key = key.replace("backbone.", "")
            
            state_dict[new_key] = state_dict.pop(key)
        self.model.backbone.load_state_dict(state_dict)

        if freeze:
            for p in self.model.backbone.parameters():
                p.requires_grad = False

        self.model.backbone.to(self.args.device)
        self.model.to(self.args.device)

    def set_lossfunc(self, weights=None) -> None:
        """
        Args:
            weights (Optional[np.ndarray]): 
        Returns:
            None
        """
        assert self.model is not None

        if weights is not None:
            weights = torch.Tensor(weights)
        
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=weights)
        self.loss_fn.to(self.args.device)

    def _train(self, loader) -> Dict:
        """
        Run train mode iteration.
        Args:
            loader:
        Returns:
            result_dict (Dict):
        """

        monitor = Monitor()
        self.model.train()

        for X, y in tqdm(loader):

            self.optimizer.zero_grad()
            X = X.to(self.args.device).float()
            y = y.to(self.args.device).float()
            pred_y = self.model(X)

            minibatch_loss = self.loss_fn(pred_y, y)

            minibatch_loss.backward()
            self.optimizer.step()

            monitor.store_loss(float(minibatch_loss) * len(X))
            monitor.store_num_data(len(X))
            monitor.store_result(y, pred_y)

        monitor.show_per_class_result()
        result_dict = {
            "score": monitor.macro_f1(), 
            "loss": monitor.average_loss(),
            "y_trues": monitor.ytrue_record,
            "y_preds": monitor.ypred_record
        }
        return result_dict
        
    def _evaluate(
        self, 
        loader, 
        dump_errors: bool=False
    ) -> Dict:
        """
        Args:
            loader :
        Returns:
            result_dict (Dict):
        """
        monitor = Monitor()
        self.model.eval()

        with torch.no_grad():

            for X, y in tqdm(loader):
                
                X = X.to(self.args.device).float()
                y = y.to(self.args.device).float()

                if self.args.neg_dataset.startswith("CPSC"):
                    pred_y = aggregator(self.model, X)
                else:
                    pred_y = self.model(X)

                minibatch_loss = self.loss_fn(pred_y, y)

                monitor.store_loss(float(minibatch_loss) * len(X))
                monitor.store_num_data(len(X))
                monitor.store_result(y, pred_y)
                if dump_errors:
                    monitor.store_input(X)

        monitor.show_per_class_result()

        if dump_errors:
            monitor.dump_errors(self.dump_loc, dump_type="fp")
            monitor.dump_errors(self.dump_loc, dump_type="fn")
            monitor.dump_errors(self.dump_loc, dump_type="tp")
            monitor.dump_errors(self.dump_loc, dump_type="tn")
        result_dict = {
            "score": monitor.macro_f1(),
            "loss": monitor.average_loss(),
            "y_trues": monitor.ytrue_record,
            "y_preds": monitor.ypred_record,
        }            
        return result_dict
    
    def run(self, train_loader, valid_loader) -> None:
        """
        Args:
            train_loader (Iterable): Dataloader for training data.
            valid_loader (Iterable): Dataloader for validation data.
            mode (str): definition of best (min or max).
        Returns:
            None
        """
        self.best = np.inf * self.flip_val # Sufficiently large or small
        if self.trial is None:
            early_stopper = EarlyStopper(
                mode=self.mode, patience=self.args.patience)

        for epoch in range(1, self.args.epochs + 1):
            print("-" * 80)
            print(f"Epoch: {epoch:03d}")
            train_result = self._train(train_loader)
            self.storer.store_epoch_result(
                epoch, train_result, is_eval=False)

            if epoch % self.args.eval_every == 0:
                eval_result = self._evaluate(valid_loader)
                self.storer.store_epoch_result(
                    epoch, eval_result, is_eval=True)

                if self.mode == "max":
                    monitor_target = eval_result["score"]
                else:
                    monitor_target = eval_result["loss"]

                # Use pruning if hyperparameter search with optuna.
                # Use early stopping if not hyperparameter search (= trial is None).
                if self.trial is not None:
                    self.trial.report(monitor_target, epoch)
                    if self.trial.should_prune():
                        raise TrialPruned()
                else:
                    if early_stopper.stop_training(monitor_target):
                        break

                self._update_best_result(monitor_target, eval_result)

            self.storer.store_logs()

    def _update_best_result(self, monitor_target, eval_result):
        """
        Args:

        Returns:
            None
        """
        
        if monitor_target * self.flip_val < self.best_val * self.flip_val:
            print(
                "Val metric improved:",
                f"{self.best_val:.4f} -> {monitor_target:.4f}"
            )
            self.best_val = monitor_target
            self.best_result = eval_result
            self.storer.save_model(self.model, monitor_target)
        else:
            print(
                "Val metric did not improve.",
                f"Current best {self.best_val:.4f}"
            )