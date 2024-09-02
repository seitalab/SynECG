import torch
import numpy as np
from tqdm import tqdm
from typing import Dict

from codes.train_base import BaseTrainer
from codes.supports.monitor import Monitor

class ModelPretrainer(BaseTrainer):
        
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

            for X, _ in tqdm(loader):
                
                X = X.to(self.args.device).float()

                minibatch_loss, _, _ = self.model(X)

                monitor.store_loss(float(minibatch_loss) * len(X))
                monitor.store_num_data(len(X))

        result_dict = {
            "score": 0,
            "loss": monitor.average_loss(),
        }            
        return result_dict

    def run(self, train_loader, valid_loader):
        """
        Args:
            train_loader (Iterable): Dataloader for training data.
            valid_loader (Iterable): Dataloader for validation data.
            mode (str): definition of best (min or max).
        Returns:
            None
        """
        self.best = np.inf * self.flip_val # Sufficiently large or small

        samples_per_epoch = len(train_loader.dataset)
        max_epochs = self.args.total_samples // samples_per_epoch + 1
        n_samples = 0
        n_samples_from_last_eval = 0
        n_samples_from_last_save = 0

        for epoch in tqdm(range(1, max_epochs + 1)):
            
            self.model.train()
            for X, _ in train_loader:

                self.optimizer.zero_grad()
                X = X.to(self.args.device).float()

                minibatch_loss, _, _ = self.model(X)

                minibatch_loss.backward()
                self.optimizer.step()

                n_samples += len(X)
                n_samples_from_last_eval += len(X)
                n_samples_from_last_save += len(X)

                # Evaluate training progress.
                if n_samples_from_last_eval > self.args.eval_every:
                    print(
                        f"{n_samples} / {self.args.total_samples} "
                        f"({n_samples / self.args.total_samples * 100:.3f})"
                    )
                    self.model.eval()
                    self._monitor_progress(
                        n_samples, train_loader, valid_loader)
                    self.storer.store_logs()
                    
                    n_samples_from_last_eval = 0
                    self.model.train()

                # Store model.
                if n_samples_from_last_save > self.args.save_model_every:
                    print(
                        "Saving interim model. "
                        f"{n_samples} / {self.args.total_samples} "
                        f"({n_samples / self.args.total_samples * 100:.3f})"
                    )
                    self.storer.save_model_interim(
                        self.model, n_samples)
                    n_samples_from_last_save = 0
                
                if n_samples > self.args.total_samples:
                    break

            if n_samples > self.args.total_samples:
                break


    def _monitor_progress(self, epoch, train_loader, valid_loader):

        train_set_result = self._evaluate(train_loader)
        self.storer.store_epoch_result(
            epoch, train_set_result, is_eval=False)

        val_set_result = self._evaluate(valid_loader)
        if self.mode == "max":
            monitor_target = val_set_result["score"]
        else:
            monitor_target = val_set_result["loss"]
        self.storer.store_epoch_result(
            epoch, val_set_result, is_eval=True)

        self._update_best_result(monitor_target, val_set_result)

    def _update_best_result(self, monitor_target, eval_result):
        """
        Args:

        Returns:
            None
        """
        
        if monitor_target * self.flip_val < self.best_val * self.flip_val:
            print(f"Val metric improved {self.best_val:.4f} -> {monitor_target:.4f}")
            self.best_val = monitor_target
            self.best_result = eval_result
            self.storer.save_model(self.model, monitor_target)
        else:
            print(f"Val metric did not improve. Current best {self.best_val:.4f}")