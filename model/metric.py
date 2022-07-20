import torchmetrics
from pytorch_accelerated.callbacks import TrainerCallback

class accuracy(TrainerCallback):
    def __init__(self):
        self.accuracy = torchmetrics.Accuracy()

    def on_training_run_start(self, trainer, **kwargs):
        self.accuracy.to(trainer.device)

    def on_evaluation_run_start(self, trainer, **kwargs):
        self.accuracy.to(trainer.device)

    def on_eval_step_end(self, trainer, batch, batch_output, **kwargs):
        preds = batch_output["model_outputs"].argmax(dim=-1)
        self.accuracy.update(preds, batch[1])

    def on_eval_epoch_end(self, trainer, **kwargs):
        trainer.run_history.update_metric("accuracy", self.accuracy.compute().item())
        self.accuracy.reset()

class iou(TrainerCallback):
    def __init__(self):
        self.iou = torchmetrics.JaccardIndex()

    def on_training_run_start(self, trainer, **kwargs):
        self.iou.to(trainer.device)

    def on_train_step_end(self, trainer, batch, batch_output, **kwargs):
        preds = batch_output["model_outputs"]
        self.accuracy.update(preds, batch[1])

    def on_eval_epoch_end(self, trainer, **kwargs):
        trainer.run_history.update_metric("accuracy", self.accuracy.compute().item())
        self.accuracy.reset()


def iou(output, target):
    with torch.no_grad():
        SMOOTH = 1.0e-6
        # You can comment out this line if you are passing tensors of equal shape
        # But if you are passing output from UNet or something it will most probably
        # be with the BATCH x 1 x H x W shape
        output = output[:, 2, :, :] > 0.5
        target = target[:, 2, :, :] > 0.5
        
        intersection = (output & target).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
        union = (output | target).float().sum((1, 2))         # Will be zzero if both are 0
        
        iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
        
        thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return thresholded.mean()  # Or thresholded.mean() if you are interested in average across the batch
