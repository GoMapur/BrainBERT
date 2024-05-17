import torch
from .base_criterion import BaseCriterion
from torch import nn
from criterions import register_criterion
import numpy as np

@register_criterion("finetune_criterion")
class FinetuneCriterion(BaseCriterion):
    def __init__(self):
        super(FinetuneCriterion, self).__init__()
        pass

    def build_criterion(self, cfg):
        self.cfg = cfg
        self.softmax = nn.Softmax(dim=1)
        self.loss_fn = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, model, batch, device, return_predicts=False, debug=False, debug_name=None):
        inputs = batch["input"].to(device) #potentially don't move to device if dataparallel
        pad_mask = batch["pad_mask"].to(device)

        output = model.forward(inputs, pad_mask)
        labels = torch.LongTensor(batch["labels"]).to(output.device)
        
        # return counts of each class
        # print("------------------")
        # print(np.unique(np.argmax(output.cpu().detach().numpy(), axis=1), return_counts=True))
        # print(np.unique(labels.cpu().numpy(), return_counts=True))
        # print("------------------")

        # print labels and predicts
        # print("training ------------------")
        # print(f"labels: {labels[:10].detach().cpu().numpy()}")
        # print(f"predicts: {output[:10].detach().cpu().numpy()}")
        # print(f"predicts: {self.softmax(output[:10]).detach().cpu().numpy()}")
        # print(f"predicts: {np.argmax(self.softmax(output[:10]).detach().cpu().numpy(), axis=1)}")
        # print("------------------")

        # output = output.squeeze(-1)
        loss = self.loss_fn(output, labels)
        images = {"wav": batch["wavs"][0],
                  "wav_label": batch["labels"][0]}
        if return_predicts:
            predicts = self.softmax(output).detach().cpu().numpy()
            logging_output = {"loss": loss.item(),
                              "predicts": predicts,
                              "images": images}
        else:
            logging_output = {"loss": loss.item(),
                              "images": images}
            
        
        if debug:
            # print(inputs)
            # print(pad_mask)
            # print(output)
            # print(labels)
            # print("-----------------")
            # save all to npz file
            # also save model.prediction_head to npz file
            dump_prediction_head = model.model["prediction_head"].state_dict()
            dump_model = model.state_dict()

            dump_prediction_head_np_arrays = {k: v.cpu().numpy() for k, v in dump_prediction_head.items()}
            np.savez_compressed(f"{debug_name}_prediction_head", **dump_prediction_head_np_arrays)
            dump_model_np_arrays = {k: v.cpu().numpy() for k, v in dump_model.items()}
            np.savez_compressed(f"{debug_name}_model", **dump_model_np_arrays)
            
            # generate an random input with fixed seed to test output
            
            # set torch random seed
            torch.manual_seed(42)
            
            rand_input = torch.rand_like(inputs, device=inputs.device, requires_grad=False)
            rand_output = model.forward(rand_input, pad_mask)
            rand_labels = torch.LongTensor([0]*rand_output.shape[0]).to(rand_output.device)
            rand_loss = self.loss_fn(rand_output, rand_labels)
            
            np.savez_compressed(debug_name, inputs=inputs.cpu().numpy(), pad_mask=pad_mask.cpu().numpy(), output=output.cpu().detach().numpy(), labels=labels.cpu().numpy(), loss=loss.detach().cpu().numpy(), rand_input=rand_input.cpu().numpy(), rand_output=rand_output.cpu().detach().numpy(), rand_labels=rand_labels.cpu().numpy(), rand_loss=rand_loss.detach().cpu().numpy())
            
        return loss, logging_output
