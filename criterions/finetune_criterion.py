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

        if debug:
            # print(inputs)
            # print(pad_mask)
            # print(output)
            # print(labels)
            # print("-----------------")
            # save all to npz file
            np.savez_compressed(debug_name, inputs=inputs.cpu().numpy(), pad_mask=pad_mask.cpu().numpy(), output=output.cpu().detach().numpy(), labels=labels.cpu().numpy())

        # return counts of each class
        # print("------------------")
        # print(np.unique(np.argmax(output.cpu().detach().numpy(), axis=1), return_counts=True))
        # print(np.unique(labels.cpu().numpy(), return_counts=True))
        # print("------------------")

        # print labels and predicts
        print("training ------------------")
        print(f"labels: {labels[:10].detach().cpu().numpy()}")
        print(f"predicts: {output[:10].detach().cpu().numpy()}")
        print(f"predicts: {self.softmax(output[:10]).detach().cpu().numpy()}")
        print(f"predicts: {np.argmax(self.softmax(output[:10]).detach().cpu().numpy(), axis=1)}")
        print("------------------")

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
        return loss, logging_output
