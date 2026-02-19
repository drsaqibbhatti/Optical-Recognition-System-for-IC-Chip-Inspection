
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

    def encode(self, text, batch_max_length=25):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts
    
    
    
    
    # dbnet_loss_safe.py



def dice_loss(pred, gt, mask, eps=1e-6):
    pred = pred * mask
    gt   = gt   * mask
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum() + eps
    return 1 - (2 * inter + eps) / union


class DBLoss(nn.Module):
    def __init__(self, bce_scale=5.0, l1_scale=10.0, dice_scale=1.0):
        super().__init__()
        self.bce_scale = bce_scale
        self.l1_scale = l1_scale
        self.dice_scale = dice_scale

    def forward(self, pred, batch):
        shrink = pred["shrink"]
        binary = pred["binary"]
        thresh = pred["thresh"]

        gt = batch["gt"]
        mask = batch["mask"]
        thresh_map = batch["thresh_map"]
        thresh_mask = batch["thresh_mask"]

        shrink_logits = pred["shrink_logits"]
        bce = F.binary_cross_entropy_with_logits(shrink_logits, gt, reduction="none")
        bce = (bce * mask).sum() / (mask.sum() + 1e-6)

        d = dice_loss(binary, gt, mask)

        tm_sum = thresh_mask.sum()
        if tm_sum.item() > 0:
            l1 = (torch.abs(thresh - thresh_map) * thresh_mask).sum() / (tm_sum + 1e-6)
        else:
            l1 = torch.zeros((), device=shrink.device)

        loss = self.bce_scale * bce + self.dice_scale * d + self.l1_scale * l1
        metrics = {"loss": float(loss.item()), "bce": float(bce.item()), "dice": float(d.item()), "l1": float(l1.item())}
        return loss, metrics








def tensor_to_pil(img_t):
    # img_t: (C,H,W), normalized -1..1
    x = (img_t.detach().cpu() * 0.5 + 0.5).clamp(0, 1)
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    x = (x.permute(1, 2, 0).numpy() * 255).astype("uint8")
    return Image.fromarray(x)

def draw_polys(pil_img, polys, width=2):
    img = pil_img.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    for poly in polys:
        if not poly or len(poly) < 3:
            continue
        pts = [(float(p[0]), float(p[1])) for p in poly]
        draw.polygon(pts, outline=(255, 0, 0))
    return img

def show_gt_and_pred(img_t, polys, pred_dict, title=""):
    pil = tensor_to_pil(img_t)

    # GT overlay
    gt_img = draw_polys(pil, polys)

    # Pred map (use shrink or binary if training)
    shrink = pred_dict["shrink"][0, 0].detach().cpu().numpy()  # (H,W) in 0..1

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(gt_img)
    plt.title(f"GT {title}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(pil, cmap="gray")
    plt.imshow(shrink, alpha=0.5)  # heat overlay
    plt.title(f"Pred(shrink) {title}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


