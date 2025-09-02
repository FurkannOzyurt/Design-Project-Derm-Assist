import torch, timm
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from typing import List, Union, Optional


class ConvNeXtTinyClassifier:
    def __init__(
        self,
        weights_path: str,
        class_names: List[str],
        transform: Optional[transforms.Compose] = None,
        device: Union[str, torch.device] = "cuda",   # 🖥️→ default GPU
        drop_path_rate: float = 0.2,
    ):
        if class_names is None:
            raise ValueError("class_names listesi gereklidir.")

        self.class_names = class_names
        self.device = torch.device(device)

        # ---------- model ----------
        self.model = timm.create_model(
            "convnext_tiny",
            pretrained=True,
            num_classes=len(class_names),
            drop_path_rate=drop_path_rate,
        )
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.model.to(self.device).eval()

        # ---------- transform ----------
        default_tf = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )
        self.transform = transform or default_tf

    def _prep(self, img: Image.Image) -> torch.Tensor:
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.transform(img).unsqueeze(0)

    @torch.no_grad()
    def predict(
        self,
        imgs: Union[str, Image.Image, List[Union[str, Image.Image]]],
        top_k: int = 1,
        return_probs: bool = False,
    ):
        if not isinstance(imgs, (list, tuple)):
            imgs = [imgs]

        batch = []
        for im in imgs:
            img = Image.open(im) if isinstance(im, (str, bytes)) else im
            batch.append(self._prep(img))
        batch_tensor = torch.cat(batch).to(self.device)

        logits = self.model(batch_tensor)
        probs = torch.softmax(logits, dim=1)

        top_probs, top_idx = torch.topk(probs, k=top_k, dim=1)
        top_idx, top_probs = top_idx.cpu(), top_probs.cpu()

        results = []
        for idxs, prbs in zip(top_idx, top_probs):
            if return_probs:
                results.append(
                    [(self.class_names[i], float(p)) for i, p in zip(idxs, prbs)]
                )
            else:
                results.append(self.class_names[idxs[0]])

        return results[0] if len(results) == 1 else results
