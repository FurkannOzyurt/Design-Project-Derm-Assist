from sentence_transformers import SentenceTransformer, util
import torch, json
from pathlib import Path
from src.llm import MedicalLLMHelper
from huggingface_hub import login
login(token=" ")

class RetrievalAugmentedGeneration:
    def __init__(
        self,
        helper: MedicalLLMHelper,
        qa_json_path: str = "dataset/dermatology_qa.json",
        encoder_name: str = "dmis-lab/biobert-base-cased-v1.1",
        device: str = "cuda",
        hybrid_alpha: float = 0.7,
    ):
        self.device = torch.device(device)
        self.helper = helper
        self.encoder = SentenceTransformer(encoder_name, device=device)

        with Path(qa_json_path).open() as f:
            self.derm_data = json.load(f)

        self.alpha = hybrid_alpha

    # ---------- retrieval ----------
    def retrieve(self, disease: str, user_q: str, top_k=5, thresh=0.75) -> dict:
        if disease not in self.derm_data:
            return

        entry = self.derm_data[disease]
        qas, desc = entry["qa_pairs"], entry["description"]

        refined_q = self.helper.reformulate_question(user_q, disease)

        q_texts = [qa["question"] for qa in qas]
        a_texts = [qa["answer"] for qa in qas]

        q_emb = self.encoder.encode(q_texts, convert_to_tensor=True)
        a_emb = self.encoder.encode(a_texts, convert_to_tensor=True)
        user_emb = self.encoder.encode(refined_q, convert_to_tensor=True)

        q_sim = util.pytorch_cos_sim(user_emb, q_emb)[0]
        a_sim = util.pytorch_cos_sim(user_emb, a_emb)[0]
        hybrid = self.alpha * q_sim + (1 - self.alpha) * a_sim

        top_scores, top_idx = torch.topk(hybrid, k=len(q_texts))
        filt = [
            qas[i]
            for i, sc in zip(top_idx, top_scores)
            if sc.item() >= thresh
        ][:top_k]

        return {
            "refined_question": refined_q,
            "description": desc,
            "matched_qa_pairs": filt,
        }