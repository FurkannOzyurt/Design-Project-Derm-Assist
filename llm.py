from ollama import Client                     # tiny wrapper around the REST API
import torch, json

class MedicalLLMHelper:
    """
    Uses the local Ollama server (default http://localhost:11434)
    with the model name you created above:  biomedllama-q8
    """

    def __init__(
        self,
        model: str = "hf.co/unsloth/medgemma-4b-it-GGUF:Q8_K_XL",
        host: str = "http://localhost:11434",
        temperature: float = 0.6,
        top_p: float = 0.9,
    ):
        self.client = Client(host=host)
        self.model = model
        self.gen_opts = {"temperature": temperature, "top_p": top_p}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------- internal helper -------------
    def _chat(self, messages: list[dict]) -> str:
        """
        One-shot chat call.  `messages` must be a list of
        {"role": "system"/"user"/"assistant", "content": "..."} dicts.
        """
        resp = self.client.chat(
            model=self.model,
            messages=messages,
            options=self.gen_opts          # same fields as the CLI --temperature ...
        )
        return resp["message"]["content"]

    # ---------- 1) Question reformulation ----------
    def reformulate_question(self, raw_question: str, disease: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful medical assistant.\n"
                    f"The patient has been diagnosed with **{disease}**.\n"
                    "Your task is to rewrite the user's medical question in a clearer, more formal way, using professional medical language if appropriate.\n"
                    "Respond ONLY with the rewritten question. Do NOT explain, rephrase, or add any extra text. Write only ONE sentence."
                ),
            },
            {"role": "user", "content": raw_question},
        ]
        return self._chat(messages)

    # ---------- 2) Final answer ----------
    def generate_answer(self, disease_name: str, user_q: str, rag_out: dict) -> str:
        if disease_name == "Normal_Image":
            return (
                "Your image does not appear to show any concerning skin disease. "
                "Everything looks normal, so there is nothing to worry about. "
                "If you have any other questions or notice new changes, feel free to let me know!"
            )

        # 1️⃣  Diyaloğu sürdüren sistem yönergesi
        system_prompt = f"""
            You are DermAI, a friendly, medically accurate dermatology assistant.

            When replying:
            1. **Answer clearly first** - give a concise, evidence-based reply to the user's question.
            2. **Add helpful context** - explain the key points (cause, symptoms, care tips, or next steps).
            3. **Show empathy** - use reassuring, supportive language.
            4. **Invite follow-up** - end with an open question (e.g., “Would you like to know home-care tips?”).
            5. **Keep it brief** (≤ 150 words) unless the user asks for more detail.

            ---
            **Disease:** {disease_name}
            **Description:** {rag_out['description']}

            **Relevant Q&A Pairs**
            {chr(10).join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in rag_out['matched_qa_pairs']])}
            """.strip()

        # 2️⃣  Mesaj listesi (kullanıcı sorusunu ve varsa rafine hâlini ilet)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"User Question: {user_q}\n"
                    f"(refined: {rag_out['refined_question']})"
                ),
            },
        ]

        # 3️⃣  Model çağrısı (kendi LLM wrapper’ına göre uyarlarsın)
        response = self._chat(messages)
        return response

