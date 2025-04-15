import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI, Query
from pydantic import BaseModel

DEFAULT_RISK_THRESHOLD = 0.6
NUM_TOP_TOKEN_LOGPROBS = 10 
MAX_TOKENS_TO_GENERATE = 20    

class GuardianModule():
    def __init__(self):
        self.nlogprobs = NUM_TOP_TOKEN_LOGPROBS
        model_path = "ibm-granite/granite-guardian-3.0-2b"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16
        ).to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)


        self.risks = [
            "harm",
            "social_bias",
            "violence",
            "profanity",
            "sexual_content",
            "unethical_behavior"
        ]

        self.target_risks = [ "violence", "unethical_behavior", "sexual_content"]

    def parse_output(self, output):
        """
        Adapted from https://huggingface.co/ibm-granite/granite-guardian-3.1-2b Quickstart sample
        """   
        label, prob_of_risk = None, None

        list_index_logprobs_i = [torch.topk(token_i, k=self.nlogprobs, largest=True, sorted=True)
                                for token_i in list(output.scores)[:-1]]
        if list_index_logprobs_i is not None:
            prob = self.get_probabilities(list_index_logprobs_i)
            prob_of_risk = prob[1]

        return prob_of_risk.item()

    def get_probabilities(self, logprobs):
        """
        Adapted from https://huggingface.co/ibm-granite/granite-guardian-3.1-2b Quickstart sample
        """    
        safe_token_prob = 1e-50
        unsafe_token_prob = 1e-50
        for gen_token_i in logprobs:
            for logprob, index in zip(gen_token_i.values.tolist()[0], gen_token_i.indices.tolist()[0]):
                decoded_token = self.tokenizer.convert_ids_to_tokens(index)
                if decoded_token.strip().lower() == "no":
                    safe_token_prob += math.exp(logprob)
                if decoded_token.strip().lower() == "yes":
                    unsafe_token_prob += math.exp(logprob)

        probabilities = torch.softmax(
            torch.tensor([math.log(safe_token_prob), math.log(unsafe_token_prob)]), dim=0
        )

        return probabilities

    def get_risk_label(self, user_text: str, risk_thresh: float):
        """
        Gets the risk label corresponding to the user text or returns None
        """
        messages = [{"role": "user", "content": user_text}]
        risk_scores = {}

        for risk in self.risks:
            guardian_config = {"risk_name": risk}

            input_ids = self.tokenizer.apply_chat_template(
                messages,
                guardian_config=guardian_config,
                add_generation_prompt=False,
                return_tensors="pt"
            ).to(self.model.device)

            input_len = input_ids.shape[1]
            self.model.eval()

            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    do_sample=False,
                    max_new_tokens=MAX_TOKENS_TO_GENERATE,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            prob_of_risk = self.parse_output(output)
            risk_scores[risk] = float(prob_of_risk)

        filtered_scores = {risk: risk_scores[risk] for risk in self.target_risks}

        max_filtered_risk = max(filtered_scores, key=filtered_scores.get)
        max_filtered_score = filtered_scores[max_filtered_risk]

        if max_filtered_score > risk_thresh:
            label = max_filtered_risk
        elif (max(risk_scores[risk] for risk in self.risks) > risk_thresh ):
            label = "toxicity"
        else:
            label = None

        return label    
    
app = FastAPI()
gm = GuardianModule()

class RiskRequest(BaseModel):
    text: str
    threshold: float = DEFAULT_RISK_THRESHOLD

@app.post("/label")
def get_label(req: RiskRequest):
    label = gm.get_risk_label(req.text, req.threshold)
    return {"label": label}