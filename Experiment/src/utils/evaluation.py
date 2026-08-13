from collections import Counter


def token_f1_score(prediction: str, gold: str) -> float:
    prediction_tokens = prediction.lower().split()
    gold_tokens = gold.lower().split()
    if not prediction_tokens or not gold_tokens:
        return 0.0

    common_count = sum(
        (Counter(prediction_tokens) & Counter(gold_tokens)).values()
    )
    if common_count == 0:
        return 0.0

    precision = common_count / len(prediction_tokens)
    recall = common_count / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

class Evaluation:
    def __init__(self, llm_ans, ds):
        self.llm_ans = llm_ans
        self.ds = ds
        self.normalize_text()
        self.validate_inputs()
        self.eval1_success = 0
        self.eval1_fail = 0
        self.token_f1_scores = []
        print("Evaluation Process Initialized")

    def normalize_text(self):
        self.llm_ans = [ans.lower() for ans in self.llm_ans]
        self.ds["answer"] = [ans.lower() for ans in self.ds["answer"]]
    def validate_inputs(self):
        if not self.llm_ans:
            raise ValueError("llm_ans must contain at least one answer.")
        if len(self.llm_ans) != len(self.ds["answer"]):
            raise ValueError("llm_ans and ds['answer'] must have the same length.")
        
    def evaluate_1(self):
        for i, ans in enumerate(self.llm_ans):
            if self.ds["answer"][i] in ans:
                self.eval1_success += 1
            else:
                self.eval1_fail += 1

        self.token_f1_scores = []
        for i, prediction in enumerate(self.llm_ans):
            gold = self.ds["answer"][i]
            self.token_f1_scores.append(token_f1_score(prediction, gold))

    def get_results(self):
        if self.eval1_success == 0 and self.eval1_fail == 0:
            self.evaluate_1()
        if not self.token_f1_scores:
            self.evaluate_2()

        return {
            "exact_match": self.eval1_success / len(self.llm_ans),
            "token_f1": sum(self.token_f1_scores) / len(self.token_f1_scores),
        }
