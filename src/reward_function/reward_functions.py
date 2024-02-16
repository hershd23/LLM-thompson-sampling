"""
Contains the reward functions for our LLM matching bandit

Code is based on the following repository:
https://github.com/pchunduri6/rag-opt/blob/main/rag.py#L375
"""

from openai_utils import llm_call

class BaseReward():
    def __init__(self, system_output:str, evaluation_set_answer: str):
        self.system_output = system_output
        self.evaluation_set_answer = evaluation_set_answer

class WordMatch(BaseReward):
    def eval(self):
        evaluation_set_terms = [self.evaluation_set_answer.lower()]
        evaluation_set_terms += self.evaluation_set_answer.lower().split(" ")

        self.system_output = self.system_output.lower()
        answer_found = 1
        for term in evaluation_set_terms:
            if self.system_output.find(term) == -1:
                answer_found = 0
                break

        return answer_found
    
class LLMEval(BaseReward):
    def __init__(self, question:str, system_output:str, evaluation_set_answer: str, model:str="gpt-3.5-turbo-0613"):
        super().__init__(system_output, evaluation_set_answer)
        self.model = model

    def eval(self):
        EVAL_prompt = f"""You are an assistant for evaluating question-answering systems.
            You are given a question and two answers.
            The first answer is generated by a question-answering system.
            The second answer is the correct answer.
            Output 1 if the first answer is correct, and 0 otherwise.
            Your response should only be 1 or 0.
            Question: {self.question}
            System answer: {self.system_output}
            Ground truth answer: {self.evaluation_set_answer}
            Output:
        """
        response, cost = llm_call(model="gpt-4", user_prompt=EVAL_prompt)
        answer_found = int(response.choices[0].message.content)

        return answer_found