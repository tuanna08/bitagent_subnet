# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 RogueTensor

# ... (phần mã nguồn khác)

import bitagent
import transformers
from common.base.miner import BaseMinerNeuron
from transformers import T5Tokenizer, T5ForConditionalGeneration
from bitagent.miners.context_util import get_relevant_context_and_citations_from_synapse
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Thêm FastAPI app
app = FastAPI()

def miner_init(self, config=None):
    transformers.logging.set_verbosity_error()
    self.tokenizer = AutoTokenizer.from_pretrained("Open-Orca/Mistral-7B-OpenOrca", legacy=False)
    self.model = AutoModelForCausalLM.from_pretrained("Open-Orca/Mistral-7B-OpenOrca", device_map=self.device)

    def llm(input_text):
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
        outputs = self.model.generate(input_ids, max_length=60)
        result = self.tokenizer.decode(outputs[0])
        # response is typically: <pad> text</s>
        result = result.replace("<pad>","").replace("</s>","").strip()
        return result

    self.llm = llm

def miner_process(self, synapse: bitagent.protocol.QnATask) -> bitagent.protocol.QnATask:
    if not synapse.urls and not synapse.datas:
        context = ""
        citations = []
    else:
        context, citations = get_relevant_context_and_citations_from_synapse(synapse)

    query_text = f"Please provide the user with an answer to their question: {synapse.prompt}.\n\n Response: "
    if context:
        query_text = f"Given the following CONTEXT:\n\n{context}\n\n{query_text}"

    llm_response = self.llm(query_text)

    synapse.response["response"] = llm_response
    synapse.response["citations"] = citations

    return synapse

# API endpoint để gọi từ client
@app.post("/run_llm_api")
async def run_llm_api(input_text: str):
    # Thực hiện cuộc gọi LLM và trả về kết quả
    llm_response = miner.llm(input_text)
    return JSONResponse(content={"result": llm_response})

# Thêm dòng mã để khởi chạy FastAPI
if __name__ == "__main__":
    import uvicorn

    # Địa chỉ và cổng sẽ là 0.0.0.0:8080
    uvicorn.run(app, host="127.0.0.1", port=8080)
