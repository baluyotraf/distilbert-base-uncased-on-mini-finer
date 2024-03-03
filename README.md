# DistilBERT on Finer-139

The goal of the project is to tune the DistilBERT model for a token 
classification problem on the `nlpaueb/finer-139` dataset. 

## Model

The model uses a fined tuned of [DistilBERT], a smaller and faster version of 
Google's BERT model. For this project, the model was used for a token
classification problem, this means that the model shall return the same number 
of labels as the input.

## Dataset

The dataset used is the [Finer-139] dataset which is based on annual and 
quarterly reports of different companies.

As a starting concept, the model is only fined tuned on the labels with the 
smaller counts. These labels are listed below:

*   CashAndCashEquivalentsFairValueDisclosure
*   RevenueFromContractWithCustomerIncludingAssessedTax
*   InterestExpense
*   EmployeeServiceShareBasedCompensationTaxBenefitFromCompensationExpense

## Performance

The model is a bit weak to getting the `InterestExpenses` and 
`RevenueFromContractWithCustomerIncludingAssessedTax` compared to the other 
selected labels. On the other hand, it has very strong performance on the
remaining labels, which improved the overall score.

Below is the summary of the performance on all the target labels.

| Label                                                                  | Precision | Recall | F1    |
|------------------------------------------------------------------------|-----------|--------|-------|
| CashAndCashEquivalentsFairValueDisclosure                              |     0.100 |  1.000 | 1.000 |
| EmployeeServiceShareBasedCompensationTaxBenefitFromCompensationExpense |     0.938 |  1.000 | 1.000 |
| InterestExpense                                                        |     0.966 |  0.935 | 0.951 |
| RevenueFromContractWithCustomerIncludingAssessedTax                    |     0.942 |  0.958 | 0.950 |
| Overall                                                                |     0.969 |  0.977 | 0.973 |


## Usage

Example codes to fine-tune, and use both the Hugging Face and ONNX versions
of the model can be found by checking the [Source]. Samples sentences can also
be fed directly on the Hugging Face [Model] page.

### Hugging Face

To use the model from the Hugging Face repository, simply load the model by
using the [Model] repository.

```python
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained(HUGGING_FACE_REPOSITORY)
model = AutoModelForTokenClassification.from_pretrained(HUGGING_FACE_REPOSITORY)
model.eval()

with torch.no_grad():
    output = model(**dict(tokenizer("I paid $10 in interest", return_tensors="pt")))
```


### ONNX

An exmple to export the ONNX model from the Hugging Face model can be seen
from [Source]. For more details, feel free to check the Hugging Face 
documentation on [Export to ONNX].

Below is an example python code to run the ONNX version of the model. 

```python
from transformers import AutoTokenizer
from onnxruntime import InferenceSession

tokenizer = AutoTokenizer.from_pretrained(ONNX_OUTPUT_PATH)
ort_session = InferenceSession(f"{ONNX_OUTPUT_PATH}/model.onnx")

ort_output = ort_session.run(
    output_names=["logits"],
    input_feed=dict(tokenizer(["I paid $10 in interest"])),
)
```


[DistilBERT]: https://huggingface.co/docs/transformers/model_doc/distilbert
[Finer-139]: https://huggingface.co/datasets/nlpaueb/finer-139
[Model]: https://huggingface.co/baluyotraf/distilbert-base-uncased-on-mini-finer/tree/main
[Source]: https://github.com/baluyotraf/distilbert-base-uncased-on-mini-finer
[Export to ONNX]: https://huggingface.co/docs/transformers/serialization
