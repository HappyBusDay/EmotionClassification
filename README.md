# Emotion Recognition in Conversation
# 월간 데이콘 발화자의 감정인식 AI 경진대회
<img width="1000" img height="210" alt="Dacon" src="https://user-images.githubusercontent.com/113493692/206966548-bb71d381-a828-4a6f-8806-d63be4e37419.png">

#### 데이터셋 다운로드 : [월간 데이콘 발화자의 감정인식 AI 경진대회](https://dacon.io/competitions/official/236027/data#)

대회 참여 기간 : 2022.11.21. ~ 2022.12.12.

### 최종 순위 : Public 3등

<img width="1000" alt="image" src="https://user-images.githubusercontent.com/113493692/206967116-f16f8827-ad37-4a32-ad29-24cdbee13b7d.png">


---

# Code

<table>
    <thead>
        <tr>
            <th>목록</th>
            <th>파일명</th>
            <th>설명</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=1>SAM Optimizer</td>
            <td>
                <a href="https://github.com/HappyBusDay/EmotionClassification/blob/main/SAM_Optimizer.ipynb">SAM_Optimizer.ipynb</a>
            </td>
            <td> SAM Optimizer </td>
        </tr>
            <td>Training</td>
            <td>
                <a href="https://github.com/HappyBusDay/EmotionClassification/blob/main/Train.ipynb">Train.ipynb</a>     
            <td> Train </td>
        </tr>
        <tr>
            <td>Inference</td>
            <td>
                <a href="https://github.com/HappyBusDay/EmotionClassification/blob/main/Inference.ipynb">Inference.ipynb</a>     
            <td> Inference </td>
        </tr>        
        <tr>
            <td rowspan=2>Model Ensemble</td>       
            <td>
                <a href="https://github.com/HappyBusDay/EmotionClassification/blob/main/HardVoting.ipynb">HardVoting.ipynb</a>
            <td> Hard Voting</td>
        </tr>
        <tr>
            <td>
                <a href="https://github.com/HappyBusDay/EmotionClassification/blob/main/SoftVoting.ipynb">SoftVoting.ipynb</a>
            <td>Soft Voting</td>
        </tr>
        
   </tbody>
</table>

---

# 가. 개발 환경

     Google Colab Pro
     
     huggingface-hub==0.10.1
     datasets==2.6.1
     tokenizers==0.13.2
     torch==1.12.1+cu113
     torchvision==0.13.1+cu113
     transformers==4.24.0
     tqdm==4.64.1
     scikit-learn==1.0.2
     sentencepiece==0.1.97


---

# 나. 데이터 예시 
 출처 : Dacon, 월간 데이콘 발화자의 감정인식 AI 경진대회 

    {"ID": "TRAIN-0002", "Utterance": "That I did. That I did.", "Speaker": "Chandler", "Dialogue_ID": "0", "Target": "neutral"}
    {"ID": "TRAIN-0007", "Utterance": "But there’ll be perhaps 30 people under you so you can dump a certain amount on them.", "Speaker": "The Interviewer", "Dialogue_ID": "0", "Target": "neutral"}
    {"ID": "TRAIN-0017", "Utterance": "No, I-I-I-I don't, I actually don't know", "Speaker": "Rachel", "Dialogue_ID": "1", "Target": "fear"}

---


# 라. 주요 소스 코드

- ## Model Load From Hugging Face
   
   
    <table>
    <thead>
        <tr>
            <th>Model</th>
            <th>Pre-Trained Dataset</th>
            <th>링크(HuggingFace)</td>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td> DistilBERT</td>
            <td>                 
                <a href="https://huggingface.co/datasets/viewer/?dataset=emotion">Twitter-Sentiment-Analysis</a></td>
            <td>
                <a href="https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion">bhadresh-savani/distilbert-base-uncased-emotion</a>
        </tr>
        <tr>
            <td> BERT</td>            
            <td>                 
                <a href="https://huggingface.co/datasets/viewer/?dataset=emotion">Twitter-Sentiment-Analysis</a></td>
            <td>
                <a href="https://huggingface.co/bhadresh-savani/bert-base-uncased-emotion">bhadresh-savani/bert-base-uncased-emotion</a>
        </tr>
        <tr>
            <td> EmoBERTa-base</td>
            <td>                
                <a href="https://github.com/tae898/multimodal-datasets/tree/a36101638a8121b422ce4a2a17746b25f23335b8">multimodal-datasets</a></td>
            <td>
                <a href="https://huggingface.co/tae898/emoberta-base">tae898/emoberta-base</a>
        </tr>
        <tr>
            <td> EmoBERTa-large</td>
            <td>                
                <a href="https://github.com/tae898/multimodal-datasets/tree/a36101638a8121b422ce4a2a17746b25f23335b8">multimodal-datasets</a></td>
            <td>
                <a href="https://huggingface.co/tae898/emoberta-large">tae898/emoberta-large</a>
        </tr>
    </tbody>
    </table>
      
    
   ```c
    # HuggingFace에서 불러오기
    from transformers import AutoTokenizer, AutoModel
    base_model = "HuggingFace주소"

    Model = AutoModel.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
   ```


- ## Add Layer
   
   > [AddLayer.ipynb 참조](https://github.com/HappyBusDay/EmotionClassification/blob/main/AddLayer.ipynb)

     ```c

    class BaseModel_AddLayer(nn.Module):
        def init(self, dropout=0.5, num_classes=len(le.classes_)) :
            super(BaseModel_AddLayer, self).__init__()
            self.dropout = nn.Dropout(dropout)
            self.linear1 = nn.Linear(768, 384)
            self.linear2 = nn.Linear(384, num_classes)
            self.gelu = nn.GELU()
            
        def forward(self, input_id, mask) :
            _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
            dropout_output = self.dropout(pooled_output)
            linear_output = self.linear1(dropout_output)
            linear_output = self.linear2(linear_output)
            final_layer = self.gelu(linear_output)
        
        return final_layer
     ```



- ## SAM Optimizer

   > [SAM_Optimizer.ipynb 참조](https://github.com/HappyBusDay/EmotionClassification/blob/main/SAM_Optimizer.ipynb)
   
    ```c

    class SAM(torch.optim.Optimizer):
        def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
            assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

            defaults = dict(rho=rho, **kwargs)
            super(SAM, self).__init__(params, defaults)

            self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
            self.param_groups = self.base_optimizer.param_groups

        @torch.no_grad()
        def first_step(self, zero_grad=False):
            grad_norm = self._grad_norm()
            for group in self.param_groups:
                scale = group["rho"] / (grad_norm + 1e-12)

                for p in group["params"]:
                    if p.grad is None: continue
                    e_w = p.grad * scale.to(p)
                    p.add_(e_w)  # climb to the local maximum "w + e(w)"
                    self.state[p]["e_w"] = e_w

            if zero_grad: self.zero_grad()

        ...    

        자세한 코드는 code/test.ipynb 참조

        return norm
    ```
    
    
- ## Ensemble
    - #### Hard Voting
        > [HardVoting.ipynb 참조](https://github.com/HappyBusDay/EmotionClassification/blob/main/HardVoting.ipynb)
        
    ```c

    def HardVoting(inference1, inference2, inference3) :

        dic_tmp = {'ID' : [], 'Target' : []}

        for i in range(len(Test)) :
            tmp = []
            tmp.append(inference1['Target'][i])
            tmp.append(inference2['Target'][i])
            tmp.append(inference3['Target'][i])

            dic_tmp['ID'].append(inference1['ID'][i])
            dic_tmp['Target'].append(Counter(tmp).most_common(n=1)[0][0])

        pd.DataFrame(dic_tmp).to_csv('Ensemble.csv', encoding = 'utf-8', index = False)

        return pd.DataFrame(dic_tmp)
    ```
        
    - #### Soft Voting        
        > [SoftVoting.ipynb 참조](https://github.com/HappyBusDay/EmotionClassification/blob/main/SoftVoting.ipynb)

    ```c
    def SoftVoting(model1, model2, model3, model4, test_loader, device):

        model1.to(device)
        model1.eval()

        model2.to(device)
        model2.eval()

        model3.to(device)
        model3.eval()

        model4.to(device)
        model4.eval()

        test_predict = []

        for input_ids, attention_mask in tqdm(test_loader):

            input_id = input_ids.to(device)
            mask = attention_mask.to(device)

            y_pred1 = model1(input_id, mask)
            y_pred2 = model2(input_id, mask)
            y_pred3 = model3(input_id, mask)
            y_pred4 = model4(input_id, mask)

            test_predict += ((y_pred1 + y_pred2 + y_pred3 + y_pred4 )/4).argmax(1).detach().cpu().numpy().tolist()

        print('Done.')
        
        return test_predict
    ``` 


---

# 마. Reference

[1] [SAM: Sharpeness-Aware Minimization](https://arxiv.org/pdf/2010.01412.pdf) : Foret, Pierre, et al. "Sharpness-aware minimization for efficiently improving generalization." arXiv preprint arXiv:2010.01412 (2020).

[2] [EmoBERTa](https://arxiv.org/pdf/2108.12009.pdf) : Kim, Taewoon, and Piek Vossen. "Emoberta: Speaker-aware emotion recognition in conversation with roberta." arXiv preprint arXiv:2108.12009 (2021).

[3] [DistillBERT](https://arxiv.org/pdf/1910.01108.pdf) : Sanh, Victor, et al. "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter." arXiv preprint arXiv:1910.01108 (2019).

[4] [BERT](https://arxiv.org/pdf/1810.04805.pdf) : Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers for language understanding." arXiv preprint arXiv:1810.04805 (2018).

---

# 바. Members
Hyoje Jung | flash1253@naver.com<br>
Yongjae Kim | dydwo322@naver.com<br>
Hyein Oh | gpdls741@naver.com<br>
Seungyong Guk | kuksy77@naver.com<br>
Jaehyeog Lee | tysl4545@naver.com<br>
Hyojin Kang | khj94111@gmail.com
