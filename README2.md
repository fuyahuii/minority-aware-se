The official codes for the paper: Yahui Fu, Zi Haur Pang, and Tatsuya Kawahara. "[Minority-Aware Satisfaction Estimation in Dialogue Systems via Preference-Adaptive Reinforcement Learning](https://arxiv.org/abs/2511.05407)." IJCNLP-AACL 2025 (Main Conference).


### **Abstract**
<p style="text-align: justify;">
User satisfaction in dialogue systems is inherently subjective. When the same response strategy is applied across users, minority users may assign different satisfaction ratings than majority users due to variations in individual intents and preferences. However, existing alignment methods typically train one-size-fits-all models that aim for broad consensus, often overlooking minority perspectives and user-specific adaptation. We propose a unified framework that models both individual- and group-level preferences for user satisfaction estimation. First, we introduce Chain-of-Personalized-Reasoning (CoPeR) to capture individual preferences through interpretable reasoning chains. Second, we propose an expectation-maximization-based Majority-Minority Preference-Aware Clustering (M2PC) algorithm that discovers distinct user groups in an unsupervised manner to learn group-level preferences. Finally, we integrate these components into a preference-adaptive reinforcement learning framework (PAda-PPO) that jointly optimizes alignment with both individual and group preferences. Experiments on the Emotional Support Conversation dataset demonstrate consistent improvements in user satisfaction estimation, particularly for underrepresented user groups.
</p>

### **Model Architecture**
<div align="center">
  <img src="./figs/arch.png" alt="Model Architecture" width="650"/>
</div>

### **Supervised Fine-Tuning (SFT)**
#### **Training**
#### Main script: ```sft_coper.py```
#### Run with coper template
```
bash scripts/sft/coper.sh
```
#### **Inference and evaluation**
```
python inference/eval_sft_coper.py
```

### **Majority-Minority Preference-Aware Clustering (M2PC)**
```
python m2pc.py
```

### **Preference-Adaptive Reinforcement Learning (PAda-PPO)**

#### **Training**
#### Main script: ```pada-ppo.py```

#### Run with coper template
<!-- ```
bash scripts/rl_base/base_train.sh
```
```
bash scripts/rl_ucot/ucot_train.sh
``` -->
```
bash scripts/rl_coper/coper_train.sh
```
#### **Inference and Evaluation**
<!-- #### Main script for using base or ucot template: ```inference/eval_rl_base.py``` -->
#### Main script for using coper template: ```inference/eval_rl_coper.py```

#### Run with coper template
<!-- ```
bash scripts/rl_base/base_infer.sh 
```
```
bash scripts/rl_ucot/ucot_infer.sh
``` -->
```
bash scripts/rl_coper/coper_infer.sh
```

### **Citation**
If you find this repository or paper useful, please kindly cite our paper:
```
@article{fu2025minority,
  title={Minority-Aware Satisfaction Estimation in Dialogue Systems via Preference-Adaptive Reinforcement Learning},
  author={Fu, Yahui and Pang, Zi Haur and Kawahara, Tatsuya},
  journal={arXiv preprint arXiv:2511.05407},
  year={2025}
}
```
### **Contact**
For any questions related to the paper or this repository, feel free to contact Yahui Fu at [fu.yahuiii@gmail.com](mailto:fu.yahuiii@gmail.com).
