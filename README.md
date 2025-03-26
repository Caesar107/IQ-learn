# **Improved Inverse Q-Learning (IQ-Learn)**  

This is an **improved version** of [IQ-Learn](https://arxiv.org/abs/2106.12142), originally proposed in **NeurIPS 2021**.  
Our modifications include:  

- ✅ **Added KL divergence and reward-based baselines**  
- ✅ **Extended support for Gym Atari and MuJoCo environments**  
- ✅ **Optimized training pipeline for better stability**  

### **Original Paper:**  
📄 **IQ-Learn: Inverse Soft-Q Learning for Imitation**  
➡️ [**arXiv Link**](https://arxiv.org/abs/2106.12142)  

### **Original Codebase:**  
🖥️ [**GitHub Repository**](https://github.com/Div99/IQ-Learn)  

---

## **Introduction**  
IQ-Learn is a **state-of-the-art imitation learning framework** that directly learns soft Q-functions from expert data. Unlike traditional adversarial approaches (e.g., GAIL, AIRL), IQ-Learn provides a **simple, stable, and data-efficient alternative** for both offline and online imitation learning.  

### **Our Key Modifications**  
1️⃣ Introduced **KL divergence** and **reward-based baselines** to improve performance.  
2️⃣ Adapted the method for **Gym Atari and MuJoCo environments**.  
3️⃣ Optimized the **training pipeline** for better efficiency and generalization.  

---

## **Key Advantages**  

✔️ **Drop-in replacement for Behavior Cloning**  
✔️ **Non-adversarial online imitation learning (successor to GAIL & AIRL)**  
✔️ **Performs well with very sparse expert data**  
✔️ **Scales to complex environments (Atari, MuJoCo)**  
✔️ **Can recover reward functions from the environment**  

---

## **Installation & Usage**  

Please refer to the [iq_learn](iq_learn) directory for installation and usage instructions.

---
#### if you can't use WANDB,then you may can use
```
$env:WANDB_MODE = "offline"
```
---

## **Trajectory Conversion Tool**  

We provide a utility script `convert_transitions.py` to **convert expert trajectories** into the format required by IQ-Learn.  

This is useful when you have custom environments or datasets and want to apply IQ-Learn directly.  

### **Usage Example:**  
```bash
python convert_transitions.py --input expert_data.pkl --output processed_data.pt
```

### **Options:**  
- `--input`: Path to your raw expert data file (e.g., a pickle or JSON file).  
- `--output`: Output file path in IQ-Learn-compatible `.pt` format.  

> Make sure your expert data includes state, action, next_state, reward, and done fields.

---

## **Demonstrations**  

### **Imitation Learning on Atari**  
IQ-Learn achieving human-level imitation in various Atari games:  

<p float="left">
<img src="videos/pong.gif" width="250">
<img src="videos/breakout.gif" width="250">
</p>
<p float="left">
<img src="videos/space.gif" width="250">
<img src="videos/qbert.gif" width="250">
</p>

## **Citing This Work**  
If you use this code, please cite the original **IQ-Learn** paper:  

```
@inproceedings{garg2021iqlearn,
title={IQ-Learn: Inverse soft-Q Learning for Imitation},
author={Divyansh Garg and Shuvam Chakraborty and Chris Cundy and Jiaming Song and Stefano Ermon},
booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
year={2021},
url={https://openreview.net/forum?id=Aeo-xqtb5p}
}
```

---

## **Contact**  
For any questions or discussions, feel free to open an issue or reach out! 🚀  

---
