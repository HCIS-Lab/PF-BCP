# [ICRA 2025] PF-BCP

  [**Project Page**](https://hcis-lab.github.io/PF-BCP/) 
| [**Video Overview**](https://www.youtube.com/watch?v=7Een9DPa9ms)
| [**Paper Preprint**](https://arxiv.org/abs/2409.15846)

![Python Version](https://img.shields.io/badge/PyThon-3.7-blue.svg)
![PyTorch Version](https://img.shields.io/badge/PyTorch-1.8.0-EE4C2C.svg)
[![arXiv](https://img.shields.io/badge/arXiv-2409.15846-b31b1b.svg)](https://arxiv.org/abs/2409.15846)
[![GitHub license](https://img.shields.io/badge/License-GPL_Version_3.0-808080.svg)](./LICENSE)


This repository contains the official code for training and evaluating the methods as described in:

<!-- ## 📋 Overview
![PF+BCP](images/teaser.png)
<img src="images/teaser.png" alt="PF+BCP" width="500"/>

**Authors:**
[**Pang-Yuan Pao**](https://github.com/WaywayPao/),
[**Shu-Wei Lu**](https://www.linkedin.com/in/shu-wei-lu/),
**Ze-Yan Lu**,
[**Yi-Ting Chen**](https://sites.google.com/site/yitingchen0524)

**Affiliation: [National Yang Ming Chiao Tung University](https://www.nycu.edu.tw/nycu/en/index)** -->


> **Potential Field as Scene Affordance for Behavior Change-Based Visual Risk Object Identification** <br/>
> [Pang-Yuan Pao](https://sites.google.com/view/pang-yuan-pao/),
[Shu-Wei Lu](https://www.linkedin.com/in/shu-wei-lu/),
Ze-Yan Lu and
[Yi-Ting Chen](https://sites.google.com/site/yitingchen0524) <br/>
> [National Yang Ming Chiao Tung University](https://www.nycu.edu.tw/nycu/en/index)


<p align="center">
   <img src="images/10_s-2_0_c_sr_sl_1_0_WetSunset_high_.gif" alt="Teaser">
   <img src="images/10_s-6_0_p_j_f_1_j_MidRainSunset_low_.gif" alt="Teaser">
   <br/>
   <sub><em> Visual Risk Object Identification by different methods. <br/>
   All detected risk objects are shown with green bounding boxes, while ground truth risks are masked in red. <br/>
   <b>GT</b>, <b>BS</b>, and <b>PF</b> refer to Ground Truth, Bird's-Eye-View Segmentation, and Potential Field, respectively.   
   </em></sub>
</p>


## 🔥 Update
* **January 2025: 🎉 PF-BCP got accepted by ICRA 2025!**
* **October 2024: 📊 ROI demo tool released!**
* **September 2024: 📢 Project page and YouTube video announced!**


## ⚙️ Getting Started

### 📝 System Setup
* **Operating System:** Linux Ubuntu 18.04
* **Python Version:** 3.7
* **PyTorch Version:** 1.8.0
* **CUDA Version:** 11.6
* **GPU:** Nvidia RTX 3090
* **CPU:** Intel Core i7-10700

### 📥 Dependency Installation

1. Clone the Repository
    ```bash
    git clone https://github.com/HCIS-Lab/PF-BCP
    ```

2. Create and activate a new [Conda](https://docs.anaconda.com/miniconda/) environment:
    ```bash
    conda create -n YOUR_ENV python=3.7
    conda activate YOUR_ENV
    cd PF-BCP
    ```

3. Run the following command to install all required packages from ``requirements.txt``:
    ```bash
    pip install -r requirements.txt
    ```

### 📦 Datasets Downloads

* Download `RiskBench_Dataset` [here](https://nycu1-my.sharepoint.com/:f:/g/personal/ychen_m365_nycu_edu_tw/EviA5ovlh6hPo_ZXEPQjxAQB2R3vNubk3HM1u4ib1VdPFA?e=WHEWdm).

* Download `metadata.zip` [here](./ROI_demo/metatdata.zip).



<!-- ## 🚀 Usage

### Comming Soon -->



## 📊 ROI Demo

We perform offline risk object identification evaluations and visualize fine-grained scenario-based analysis using preserved prediction data as input.


To perform the evaluation, please follow the instructions provided [here](./ROI_demo).


## 📄 BibTeX

If our work contributes to your research, please consider citing it with the following BibTeX entry:

```bibtex
@inproceedings{pao2025pfbcp,
	title   = {{Potential Field as Scene Affordance for Behavior Change-Based Visual Risk Object Identification}},
	author  = {Pang-Yuan Pao and Shu-Wei Lu and Ze-Yan Lu and Yi-Ting Chen},
	journal = {IEEE International Conference on Robotics and Automation (ICRA)},
	year    = {2025}
}
```

<!-- # 📜 License

This project is licensed under the [GPL-3.0 LICENSE](./LICENSE). -->


## 🙌 Acknowledgment

We acknowledge that the dataset and baselines used in this project are adapted from [RiskBench](https://hcis-lab.github.io/RiskBench/).



