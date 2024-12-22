# **DiffusionVel: Multi-Information Integrated Seismic Velocity Inversion Using Generative Models**  
*Incorporating geoscience data for a more accurate and certain estimation of subsurface models*

![Project Banner](https://via.placeholder.com/800x200.png?text=Project+Banner)  
*(Replace with an illustrative image or your project's logo)*

---

## **Table of Contents**
1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Dataset](#dataset)
6. [Results](#results)
7. [Contributing](#contributing)
8. [License](#license)

---

## **Introduction**
**DiffusionVel** is a cutting-edge data-driven machine learning project designed to predict subsurface velocity models by integrating multiple sources of geoscience data using generative diffusion models (GDMs) in a Plug-and-Play manner. By constructing a new conditional score function with original score functions in diffusion models, it incorporates diverse information such as seismic data, well logs, background velocity, and geological knowledge to obtain precise and reliable inversion results. For more information, please read our article at..... 

Our project is built upon the **Stable Diffusion Project**, which utilizes the **PyTorch Lightning** framework. PyTorch Lightning is user-friendly and provides excellent support for utilizing multiple GPUs and other advanced features.  Additionally, we incorporate code from **OpenFWI** to efficiently load datasets. We sincerely express our gratitude and tribute to these projects for their contributions to the community.

### **Key Features**
- **Unified Integration:** Combine multiple geoscience data sources for enhanced predictions.
- **Priors Correction:** Provide an intuitive solution to refine and correct geological priors in pre-trained diffusion models.

### **Note**
For convenience in testing the control factors, we use separate pre-trained diffusion models (or learned scores) for each type of information. Users are encouraged to train a conditional diffusion model that integrates all these conditions into a single unified model.



---

## **Preparation**

### **Steps**
1. Clone the repository:
   ```bash
   git clone https://github.com/hao-zhang-rl/DiffusionVel.git
   cd diffusionvel
   ```
2. Install dependencies:
Install the project's dependencies using the following command:  
*(This is for CUDA Version 11.5. Please adapt to your specific requirements to ensure the project runs correctly.)*  
```bash
conda create --name DiffusionVel
conda activate DiffusionVel
pip install -r requirements.txt
```


3. Download necessary datasets:
Most of the datasets used in this project are sourced from **OpenFWI**. For our final tests, we independently prepared the geological distribution of velocity models for the **Hess Model**.  The datasets provided by OpenFWI are formatted as supervised training datasets: seismic data paired with velocity models. A whole realease of these datasets and details can be found at [OpenFWI Website](#).  

The geological prior of Hess model used in this project can be found at [Geological Prior Link](#). This distribution contains only velocity models. If you need seismic data simulations corresponding to these velocity models, we recommend using the **DeepWave Python Package**, available at [DeepWave Website](#), for convenient simulation.
4. Download Pre-trained Checkpoints:
We encourage users to train their own models to test the remarkable results of multi-information integration. However, for a quick start, users can download the pre-trained checkpoints to run the scripts of our project smoothly. We prepare for users both the training scrip and test script to directly apply the pre-trained models.

We provide five types of checkpoints, each corresponding to a specific generative diffusion model (GDM):

Seismic-data GDM
Well-log GDM
Background velocity GDM
Geological-oriented GDM
These checkpoints were trained on the CurveFault-B dataset and are referenced in the examples section of our paper.
---

## **Usage**
### **Train Your Own Diffusion Models**
1. Seismic Data GDM
```bash
python train.py --config config.yaml


---

## **Acknowledgments**
- **Dataset:** [Link or source]
- **Inspiration:** [Inspirational projects or papers]
- **Tools:** [Frameworks or libraries used]
