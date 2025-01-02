# **DiffusionVel: Multi-Information Integrated Seismic Velocity Inversion Using Generative Models**  
### ***Incorporating geoscience data for a more accurate and certain estimation of subsurface models***  

<p align="center">
  <img src="./Multiple%20information.png" alt="Overview" title="An Overview of DiffusionVel" style="width:100%; border: 1px solid #ccc; padding: 5px;"/>
</p>
<p align="center">
  <em>Figure 1: An Overview of DiffusionVel.</em>
</p>




## **Table of Contents**
1. [Introduction](#introduction)
2. [Preparation](#preparation)
3. [Usage](#usage)
4. [Results](#result)
6. [Acknowledgement](#acknowledgement)


---

## **Introduction**
**DiffusionVel** is a state-of-the-art, data-driven machine learning framework designed to enhance the accuracy and reliability of subsurface velocity predictions. By seamlessly integrating multiple geoscience data sources through generative diffusion models (GDMs) in a Plug-and-Play manner, DiffusionVel redefines velocity inversion workflows.

At its core, DiffusionVel constructs a novel conditional score function, leveraging original score functions within diffusion models to incorporate diverse inputs, including seismic data, well logs, background velocity models, and geological insights. This synergy ensures precise and trustworthy inversion results.

For a detailed explanation, please refer to our article at [insert link]. Our work highlights the exceptional potential of diffusion models in geophysical inversions, particularly within data-driven contexts. We sincerely hope this research inspires further exploration and application of diffusion models in geophysics

Our project is built upon the **Stable Diffusion Project** and the **PyTorch Lightning** framework. PyTorch Lightning is user-friendly and provides excellent support for utilizing multiple GPUs and other advanced features.  Additionally, we incorporate code from **OpenFWI** to efficiently load datasets. We sincerely express our gratitude and tribute to these projects for their contributions to the community.

### **Key Features**
- **Unified Integration:** Combine multiple geoscience data sources for enhanced predictions.
- **Priors Correction:** Provide an intuitive solution to refine and correct geological priors in pre-trained diffusion models.

### **Note**
For convenience in testing the control factors, we use separate pre-trained diffusion models (or learned scores) for each type of information. Users are encouraged to train a conditional diffusion model that integrates all these conditions into a single unified model.



---

## **Preparation**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/hao-zhang-rl/DiffusionVel.git
   cd diffusionvel
   ```
2. **Install dependencies:**
Install the project's dependencies using the following command:  
    (This is for CUDA Version 11.5. Please adapt to your specific requirements to ensure the project runs correctly.)  
    ```bash
    conda create --name DiffusionVel
    conda activate DiffusionVel
    pip install -r requirements.txt
    ```
    
<p align="center">
  <img src="./datasets_examples_vel.png" alt="Overview" title="OpenFWI Datasets" style="width:80%; border: 1px solid #ccc; padding: 5px;"/>
</p>
<p align="center">
  <em>Figure 3: Examples of velocity models in OpenFWI datasets .</em>
</p>
<p align="center">
  <img src="./datasets_examples_data_page_1.png" alt="Overview" title="OpenFWI Datasets" style="width:80%; border: 1px solid #ccc; padding: 5px;"/>
</p>
<p align="center">
  <em>Figure 4: Examples of seismic data in OpenFWI datasets .</em>
</p>


3. **Download necessary datasets:**
Most of the datasets used in this project are sourced from **OpenFWI** (please see [OpenFWI Collection](https://github.com/lanl/OpenFWI) and [OpenFWI Datasets](https://openfwi-lanl.github.io/docs/data.html#vel)). For our final tests, we independently prepared the geological distribution of velocity models for the **Hess Model**. As shown in Figure 3 and Figure 4, the datasets provided by OpenFWI are formatted as supervised training datasets: seismic data paired with velocity models. A whole realease of these datasets and details can be found at [OpenFWI Datasets](https://openfwi-lanl.github.io/docs/data.html#vel). The geological prior of Hess model used in this project can be found at [Geological Prior Link](#). This distribution contains only velocity models. If you need seismic data simulations corresponding to these velocity models, we recommend using the **DeepWave Python Package**, available at [DeepWave](https://github.com/ar4/deepwave), for convenient simulation.

4. **Download Pre-trained Checkpoints (optional):**  
   We encourage users to train their own models to test the remarkable results of multi-information integration. However, for a quick start, users can download the pre-trained checkpoints to run the scripts of our project smoothly.
   
   We provide the checkpoints corresponding to different conditional GDMs: Seismic-data GDM, Well-log GDM, and Background velocity GDM. These checkpoints have been trained on the CurveFault-B dataset. Additionally, we offer a checkpoint for the geology-oriented GDM, which has been trained on the FlatFault-B dataset. Users are encouraged to either download this checkpoint or train their own geology-oriented GDM to test the results of corrected priors. All these checkpoints can be downloaded via the following link: [DiffusionVel_checkpoints](https://drive.google.com/drive/folders/17Qe5lWpAy-vGqWZstTvPBsDUgRUiV2uA?usp=drive_link)




## **Usage**

The `script.sh` file contains commands to train and test diffusion models on seismic datasets. Below are examples of how to use these commands:

### Training Your Own GDM
For example, to train your seismic Generative Diffusion Model (GDM) on the FlatFault-B dataset, run the following command:

```bash
python DiffusionVel.py --if_train --batch_size=16 --learning_rate=1e-4 --check_val_every_n_epoch=10 \
--anno_path=./split_files --train_anno=flatfault_b_train_test.txt --val_anno=flatfault_b_test_test.txt \
--test_anno=flatfault_b_test_test.txt --dataset=flatfault-b --max_epochs=200 --model_config=./models/dpm.yaml \
--conditioning_key=seis_concat
```
### Testing a Trained GDM
For example, to test the trained seismic GDM on the CurveFault-B dataset, use this command:
```bash
python DiffusionVel.py --batch_size=16 --anno_path=./split_files --test_anno=curvefault_b_test_test.txt \
--dataset=curvefault-b --model_config=./models/dpm.yaml --conditioning_key=seis_concat --seis=your_path
```
### Integrating Multi-Information with Pre-Trained GDMs
To use pre-trained GDMs for information integration, execute the command below:
```bash
python DiffusionVel.py --factor_0=0.25 --factor_1=0.5 --factor_2=0.25 --batch_size=16 \
--anno_path=./split_files --test_anno=curvefault_b_test_test.txt --dataset=curvefault-b \
--model_config=./models/dpm.yaml --conditioning_key=seis_well_back_concat \
--well=your_path_well --back=your_path_back --seis=your_path_seis --geo=your_path_geo
```

If you have downloaded the checkpoints, ensure they are placed in the appropriate paths specified as `your_path_well`, `your_path_back`, `your_path_geo`, and `your_path_seis`. **Note:** It is not mandatory to use all four sources. Simply adapt the `--conditioning_key` argument to match the sources you intend to include.

## **Results**
<p align="center">
  <img src="./datasets_part1_replot_page_1.png" alt="Overview" title="" style="width:60%; border: 1px solid #ccc; padding: 5px;"/>
</p>
<p align="center">
  <em>Figure 5: Inversion resutls of DiffusionVel using only seismic data, with comparasion with exsisting methods.</em>
</p>
Figure 5 shows the inversion results of DiffusionVel using only seismic data, compared with existing methods: Conventional FWI, InversionNet, and VelocityGAN.
<p align="center">
  <img src="./comparison_with_residuals.gif" alt="Overview" title="Generation Process of DiffusionVel With Multi-Information Integration" style="width:100%; border: 1px solid #ccc; padding: 5px;"/>
</p>
<p align="justify">
  <em>Figure 6: Generation Process (predicted x_0) of DiffusionVel With Multi-Information Integration of Seismic Data, Well Logs and Background Velocity.</em>
</p>
Figure 6 shows the generation process of DiffusionVel With Multi-Information Integration (Here we use 200 DDIM sampling steps for better demonstration. Usually 10 timesteps will be enough for good generation.) Figure 7 displays the integration results of correct geological priors. 

<p align="center">
  <img src="./datasets_geo_reg_0.5_even_label_page_1.png" alt="Figure 7a" style="width:30%; margin: 5px;">
  <img src="./datasets_geo_reg_0_even_samples_page_1.png" alt="Figure 7b" style="width:30%; margin: 5px;">
</p>



<p align="center">
  <img src="./datasets_geo_reg_0.5_even_samples_page_1.png" alt="Figure 7c" style="width:30%; margin: 5px;">
  <img src="./datasets_geo_reg_1_even_samples_page_1.png" alt="Figure 7d" style="width:30%; margin: 5px;">
</p>
<p align="justify">
  <em>Figure 7 Demonstration of corrected geological priors. Top Left: Ground Truth from Flatfault-b datasets. Top Right: Estimated results using only seismic GDM pre-trained on CurveFault-b datasets. Bottom Left: Integration of seismic data and prior geological information. Bottom  Right: Estimated results using only geology-oriented GDM pre-trained on Flatfault-b datasets</em>
</p>



## **Acknowledgments**
- **Dataset:** [Link or source]
- **Inspiration:** [Inspirational projects or papers]
- **Tools:** [Frameworks or libraries used]
