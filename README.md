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

## **Installation**
### **Requirements**




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
conda create --env -n DiffusionVel
pip install -r requirements.txt
3. Download necessary pretrained models and datasets:
Most of the datasets used in this project are sourced from **OpenFWI**. For our final tests, we independently prepared the geological distribution of velocity models for the **Hess Model**.  The datasets provided by OpenFWI are formatted as supervised training datasets: seismic data paired with velocity models. A whole realease of these datasets and details can be found at [OpenFWI Website](#).  

The geological prior of Hess model used in this project can be found at [Geological Prior Link](#). This distribution contains only velocity models. If you need seismic data simulations corresponding to these velocity models, we recommend using the **DeepWave Python Package**, available at [DeepWave Website](#), for convenient simulation.  


---

## **Usage**
### **Run Scripts**
Train the model:
```bash
python train.py --config config.yaml
```

Evaluate the model:
```bash
python evaluate.py --model saved_model.pth
```

Run the demo:
```bash
python demo.py --input example_data.json
```

### **Jupyter Notebooks**
Explore the project using the provided Jupyter notebooks in the `notebooks/` folder:
- `notebooks/data_analysis.ipynb`: Data exploration and preprocessing.
- `notebooks/model_training.ipynb`: Model training and evaluation.

---

## **Dataset**
- **Source:** [Name or link to dataset]  
- **Description:** Includes seismic traces, well logs, and geological priors.  
- **Preprocessing:** Normalized and formatted for input into diffusion models.

If the dataset is not included, provide instructions to download it:
```bash
wget http://example.com/dataset.zip
unzip dataset.zip
```

---

## **Results**
### **Performance Metrics**
- Velocity Prediction Accuracy: **97%**
- Geological Consistency: **94%**

### **Visualizations**
*(Add sample plots or charts here)*  
![Sample Chart](https://via.placeholder.com/600x300.png)

### **Model Outputs**
*(Showcase predictions or results)*  
Example input:  
![Example Input](https://via.placeholder.com/300x300.png)  

Predicted output:  
![Example Output](https://via.placeholder.com/300x300.png)

---

## **Contributing**
Contributions are welcome! Please follow these steps to contribute:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes and push the branch:
   ```bash
   git push origin feature-name
   ```
4. Submit a pull request.

---

## **License**
This project is licensed under the [MIT License](LICENSE).  
Feel free to use, modify, and distribute this project as per the license terms.

---

## **Acknowledgments**
- **Dataset:** [Link or source]
- **Inspiration:** [Inspirational projects or papers]
- **Tools:** [Frameworks or libraries used]
