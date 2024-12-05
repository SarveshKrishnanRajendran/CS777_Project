# AWS Spot Price Prediction System

### CS 777 – Big Data Analysis Term Project  
**Contributors:** Sarvesh Krishnan Rajendran, Shlok Mandloi  
**University:** Boston University Metropolitan College  
**Guidance:** Dr. Farshid Alizadeh-Shabdiz  

---

## 📘 Project Overview  
The AWS Spot Price Prediction System leverages **machine learning** and **distributed computing** to predict dynamic pricing trends for AWS EC2 spot instances. This system empowers users to optimize resource allocation strategies by providing insights into price fluctuations, reducing costs, and mitigating risks.

### 🔍 Objectives:  
- Predict AWS spot instance prices using historical data.
- Build scalable, efficient pipelines for data preprocessing and modeling.
- Utilize advanced regression techniques and distributed deep learning for high predictive accuracy.

---

## 📊 Data  
- **Source:** [Kaggle AWS Spot Pricing Market](https://www.kaggle.com/datasets/noqcks/aws-spot-pricing-market)  
- **Schema:**  
  - `datetime`: Timestamp of recorded price  
  - `instance_type`: AWS instance type (e.g., t2.micro)  
  - `os`: Operating system type (Linux/Windows)  
  - `region`: AWS region  
  - `price`: Spot price in USD  

---

## 🛠️ Features & Tools  

### **Data Preprocessing**  
- Removal of duplicates and handling missing values.  
- Filtering invalid or outlier prices.  
- **Feature Engineering**:  
  - `StringIndexer` and `OneHotEncoder` for categorical encoding.  
  - Dropping irrelevant columns to streamline datasets.  
- Output stored in **Parquet** format for scalability.

### **Modeling Approaches**  
1. **Linear Regression**  
   - Baseline model for performance benchmarking.  
   - Hyperparameter tuning using `regParam` and `elasticNetParam`.  

2. **Tree-Based Models**  
   - **Decision Tree Regressor** and **Random Forest Regressor** for non-linear relationships.  
   - Hyperparameter optimization for depth, bins, and trees.  

3. **Distributed Neural Networks**  
   - Feed-forward architecture with **Elephas** for distributed training on Spark clusters.  
   - Multi-layer architecture with dropout regularization for robust predictions.  

### **Infrastructure**  
- **PySpark** for distributed preprocessing and modeling.  
- **Google Cloud Storage & Dataproc** for scalable, efficient workflows.  
- **TensorFlow & Keras** for building and evaluating neural networks.

---

## 🚀 Results  
| Metric                 | Linear Regression | Decision Tree | Random Forest | Neural Network |
|------------------------|------------------|---------------|---------------|----------------|
| **MSE**               | 16.57           | 12.65         | 12.83         | 0.0004         |
| **RMSE**              | 4.07            | 3.55          | 3.58          | 0.0210         |

The distributed deep learning model demonstrated superior performance, leveraging the scalability of Elephas and PySpark for efficient large-scale data processing.

---

## 🔮 Future Scope  
- **Real-time Optimization**: Use AWS SDKs (e.g., Boto3) to fetch live spot prices and optimize bidding strategies in real time.  
- **Streaming with Kafka**: Integrate Apache Kafka for low-latency price monitoring and dynamic resource allocation.  
- **Advanced Architectures**: Explore transformer-based or recurrent neural network models for improved predictions.

---

## 📂 Repository Structure  
- `nn_elephas.py`: Neural network implementation with Elephas and PySpark.  
- `preprocessing.py`: Data preprocessing pipeline using PySpark.  
- `model_testing.py`: Evaluation of regression models.  
- `install_tf.sh`: Shell script for installing TensorFlow and Elephas dependencies.  
- **Report and Presentation**:  
  - `Rajendran_SarveshKrishnan_Mandloi_Shlok_report.pdf`  
  - `Rajendran_SarveshKrishnan_Mandloi_Shlok_ppt.pptx`  

---

## 🤝 Acknowledgements  
We thank **Dr. Farshid Alizadeh-Shabdiz** for guidance and Boston University for providing resources to undertake this project.

 
