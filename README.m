# 🧠 Rubbish Classification App

This is a Streamlit web application designed to classify three types of waste: **plastic**, **glass**, and **metal**.

The app is part of a **supervised learning project** led by **Nadia Urban** at **Shanghai Thomas School**, where students learn how to design and deploy machine learning models through hands-on, real-world problems.

---

## 🎓 Project Overview

This project follows five main stages:

1. **Model Design** – Planning the purpose and scope of the classification model  
2. **Data Collection** – Gathering image data of plastic, glass, and metal rubbish  
3. **Model Training** – Using Google's Teachable Machine to train a CNN classifier  
4. **Model Assessment** – Evaluating model performance and class balance  
5. **Web App Design** – Deploying the model in an interactive, user-friendly Streamlit app

---

## 🛠️ App Description

The goal of this app is to **help people correctly classify their rubbish**, especially when it's not obvious how to sort it.

### 🧾 Model Information
- **Classes:**  
  1. Plastic  
  2. Glass  
  3. Metal  
- **Goal:** 🎯 We're afraid that someone may not be able to distinguish how to sort garbage. We want to help people classify their rubbish.  
- **Data Type:** 🖼️ Images of plastic, glass, and metal rubbish  
- **Data Source:** 🌐 Collected from **Baidu**  and **kaggle.com**
- **Training:** 🏋️ Teachable Machine  
- **Model Type:** 🧠 Convolutional Neural Network (CNN)

---

## 🖼️ Training Data Samples

| Class    | Image Preview        | Number of Training Images |
|----------|----------------------|----------------------------|
| Plastic  | `example1.jpg`       | 468 photos                 |
| Glass    | `example2.jpg`       | 486 photos                 |
| Metal    | `example3.jpg`       | 396 photos                 |

(*These example images are included in the app sidebar for reference.*)

---

## 👩‍🔬 Model Authors

- **刘素婷 Sophie Liu**  
- **乔馨仪 Joy Qiao**

---

## ✨ Credits

This project was developed as part of the **AI & Machine Learning program** at **Shanghai Thomas School** which is design and taught by **Nadia Urban**.


## 🚀 Deployment

The app is deployed using [Streamlit Cloud](https://streamlit.io/cloud) and can be run locally by installing the required dependencies:

```bash
pip install streamlit tensorflow pillow numpy
streamlit run app.py
