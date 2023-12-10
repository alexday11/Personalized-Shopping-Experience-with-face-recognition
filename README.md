# **Personalized Shopping Experience with Face Recognition**

This project introduces an innovative approach to enhance the shopping experience by incorporating facial recognition technology. Inspired by the notion of advertising booths in department stores or shopping malls, the concept envisions personalized product recommendations based on the customer's profile and preferences. When a customer, especially a member of the department store, approaches an advertising booth, the system captures their facial features and recommends products tailored to their interests. This recommendation is not only based on current preferences but also considers discounted items that align with the customer's tastes.

The project comprises two interconnected systems:

1. **Collaborative Filtering Product Recommendation System:**
    This system leverages collaborative filtering principles to recommend products based on similarities in customer behavior. Similar to discreetly observing other customers' shopping carts, the system identifies commonalities and suggests relevant products. The recommendation model employs word2vec for feature creation, generating unique vector values for each product. By calculating the similarity value of product codes, the system provides personalized recommendations without considering the product content or keywords explicitly.

2. **Facial Recognition System:**
    The facial recognition system consists of two main components:

    - Face Detection: Identifying the location of the face in the image, followed by cropping to isolate the facial region for processing before model training.

    - Face Recognition: Employing a facial classification system using the Face Embedding method, specifically Facenet. This involves extracting prominent features from face images to emphasize essential vector values. The input to the model is a 160x160x3 face image, resulting in a 512-dimensional vector. The project focuses on classifying faces into 10 categories, including the creator's face, using a straightforward model—Support Vector Classification (SVC). The model's accuracy is an impressive 100%.

By integrating both systems, the project facilitates efficient product recommendations based on customers' faces. This personalized shopping experience not only adds convenience but also enhances customer satisfaction by tailoring suggestions to individual preferences with high accuracy.



![example1](img1.png)

![example2](img2.png)

### **How to Use**
1. Open the command prompt (cmd) and navigate to the folder where you want to store the project. For example, use the following command: cd 'path/to/your/directory'
2. Clone the project repository by entering the command: git clone https://github.com/alexday11/Personalized-Shopping-Experience-with-face-recognition.git
3. Move into the project directory: cd 'Personalized-Shopping-Experience-with-face-recognition'
4. Install the required dependencies by running: pip install -r requirements.txt
5. The "Image" folder contains the file "10classes.zip." Extract the file to use the images for testing the app.
6. Return to the command prompt and start the app with the following command: streamlit run app.py


