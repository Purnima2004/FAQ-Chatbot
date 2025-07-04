College Name: DDD  
Student Name: XXX  
Email Address: student@domain.com  

# GEN AI PROJECT SUBMISSION DOCUMENT

---

## 1. Project Title
**Next Sentence Prediction using Generative AI**

## 2. Summary of Work Done

### Proposal and Idea Submission
In this phase we identified the problem statement and proposed the idea of developing a **Next Sentence Prediction** (NSP) system using Generative AI, specifically leveraging GPT-based models.  
Objectives:
- Understand the working of generative models in NLP.  
- Use pre-trained models to generate context-aware sentences.  
- Create a user interface to interact with the model.  

A detailed proposal covering the problem definition, objectives, required tools and expected outcomes was submitted.

### Execution and Demonstration
Implementation was carried out with **Python**, **HuggingFace Transformers**, and **Streamlit**.  The following tasks were completed:
- Built a web-based interface using Streamlit.  
- Loaded a pre-trained GPT-2 model to generate the next sentence from user input.  
- Configured the application to accept text, query the model and display the **top-3** predicted next sentences.  
- Tested the model with diverse inputs to validate performance and relevance.

Sample outputs and the complete source code have been documented and are available in the linked repository.

---
College Name: DDD  
Student Name: XXX  
Email Address: student@domain.com  

## 3. GitHub Repository Link
You can access the complete code-base, README instructions and supporting resources at the following link:  
👉 **GitHub Repository – Next Sentence Prediction using Gen AI**  
*(Replace this line with the actual GitHub URL)*

---

## 4. Testing Phase

### 4.1 Testing Strategy
The system was evaluated across a variety of use-cases to ensure robustness and accuracy, employing both manual and automated techniques to verify:
- **Input Handling:** Correct processing of different input lengths (short, long, incomplete).  
- **Contextual Relevance:** Generated sentences remain coherent with the prompt.  
- **Edge-Case Handling:** Behaviour with incomplete or nonsensical input.

### 4.2 Types of Testing Conducted
1. **Unit Testing** – Verified each module (generation function, UI components, API wrapper) in isolation.  
2. **Integration Testing** – Ensured seamless interaction between the GPT-2 model and the Streamlit interface.  
3. **User Testing** – Collected feedback from test users regarding usability, design and output quality.  
4. **Performance Testing** – Measured response times on varying input sizes.

### 4.3 Results
- **Accuracy:** The system consistently produced contextually relevant predictions.  
  *Example*: Input *"I went to the park to"* produced outputs such as *"play basketball"* and *"have fun with my friends"*.  
- **Response Time:** Predictions were generated with minimal delay.  
- **Edge Cases:** Even for inputs like *"Flim flam foo"* the model returned plausible continuations, illustrating resilience to unknown contexts.

---

## 5. Future Work
1. **Model Fine-tuning** – Train on domain-specific corpora for specialised use-cases.  
2. **Multi-Modal Extension** – Incorporate images / video context for richer predictions.  
3. **Real-time Collaboration** – Enable multiple users to co-create or refine outputs simultaneously.  
4. **User Feedback Loop** – Allow rating of predictions to support continual model improvement.  
5. **Multi-Language Support** – Expand generation capabilities to additional languages using multilingual models.

---

## 6. Conclusion
This project demonstrates the capability of Generative AI to perform meaningful natural-language predictions. Progressing from concept to fully-tested application, it showcases how transformer-based models can underpin practical NLP tools such as sentence completion, writing assistance and chatbot development.

---
College Name: DDD  
Student Name: XXX  
Email Address: student@domain.com
