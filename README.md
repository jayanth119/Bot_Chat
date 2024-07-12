# ChatBot

# Food Recipes Chatbot using Google Gemini

Welcome to the Food Recipes Chatbot repository! This project features a chatbot built with Google Gemini that provides users with delicious and easy-to-follow food recipes.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)

## Introduction

This chatbot is designed to help users discover new recipes and find detailed cooking instructions based on their preferences and available ingredients. Whether you're a seasoned chef or a beginner in the kitchen, our chatbot can assist you in creating amazing dishes.

## Features

- **Recipe Suggestions**: Get personalized recipe suggestions based on your preferences.
- **Ingredient-Based Search**: Find recipes based on the ingredients you have.
- **Step-by-Step Instructions**: Follow detailed cooking instructions to create delicious meals.
- **User-Friendly Interface**: Interact with the chatbot through an easy-to-use interface.

## Technologies Used

- **Python**: Programming language .
- **Google Gemini**: For natural language understanding and processing.
- **LangChain**: To build and manage language models and pipelines.
- **Pinecone**: As a vector database for efficient search and retrieval of recipes.

## Installation

To run this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/jayanth119/Bot_Chat.git
   cd Bot_Chat
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up the vector database**:
   - Follow Pinecone's documentation to set up your vector database and get your API key.
   - Create an `.env` file in the root directory and add your Pinecone API key:
     ```
     PINECONE_API_KEY=your_pinecone_api_key
     ```


Feel free to customize this template according to your project's specific details and requirements. 
