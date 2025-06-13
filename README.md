# Streamlit LangChain Chatbot with Groq

A simple chatbot application built with Streamlit and LangChain, powered by Groq's LLM models.

## Features

- Interactive chat interface with Streamlit
- Conversation memory using LangChain's ConversationBufferMemory
- Powered by Groq's LLM models
- Environment variable management for API keys

## Prerequisites

- Python 3.8 or higher
- A Groq API key (sign up at https://console.groq.com/)

## Installation

1. Clone this repository or download the files

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Add your Groq API key to the `.env` file:

```
GROQ_API_KEY=your_actual_api_key_here
```

## Running the Application

Run the Streamlit app with the following command:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## Usage

1. Type your message in the chat input field at the bottom of the page
2. Press Enter or click the send button
3. The AI will respond to your message
4. Your conversation history is maintained during the session

## Customization

You can modify the `app.py` file to:

- Change the LLM model (update the `model_name` parameter)
- Adjust the prompt template
- Add additional features to the Streamlit interface
- Implement different memory types from LangChain

## License

This project is open source and available under the MIT License.