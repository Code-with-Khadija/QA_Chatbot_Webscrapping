# Web Scraping and NLP-based Chatbot

This repository contains a Python script that scrapes text from a website, preprocesses it, and implements a simple chatbot that can answer questions based on the scraped content. The chatbot uses a combination of TF-IDF vectorization, cosine similarity, and a pre-trained transformer model for question-answering.

## Features

- **Web Scraping**: Extracts text content from a given URL using `requests` and `BeautifulSoup`.
- **Text Cleaning and Preprocessing**: Cleans the scraped text by removing special characters, references, and stopwords.
- **TF-IDF Vectorization**: Converts the cleaned text into TF-IDF vectors for similarity search.
- **Cosine Similarity Search**: Finds the most relevant sentence in the scraped text based on a user query.
- **Question-Answering with Transformers**: Uses a pre-trained transformer model (`deepset/roberta-base-squad2`) to answer questions based on the scraped text.
- **Interactive Chatbot**: Provides an interactive command-line interface for users to ask questions and receive answers.

## Requirements

- Python 3.x
- Libraries: `requests`, `beautifulsoup4`, `nltk`, `numpy`, `scikit-learn`, `transformers`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download NLTK data:
   ```python
   import nltk
   nltk.download("punkt")
   nltk.download("stopwords")
   ```

## Usage

1. Run the script:
   ```bash
   python your_script_name.py
   ```

2. The chatbot will prompt you to ask questions. Type your question and press Enter.

3. To exit the chatbot, type `exit`, `quit`, or `stop`.

## Example

```bash
Ask me anything about the scraped website!
You: What is machine learning?
Bot: Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.
You: exit
Goodbye!
```

## Customization

- **Change the URL**: Modify the `url` variable in the script to scrape a different website.
- **Adjust Relevance Threshold**: Change the `similarities[best_idx] > 0.1` condition in the `search_website` function to adjust the relevance threshold for cosine similarity search.
- **Use a Different Model**: Replace `deepset/roberta-base-squad2` with another pre-trained model from the Hugging Face Model Hub.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [NLTK](https://www.nltk.org/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/)
