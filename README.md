# Empório da Música — AI Customer Support Agent

The project aims to help the customer support team of Empório da Música, a fictional musical instruments store based in Campo Grande, Brazil, by automating responses to frequently asked questions about products, orders, promotions, and store policies.

## Overview

The assistant uses a **ReAct (Reasoning and Acting) agent** to understand customer requests, decide which tools to use, retrieve relevant information, and generate grounded responses.

The solution combines structured data retrieval from CSV files with Retrieval-Augmented Generation (RAG) for unstructured information contained in the store's policy manual.

### Main capabilities

- **Product search:** Find products by name, category, price range, specifications, and availability.
- **Category search:** Retrieve product categories and their descriptions.
- **Order tracking:** Check order status, payment methods, delivery estimates, and related information.
- **Customer information:** Retrieve customer data and identify their orders.
- **Promotions:** Search active promotions and identify eligible products.
- **Store policies:** Answer questions about payment methods, returns, exchanges, and delivery conditions using the internal policy manual.
- **Out-of-scope handling:** Politely redirect requests unrelated to the store's services.

## Architecture

The application follows a tool-based agent architecture, in which the LLM determines which tools are necessary to answer each request.

### Workflow

1. The user submits a question through the chat interface or command-line interface.
2. The LLM interprets the request and determines which tools are necessary.
3. The agent selects and invokes the appropriate tool or combination of tools.
4. The tools retrieve information from structured CSV data or the policy document through the RAG pipeline.
5. The retrieved information is returned to the agent.
6. The LLM generates a natural-language response based on the available information.

### Data retrieval

**Structured data — CSV files**

Operational information is retrieved through dedicated Python tools using Pandas.

Each tool is responsible for a specific domain, such as products, categories, customers, orders, order items, and promotions.

**Unstructured data — RAG**

Store policies are handled through a Retrieval-Augmented Generation pipeline:

- PDF document loading using `PyPDFLoader`.
- Text preprocessing and chunking using `RecursiveCharacterTextSplitter`.
- Embedding generation using Google's embedding model.
- Vector storage and similarity search using ChromaDB.
- Retrieval of relevant document passages to provide context to the agent.

The agent is instructed to ground policy-related answers in the retrieved documentation and avoid inventing information that is not supported by the source.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| LangChain | Agent orchestration and tool integration |
| Google Gemini | Language model |
| Google Generative AI Embeddings | Text embeddings |
| ChromaDB | Vector database |
| Pandas | CSV data processing |
| PyPDFLoader | PDF document loading |
| Streamlit | Chat user interface |
| Pytest | Automated testing |

## Project Structure

```text
.
├── agent/
│   ├── __init__.py
│   └── agent.py
├── tools/
│   ├── __init__.py
│   ├── products.py
│   ├── categories.py
│   ├── customers.py
│   ├── orders.py
│   ├── order_items.py
│   ├── promotions.py
│   └── helpers.py
├── rag/
│   └── inicialize_rag.py
├── data/
│   ├── *.csv
│   ├── politicas_da_loja.pdf
│   └── chroma_db/
├── tests/
├── app.py
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10 or higher.
- A Google AI API key with access to the selected Gemini models.

### 1. Clone the repository

```bash
git clone https://github.com/yasminaltino/emporio-musica-agent
cd emporio-musica-agent
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root directory:

```env
GOOGLE_API_KEY=your_api_key
```

The application loads environment variables from this file using `python-dotenv`.

Make sure `.env` is included in `.gitignore` to prevent committing sensitive credentials.

### 5. Run the assistant

**Command-line interface:**

```bash
python -m agent.agent
```

**Streamlit interface:**

```bash
streamlit run app.py
```

## Example Queries

The following examples illustrate the types of questions the assistant can handle.

| Category | Example |
|---|---|
| Products | Which acoustic guitars are available for up to R$700? |
| Categories | What products belong to the guitar category? |
| Promotions | Which products currently have active promotions? |
| Orders | What is the status of order #15? |
| Customers | Which orders were placed by a specific customer? |
| Payments | What payment methods does the store accept? |
| Returns | What is the store's return and exchange policy? |
| Delivery | What are the delivery timeframes for different shipping methods? |
| Out of scope | What is the weather forecast for tomorrow? |

## Tests

The project includes unit tests to validate the behavior of individual components and tools.

Run the tests with:

```bash
pytest -v
```

## Technical Decisions

The main technical decisions were guided by the challenge's scope, implementation simplicity, and the need for reliable information retrieval.

- **ReAct and LangChain:** The ReAct approach was chosen to allow the agent to dynamically select and combine tools according to each request. LangChain provides the abstractions needed for tool integration and agent orchestration.

- **No custom LangGraph workflow:** A custom LangGraph implementation was considered unnecessary for this prototype. The interaction flow is relatively straightforward, and LangChain's agent abstraction was sufficient without introducing additional orchestration complexity.

- **Structured data with Pandas:** CSV files are queried directly using Pandas rather than embeddings. This ensures precise filtering of values such as prices, stock quantities, and order statuses.

- **RAG for unstructured data:** The policy manual contains natural-language information, making semantic retrieval more appropriate than direct filtering. ChromaDB was selected as a lightweight vector database, while Google's embedding model enables similarity-based search.

- **Gemini:** Selected for its integration with LangChain and support for tool calling and natural-language generation.

- **Streamlit:** Chosen to provide a functional chat interface quickly, without requiring a separate frontend application.

- **Independent requests:** Conversation history is not automatically passed between requests. This avoids unrelated previous messages influencing tool selection and data retrieval, making responses more predictable.

## Limitations and Future Improvements

This is a prototype developed within the scope of a technical challenge. Some limitations and possible improvements include:

- **Retrieval quality:** Improve document preprocessing, chunking, and retrieval strategies to ensure that broad questions retrieve all relevant policy information with fewer tool calls.
- **Latency:** Reduce unnecessary LLM invocations and optimize multi-step agent execution.
- **Evaluation:** Expand the unit tests and introduce more systematic evaluation metrics for answer accuracy, grounding, and tool selection.
- **Conversation context:** Introduce controlled conversational memory for follow-up questions without allowing previous context to interfere with independent searches.
- **Error handling:** Improve resilience to API timeouts, rate limits, and external service failures.
- **Scalability:** Replace CSV-based retrieval with a database if the application needs to support larger datasets or concurrent requests.

## AI-Assisted Development

Copilot was used as a coding assistant, mainly to support documentation, debugging, code review, and questions about specific implementation approaches. Suggestions were reviewed and adapted to the project's context and requirements.
