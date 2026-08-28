# Opinion-Based Search for Product Reviews

An intelligent product review analysis and search system that identifies and interprets user opinions on specific aspects of smartphones, not just overall sentiment.

## Features

- **Aspect-Based Sentiment Analysis (ABSA)**: Extracts specific features (camera, battery, screen, etc.) and associated opinions from reviews
- **Semantic Search**: Search through reviews using natural language queries
- **ChatGPT-like Interface**: Simple, intuitive web interface for querying product reviews
- **Feature Inference**: Automatically identifies relevant aspects from user queries
- **Dockerized**: Fully containerized with Docker Compose for easy deployment

## Project Structure

```
.
├── preprocess_and_train.py    # Main preprocessing script
├── preprocessing/             # Preprocessing modules
│   ├── config.py
│   ├── dataset_loader.py
│   ├── text_processor.py
│   ├── aspect_extractor.py
│   └── elasticsearch_indexer.py
├── backend/                   # Backend API
│   ├── app.py                # Flask application
│   ├── config.py             # Configuration
│   ├── models/               # Data models
│   │   └── aspect_extractor.py
│   ├── services/             # Business logic
│   │   ├── search_service.py
│   │   └── fallback_search.py
│   └── requirements.txt
├── frontend/                  # React frontend
│   ├── src/
│   │   ├── App.js
│   │   ├── App.css
│   │   └── index.js
│   ├── package.json
│   └── nginx.conf
├── docker-compose.yml         # Docker Compose configuration
├── requirements.txt           # Python dependencies (for preprocessing)
└── README.md
```

## Quick Start with Docker (Recommended)

### Prerequisites

- Docker and Docker Compose installed
- At least 4GB of available RAM (for Elasticsearch)

### 1. Build and Start All Services

```bash
docker-compose up --build
```

This will:
- Start Elasticsearch
- Run preprocessing pipeline to extract and index data
- Start the backend API
- Start the frontend web app

### 2. Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5000
- **Elasticsearch**: http://localhost:9200

### 3. Stop Services

```bash
docker-compose down
```

To also remove volumes (including Elasticsearch data):

```bash
docker-compose down -v
```

## Manual Setup (Without Docker)

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Install Elasticsearch (Optional but Recommended)

Elasticsearch is used for efficient search. If you don't have it installed:

- **macOS**: `brew install elasticsearch` then `brew services start elasticsearch`
- **Linux**: Follow [official installation guide](https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html)
- **Windows**: Download from [Elasticsearch website](https://www.elastic.co/downloads/elasticsearch)

The system will work without Elasticsearch using fallback in-memory search, but performance will be limited.

### 3. Run Preprocessing Pipeline

```bash
python preprocess_and_train.py
```

This will:
- Load the Amazon Reviews dataset (Cell_Phones_and_Accessories)
- Extract aspects and opinions from reviews
- Index data in Elasticsearch (if available)
- Save extractions to `data/extractions.json` as backup

**Note**: The full dataset is very large. The script limits to 1000 reviews by default for testing. Adjust the `DATASET_LIMIT` environment variable as needed.

### 4. Start Backend API

```bash
cd backend
pip install -r requirements.txt
python -m backend.app
```

The API will be available at `http://localhost:5000`

### 5. Start Frontend

```bash
cd frontend
npm install
npm start
```

The web app will open at `http://localhost:3000`

## Usage

1. Open the web application in your browser (http://localhost:3000)
2. Type questions about smartphone features, for example:
   - "What do users say about the camera?"
   - "How is the battery life?"
   - "Tell me about the screen quality"
   - "What are opinions on the processor performance?"
3. View the results showing:
   - Summary of findings
   - Detailed review excerpts
   - Aspect and sentiment information

## Docker Services

The application consists of 4 Docker services:

1. **elasticsearch**: Search engine for indexing and querying reviews
2. **preprocessing**: Runs the data preprocessing pipeline (runs once on startup)
3. **backend**: Flask API server for handling search queries
4. **frontend**: React web application served via Nginx

### Environment Variables

You can customize the setup using environment variables in `docker-compose.yml`:

- `DATASET_LIMIT`: Number of reviews to process
- `ELASTICSEARCH_HOST`: Elasticsearch hostname (default: elasticsearch)
- `ELASTICSEARCH_PORT`: Elasticsearch port (default: 9200)
- `FLASK_PORT`: Backend API port (default: 5000)
- `CORS_ORIGINS`: Allowed CORS origins (comma-separated)

## API Endpoints

### POST `/api/search`
Search for reviews based on a query.

**Request:**
```json
{
  "query": "What do users say about the camera?"
}
```

**Response:**
```json
{
  "query": "What do users say about the camera?",
  "summary": "Found 5 relevant reviews. Most reviews about camera are positive (4/5).",
  "results": [
    {
      "product_id": "B08N5WRWNW",
      "aspect": "camera",
      "opinion": "amazing photos",
      "sentiment": "positive",
      "review_text": "The camera on this phone is amazing!..."
    }
  ],
  "count": 5
}
```

### GET `/api/health`
Check API health and Elasticsearch connection status.

## Technical Details

### Aspect Extraction

The system identifies the following aspects:
- Camera
- Battery
- Screen/Display
- Processor/Performance
- Design
- Sound/Audio
- Software/OS
- Price

### Sentiment Analysis

Each extracted aspect-opinion pair is classified as:
- **Positive**: Good, great, excellent, amazing, etc.
- **Negative**: Bad, poor, terrible, awful, etc.
- **Neutral**: Mixed or unclear sentiment

### Models Used

- **ABSA Model**: `yangheng/deberta-v3-base-absa-v1.1` (referenced, simplified implementation for demo)
- **Text Processing**: spaCy for preprocessing
- **Search**: Elasticsearch for indexing and retrieval

## Development

### Running Individual Services

You can run services individually for development:

```bash
# Start only Elasticsearch
docker-compose up elasticsearch

# Run preprocessing manually
docker-compose run --rm preprocessing

# Start backend only
docker-compose up backend

# Start frontend only
docker-compose up frontend
```

### Rebuilding Services

After code changes, rebuild specific services:

```bash
# Rebuild backend
docker-compose build backend
docker-compose up backend

# Rebuild frontend
docker-compose build frontend
docker-compose up frontend
```

### Viewing Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f preprocessing
```

## Limitations

- The current implementation uses simplified keyword-based extraction as a fallback
- Full ABSA model integration requires additional setup and model fine-tuning
- Dataset size is limited for demo purposes
- Elasticsearch is optional but recommended for production use

## Future Improvements

- Full integration of ABSA models from Hugging Face
- Feature inference mechanism for composite aspects
- Support for more product categories
- Enhanced semantic search with embeddings
- Multi-language support

## License

This project is for educational purposes.

