# Running costs

Figures used for the demo's arithmetic examples.

## Vector database

A managed vector database tier is billed at 47 dollars per month per index.
Three indexes run in production and one in staging.

## Embedding compute

Batch embedding runs cost roughly 12 dollars per million chunks. Last quarter
8.5 million chunks were embedded across all corpora.

## Model calls

The free tier of the answering model allows 20 requests per day. An agentic
question costs 2 to 3 requests because each tool round-trip is its own call.
