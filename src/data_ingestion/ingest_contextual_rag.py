#!/usr/bin/env python3
"""
Optimized Contextual RAG ingestion:
 - batched embeddings
 - fast bulk DB insertion (psycopg2.execute_values when available)
 - detect & reconcile vector dimension mismatch (recreate table when necessary)
 - robust embedder adapter resolution
Save to: src/data_ingestion/ingest_contextual_rag.py
"""

from __future__ import annotations
import os
import sys
import json
import logging
import copy
import time
from datetime import datetime
from typing import List, Iterable

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import TextNode, Document

import openai

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------- Config (tweak these) ----------
MD_DIR = os.getenv("MD_DIR", "data/raw")

DB_HOST = os.getenv("DB_HOST", os.getenv("DATABASE_HOST", "db"))
DB_PORT = int(os.getenv("DB_PORT", os.getenv("DATABASE_PORT", "5432")))
DB_NAME = os.getenv("DB_NAME", "rag_db")
DB_USER = os.getenv("DB_USER", "rag_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "rag_password")
DATABASE_URL = os.getenv("DATABASE_URL", f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

TABLE_NAME = "doc_md_contextual_20250830"
TABLE_FULLNAME = f"data_{TABLE_NAME}"

# OpenAI / embeddings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "128"))
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))

# Operational
BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", "32"))  # batch size for embeddings & DB inserts
NON_DESTRUCTIVE = os.getenv("INGEST_NON_DESTRUCTIVE", "false").lower() in ("1", "true", "yes")

# set openai key if provided
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# ---------- helpers ----------
def _retry_backoff(fn, attempts: int = 4, base: float = 1.5):
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            if i + 1 == attempts:
                raise
            wait = base ** i
            logger.warning("Transient error (attempt %d/%d): %s — retrying in %.1fs", i+1, attempts, e, wait)
            time.sleep(wait)

# Chat completion wrapper that prefers the new client shape but falls back to older API.
def chat_completion_text(model: str, messages: List[dict], max_tokens: int, temperature: float) -> str:
    # try new client style
    NewClient = getattr(openai, "OpenAI", None)
    if NewClient:
        try:
            client = NewClient()
            resp = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)
            # try to read the common shapes
            try:
                return resp.choices[0].message["content"].strip()
            except Exception:
                return resp.choices[0].message.content.strip()
        except Exception:
            # fall through to old api
            pass
    # old fallback
    resp = openai.ChatCompletion.create(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)
    return resp["choices"][0]["message"]["content"].strip()

def chunked_iter(seq: List, n: int) -> Iterable[List]:
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

# ---------- embedding adapter resolution ----------
def resolve_embedder() -> object:
    # If Settings.embed_model is preconfigured, use it.
    if getattr(Settings, "embed_model", None):
        logger.info("Using preconfigured Settings.embed_model")
        return Settings.embed_model

    # Try to import runtime BaseEmbedding class so our adapter will be isinstance-compatible
    runtime_base = None
    for candidate in ("llama_index.core.embeddings.base", "llama_index.embeddings.base"):
        try:
            m = __import__(candidate, fromlist=["BaseEmbedding"])
            runtime_base = getattr(m, "BaseEmbedding")
            logger.debug("Found BaseEmbedding in %s", candidate)
            break
        except Exception:
            continue

    # Build an OpenAI adapter that subclasses runtime_base if available
    class _OpenAIAdapter:
        def __init__(self, model_name: str = None):
            self.model_name = model_name or OPENAI_EMBEDDING_MODEL
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            if not isinstance(texts, (list, tuple)):
                texts = [texts]
            def call():
                resp = openai.Embedding.create(model=self.model_name, input=texts)
                return [item["embedding"] for item in resp["data"]]
            return _retry_backoff(call, attempts=3, base=2.0)
        def embed_query(self, text: str) -> List[float]:
            def call():
                resp = openai.Embedding.create(model=self.model_name, input=text)
                return resp["data"][0]["embedding"]
            return _retry_backoff(call, attempts=3, base=2.0)

    if runtime_base:
        # create a subclass at runtime
        AdapterClass = type("OpenAIEmbeddingRuntimeAdapter", (runtime_base, _OpenAIAdapter), {})
        return AdapterClass()
    else:
        return _OpenAIAdapter()

# ---------- document/context processing ----------
def generate_chunk_context_openai(chunk_text: str, max_lines: int = 2) -> str:
    prompt = f"""Summarize this text in {max_lines} short lines that capture the key information and intent.

Text:
{chunk_text}

Summary ({max_lines} lines only):"""
    def call():
        content = chat_completion_text(model=os.getenv("OPENAI_CHAT_MODEL", OPENAI_MODEL),
                                      messages=[{"role": "user", "content": prompt}],
                                      max_tokens=OPENAI_MAX_TOKENS,
                                      temperature=OPENAI_TEMPERATURE)
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        return "\n".join(lines[:max_lines]) if lines else ""
    try:
        return _retry_backoff(call, attempts=3, base=2.0)
    except Exception as e:
        logger.warning("OpenAI context generation failed: %s", e)
        return "Document chunk containing policy text.\nRelevant for organizational procedures."

def extract_page_number_from_text(text: str, chunk_index: int) -> int:
    return max(1, (chunk_index * 800) // 2000 + 1)

def create_contextual_nodes(nodes: List[TextNode], whole_document: str) -> List[TextNode]:
    logger.info("Creating contextual nodes for %d chunks...", len(nodes))
    enhanced_nodes = []
    for i, node in enumerate(nodes):
        enhanced_node = copy.deepcopy(node)
        context = None
        if OPENAI_API_KEY:
            try:
                context = generate_chunk_context_openai(node.text, max_lines=2)
            except Exception as e:
                logger.warning("Context generation failed for node %d: %s", i, e)
                context = None
        if not context:
            context = f"Part of {node.metadata.get('file_name', 'document')}"
        enhanced_node.metadata = dict(enhanced_node.metadata or {})
        enhanced_node.metadata["context"] = context
        enhanced_node.metadata["page_number"] = extract_page_number_from_text(node.text, i)
        enhanced_nodes.append(enhanced_node)
        if (i + 1) % 50 == 0:
            logger.info("Generated context for %d/%d nodes", i+1, len(nodes))
    logger.info("✅ Created %d contextual nodes", len(enhanced_nodes))
    return enhanced_nodes

# ---------- DB helpers ----------
def make_engine(url: str) -> Engine:
    return create_engine(url, pool_pre_ping=True)

def ensure_table_with_dim(engine: Engine, table: str, dim: int, drop_if_mismatch: bool = True) -> None:
    """Ensure public.{table} exists and embedding is vector(dim). If exists with mismatch:
       - if drop_if_mismatch True -> drop & recreate
       - else raise RuntimeError
    """
    with engine.begin() as conn:
        # detect existence
        exists = bool(conn.execute(text("SELECT to_regclass(:name)"), {"name": f"public.{table}"}).fetchone()[0])
        if exists:
            # try to inspect pg_type/format_type for embedding column
            existing_dim = None
            try:
                q = text("""
                    SELECT pg_catalog.col_description(c.oid, a.attnum) as col_comment,
                           format_type(a.atttypid, a.atttypmod) as type_repr
                    FROM pg_attribute a
                    JOIN pg_class c ON a.attrelid = c.oid
                    WHERE c.relname = :tbl AND a.attname = 'embedding' AND a.attnum > 0;
                """)
                r = conn.execute(q, {"tbl": table}).fetchone()
                if r and r["type_repr"]:
                    import re
                    m = re.search(r"vector\s*\(\s*(\d+)\s*\)", r["type_repr"])
                    if m:
                        existing_dim = int(m.group(1))
            except Exception:
                existing_dim = None

            if existing_dim is None:
                # fallback: try to query the column definition via information_schema (best-effort)
                try:
                    q2 = text("""
                        SELECT udt_name
                        FROM information_schema.columns
                        WHERE table_schema = 'public' AND table_name = :tbl AND column_name = 'embedding'
                    """)
                    r2 = conn.execute(q2, {"tbl": table}).fetchone()
                    if r2 and r2[0] and 'vector' in r2[0]:
                        # may not contain dimension
                        existing_dim = None
                except Exception:
                    existing_dim = None

            if existing_dim is not None and existing_dim != dim:
                msg = f"Existing table public.{table} has vector dim={existing_dim} but model emits {dim}"
                if not drop_if_mismatch:
                    raise RuntimeError(msg)
                logger.warning(msg + " — dropping & recreating table.")
                conn.execute(text(f"DROP TABLE IF EXISTS public.{table} CASCADE"))
                exists = False
            elif existing_dim is None:
                # unknown existing dim: to be safe, drop & recreate
                if not drop_if_mismatch:
                    raise RuntimeError("Existing table found but could not determine embedding dimension; enable drop_if_mismatch to allow recreate")
                logger.info("Existing table found but dimension unknown — dropping & recreating to be safe.")
                conn.execute(text(f"DROP TABLE IF EXISTS public.{table} CASCADE"))
                exists = False
            else:
                logger.info("Existing table public.%s already matches embedding dim=%d", table, dim)

        if not exists:
            create_sql = f"""
            CREATE TABLE public.{table} (
                id bigserial PRIMARY KEY,
                doc_id text,
                text_content text,
                metadata jsonb,
                embedding vector({dim}),
                created_at timestamptz DEFAULT now()
            );
            CREATE INDEX IF NOT EXISTS {table}_embedding_idx ON public.{table} USING ivfflat (embedding);
            """
            conn.execute(text(create_sql))
            logger.info("✅ Ensured table public.%s exists (vector dim=%d)", table, dim)

# ---------- Main ----------
def main():
    logger.info("=== Contextual RAG ingest starting ===")
    # DB connection check
    engine = make_engine(DATABASE_URL)
    try:
        with engine.connect() as conn:
            version = conn.execute(text("SELECT version()")).fetchone()[0]
            logger.info("Connected to DB: %s", version)
            has_vec = conn.execute(text("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')")).fetchone()[0]
            if not has_vec:
                logger.error("pgvector extension not found in DB — aborting.")
                return
    except Exception as e:
        logger.exception("Database connection failed: %s", e)
        return

    # Load documents
    if not os.path.exists(MD_DIR):
        logger.error("MD_DIR not found: %s", MD_DIR)
        return
    reader = SimpleDirectoryReader(input_dir=MD_DIR, recursive=True)
    documents: List[Document] = reader.load_data()
    if not documents:
        logger.error("No documents loaded from %s", MD_DIR)
        return
    logger.info("Loaded %d documents", len(documents))

    # Build pipeline & nodes
    pipeline = IngestionPipeline(transformations=[
        MarkdownNodeParser(include_metadata=True),
        TokenTextSplitter(chunk_size=800, chunk_overlap=200),
    ])
    all_nodes: List[TextNode] = []
    for d in documents:
        if hasattr(pipeline, "create_nodes_from_document"):
            nodes = pipeline.create_nodes_from_document(d)
        elif hasattr(pipeline, "create_nodes"):
            nodes = pipeline.create_nodes(d)
        else:
            # fallback
            from llama_index.core.schema import TextNode as LITextNode
            nodes = [LITextNode(text=d.text, metadata=d.metadata)]
        all_nodes.extend(nodes)
    logger.info("Split into %d nodes", len(all_nodes))

    # Generate contextual metadata
    enhanced_nodes = create_contextual_nodes(all_nodes, whole_document=" ".join([d.text for d in documents])[:8000])
    if not enhanced_nodes:
        logger.info("No nodes after contextualization — exiting.")
        return

    # Resolve embedder
    embedder = resolve_embedder()

    # batched embedding compute helper
    def compute_embeddings_for_texts(texts: List[str]) -> List[List[float]]:
        # prefer adapter embed_documents if available
        if hasattr(embedder, "embed_documents"):
            return embedder.embed_documents(texts)
        # fallback to per-text embed_query/get_text_embedding
        out = []
        for t in texts:
            if hasattr(embedder, "embed_query"):
                out.append(embedder.embed_query(t))
            elif hasattr(embedder, "get_text_embedding"):
                out.append(embedder.get_text_embedding(t))
            else:
                resp = openai.Embedding.create(model=OPENAI_EMBEDDING_MODEL, input=t)
                out.append(resp["data"][0]["embedding"])
        return out

    # Compute first batch to detect dimension
    first_batch_texts = [n.text for n in enhanced_nodes[:BATCH_SIZE]]
    logger.info("Computing embeddings for initial batch (size=%d)...", len(first_batch_texts))
    first_embeddings = []
    for batch in chunked_iter(first_batch_texts, BATCH_SIZE):
        first_embeddings.extend(compute_embeddings_for_texts(batch))
    if not first_embeddings or not isinstance(first_embeddings[0], (list, tuple)):
        logger.error("Failed to compute embeddings for the first batch")
        return
    detected_dim = len(first_embeddings[0])
    logger.info("Detected embedding dimension: %d", detected_dim)

    # Ensure DB table (and reconcile dim mismatch)
    try:
        ensure_table_with_dim(engine, TABLE_FULLNAME, detected_dim, drop_if_mismatch=not NON_DESTRUCTIVE)
    except Exception as e:
        logger.exception("Failed ensuring table with right dimension: %s", e)
        return

    # Try to import psycopg2 for fast bulk insert
    use_psycopg2 = False
    try:
        import psycopg2
        import psycopg2.extras as pg_extras
        use_psycopg2 = True
        logger.info("psycopg2 available: will use fast bulk insert (execute_values).")
    except Exception:
        logger.info("psycopg2 not available: falling back to SQLAlchemy batch inserts (slower).")

    total_inserted = 0

    if use_psycopg2:
        # bulk insert using psycopg2.execute_values with template that casts embedding to vector
        conn_info = {
            "dbname": DB_NAME,
            "user": DB_USER,
            "password": DB_PASSWORD,
            "host": DB_HOST,
            "port": DB_PORT,
        }
        try:
            pg_conn = psycopg2.connect(**conn_info)
            pg_cur = pg_conn.cursor()
            from psycopg2.extras import execute_values

            # function to produce rows
            def rows_for_batch(nodes_batch, embeddings_batch):
                rows = []
                for node, emb in zip(nodes_batch, embeddings_batch):
                    metadata_json = json.dumps(node.metadata or {})
                    emb_text = '[' + ','.join(map(str, emb)) + ']'
                    doc_id_val = (node.metadata or {}).get("doc_id")
                    rows.append((doc_id_val, node.text, metadata_json, emb_text))
                return rows

            # first batch already computed
            start_idx = 0
            batch_nodes = enhanced_nodes[:len(first_batch_texts)]
            rows = rows_for_batch(batch_nodes, first_embeddings)
            template = "(%s,%s,%s,CAST(%s AS vector))"
            execute_values(pg_cur,
                           f"INSERT INTO public.{TABLE_FULLNAME} (doc_id, text_content, metadata, embedding) VALUES %s",
                           rows, template=template)
            pg_conn.commit()
            total_inserted += len(rows)
            logger.info("Inserted initial batch of %d rows (psycopg2).", len(rows))

            # remaining batches
            for start in range(len(first_batch_texts), len(enhanced_nodes), BATCH_SIZE):
                batch_nodes = enhanced_nodes[start:start+BATCH_SIZE]
                texts = [n.text for n in batch_nodes]
                embeddings = []
                for b in chunked_iter(texts, BATCH_SIZE):
                    embeddings.extend(compute_embeddings_for_texts(b))
                rows = rows_for_batch(batch_nodes, embeddings)
                execute_values(pg_cur,
                               f"INSERT INTO public.{TABLE_FULLNAME} (doc_id, text_content, metadata, embedding) VALUES %s",
                               rows, template=template)
                pg_conn.commit()
                total_inserted += len(rows)
                logger.info("Inserted batch starting %d (%d rows).", start, len(rows))

            pg_cur.close()
            pg_conn.close()
            logger.info("✅ Inserted %d rows into public.%s", total_inserted, TABLE_FULLNAME)
        except Exception as e:
            logger.exception("psycopg2 bulk insert failed: %s", e)
            # fallback to SQLAlchemy below
    if not use_psycopg2 or total_inserted == 0:
        # Use SQLAlchemy insertion per-batch (parameterized + CAST)
        with engine.begin() as conn:
            # insert first batch (if not done)
            def insert_batch_sqlalchemy(nodes_batch, embeddings_batch):
                insert_sql = text(
                    f"INSERT INTO public.{TABLE_FULLNAME} (doc_id, text_content, metadata, embedding) "
                    f"VALUES (:doc_id, :text_content, :metadata, CAST(:embedding AS vector))"
                )
                for node, emb in zip(nodes_batch, embeddings_batch):
                    metadata_json = json.dumps(node.metadata or {})
                    emb_text = '[' + ','.join(map(str, emb)) + ']'
                    doc_id_val = (node.metadata or {}).get('doc_id')
                    conn.execute(insert_sql, {"doc_id": doc_id_val, "text_content": node.text, "metadata": metadata_json, "embedding": emb_text})

            # if first batch wasn't inserted by psycopg2, insert it now
            if total_inserted == 0:
                insert_batch_sqlalchemy(enhanced_nodes[:len(first_batch_texts)], first_embeddings)
                total_inserted += len(first_batch_texts)
                logger.info("Inserted initial batch of %d rows (SQLAlchemy).", len(first_batch_texts))

            for start in range(len(first_batch_texts), len(enhanced_nodes), BATCH_SIZE):
                batch_nodes = enhanced_nodes[start:start+BATCH_SIZE]
                texts = [n.text for n in batch_nodes]
                embeddings = []
                for b in chunked_iter(texts, BATCH_SIZE):
                    embeddings.extend(compute_embeddings_for_texts(b))
                insert_batch_sqlalchemy(batch_nodes, embeddings)
                total_inserted += len(batch_nodes)
                logger.info("Inserted batch starting %d (%d rows).", start, len(batch_nodes))

        logger.info("✅ Inserted %d rows into public.%s (SQLAlchemy path)", total_inserted, TABLE_FULLNAME)

    logger.info("=== Indexing finished successfully ===")

if __name__ == "__main__":
    main()
