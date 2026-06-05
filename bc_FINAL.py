#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import použitých knižníc pre prácu s databázou, dátami, embeddingami, streamovým učením, vizualizáciou a LDA analýzou.
import sys
get_ipython().system('{sys.executable} -m pip install -U river -U gensim')

import sqlite3
import pandas as pd
import numpy as np
import re
import random
import gensim.downloader as api
import matplotlib.pyplot as plt

from collections import deque, Counter
from datetime import datetime, UTC

from river import metrics, drift, forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


# Základné konfiguračné parametre experimentu:
# cesta k databáze, cesta k labelom, filtrované roky, horný limit počtu dokumentov za mesiac, seed a názov embedding modelu.
DBPATH = "nela-gt-2021.db"
LABELSPATH = "labels.csv"
YEARS = {"2021", "2022"}
MONTHCAPK = 10000
RANDOM_SEED = 42
EMBEDDING_MODEL = "glove-wiki-gigaword-100"


# In[3]:


# Načítanie databázy a labelov.
# V tejto časti sa vytvorí mapovanie zdrojov na triedy a zobrazia sa základné informácie o dátach.
conn = sqlite3.connect(DBPATH)
cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM newsdata")
print("newsdata:", cursor.fetchone()[0])

cursor.execute("SELECT COUNT(*) FROM tweet")
print("tweet:", cursor.fetchone()[0])

labels = pd.read_csv(LABELSPATH)
label_map = dict(zip(labels["source"], labels["label"]))

print(labels["label"].value_counts())
print("Pocet zdrojov v labels:", len(label_map))
display(labels.head())


# In[4]:


# Predspracovanie textu:
# odstránenie URL adries, HTML značiek a nadbytočných medzier, plus príprava tokenizácie pre embedding reprezentáciu.
url_re = re.compile(r"https?://\S+|www\.\S+")
html_re = re.compile(r"<.*?>")
ws_re = re.compile(r"\s+")
token_re = re.compile(r"[a-zA-Z]+")

def clean_text(s: str) -> str:
    s = s or ""
    s = s.lower()
    s = url_re.sub(" ", s)
    s = html_re.sub(" ", s)
    s = ws_re.sub(" ", s).strip()
    return s

def make_text(content):
    return clean_text(content or "")

def tokenize_for_embeddings(text):
    if not isinstance(text, str) or not text.strip():
        return []
    return token_re.findall(text.lower())


# In[5]:


# Streamovanie článkov z databázy v časovom poradí.
# Každý dokument sa prevedie na text, s časovou značkou a binárnou triedu. 
# Binárna trieda vznikla spojením nedôveryhodných zdrojov 1 a 2 do 1, vs dôveryhodné zdroje 0
def stream_articles(conn, label_map, years=None, batch=5000):
    cur = conn.cursor()
    cur.execute("""
        SELECT published_utc, source, content
        FROM newsdata
        WHERE published_utc IS NOT NULL
        ORDER BY published_utc
    """)

    while True:
        rows = cur.fetchmany(batch)
        if not rows:
            break

        for ts, source, content in rows:
            lab = label_map.get(source)
            if lab is None:
                continue

            year = pd.to_datetime(ts, unit="s").strftime("%Y")
            if years is not None and year not in years:
                continue

            y = 1 if int(lab) > 0 else 0
            yield {"text": make_text(content), "ts": ts, "y": y}


# In[6]:


# Výber dokumentov po mesiacoch s limitom na maximálny počet vzoriek.
# Pri väčšom počte článkov sa zachová približný pomer tried 0 a 1.
def stream_per_month(conn, label_map, years=None, k=10000, batch=5000, seed=42):
    rng = random.Random(seed)
    cur = conn.cursor()

    cur.execute("""
        SELECT rowid, published_utc, source
        FROM newsdata
        WHERE published_utc IS NOT NULL
        ORDER BY published_utc
    """)

    month_rows = {}

    while True:
        rows = cur.fetchmany(batch)
        if not rows:
            break

        for rowid, ts, source in rows:
            lab = label_map.get(source)
            if lab is None:
                continue

            dt = pd.to_datetime(ts, unit="s")
            year = dt.strftime("%Y")
            if years is not None and year not in years:
                continue

            y = 1 if int(lab) > 0 else 0
            ym = dt.strftime("%Y-%m")
            month_rows.setdefault(ym, []).append((rowid, ts, source, y))

    selected_rows = []

    for ym, rows in month_rows.items():
        if len(rows) <= k:
            selected_rows.extend(rows)
            continue

        rows_0 = [r for r in rows if r[3] == 0]
        rows_1 = [r for r in rows if r[3] == 1]

        total = len(rows)
        target_0 = round(k * len(rows_0) / total)
        target_1 = k - target_0

        sel_0 = rows_0 if len(rows_0) <= target_0 else rng.sample(rows_0, target_0)
        sel_1 = rows_1 if len(rows_1) <= target_1 else rng.sample(rows_1, target_1)

        current = sel_0 + sel_1
        remaining = k - len(current)

        if remaining > 0:
            leftover_0 = [r for r in rows_0 if r not in sel_0]
            leftover_1 = [r for r in rows_1 if r not in sel_1]
            leftovers = leftover_0 + leftover_1

            if len(leftovers) <= remaining:
                current.extend(leftovers)
            else:
                current.extend(rng.sample(leftovers, remaining))

        selected_rows.extend(current)

    selected_rows = sorted(selected_rows, key=lambda r: r[1])

    for rowid, ts, source, y in selected_rows:
        cur.execute("SELECT content FROM newsdata WHERE rowid = ?", (rowid,))
        content = cur.fetchone()[0]
        yield {"text": make_text(content), "ts": ts, "y": y}


# In[7]:


# Načítanie predtrénovaného embedding modelu GloVe a zistenie rozmeru výsledných vektorov.
pretrained_wv = api.load(EMBEDDING_MODEL)
EMBED_DIM = pretrained_wv.vector_size

print("Loaded pretrained model:", EMBEDDING_MODEL)
print("Vector size:", EMBED_DIM)


# In[8]:


# Prevod textu na dokumentový embedding pomocou priemeru embeddingov jednotlivých slov a následná transformácia do slovníkového vstupu pre model.
def text_to_doc_vector(text, keyed_vectors):
    tokens = tokenize_for_embeddings(text)
    vecs = [keyed_vectors[w] for w in tokens if w in keyed_vectors]
    if not vecs:
        return None
    return np.mean(vecs, axis=0).astype(np.float32)

def vector_to_feature_dict(vec):
    return {f"f{i}": float(v) for i, v in enumerate(vec)}


# In[71]:


# Definícia dvoch klasifikačných modelov: adaptívny Random Forest a statický Random Forest
def make_online_embedding_model():
    return forest.ARFClassifier(
        n_models=50,
        seed=RANDOM_SEED
    )

def make_static_embedding_model():
    return RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )


# In[10]:


# Materializácia streamu do pamäte v podobe pripravených vektorov, aby sa ten istý dátový tok dal použiť v oboch režimoch experimentu.
def materialize_stream_vectors(conn, label_map, years, batch=5000, monthcap_k=10000):
    rows_out = []

    iterator = stream_per_month(
        conn,
        label_map,
        years=years,
        k=monthcap_k,
        batch=batch,
        seed=RANDOM_SEED
    )

    for row in iterator:
        text, ts, y = row["text"], row["ts"], row["y"]
        vec = text_to_doc_vector(text, pretrained_wv)
        if vec is None:
            continue

        rows_out.append({
            "text": text,
            "ts": ts,
            "y": y,
            "vec": vec,
            "x_dict": vector_to_feature_dict(vec)
        })

    return rows_out


# In[70]:


# Nastavenie podielu trénovacej časti pre statický model a príprava dátového toku pre ďalšie experimenty.
STATIC_TRAIN_RATIO = 0.07

stream_rows = materialize_stream_vectors(
    conn=conn,
    label_map=label_map,
    years=YEARS,
    batch=5000,
    monthcap_k=MONTHCAPK
)

print("Pocet pouzitelnych dokumentov vo streame:", len(stream_rows))
print("Trénovacia časť pre statický model (7%):", int(len(stream_rows) * STATIC_TRAIN_RATIO))


# In[56]:


# Hlavná experimentálna funkcia, ktorá spustí adaptívny alebo statický režim,
# priebežne vyhodnocuje metriky a zaznamenáva detegované drifty.
def run_experiment(
    stream_rows,
    mode="adaptive",
    delta=0.002,
    metric_log_every=500,
    static_train_ratio=0.10,
    track_drifts=True,
):
    if mode not in {"adaptive", "static"}:
        raise ValueError(f"Unknown mode: {mode}")

    n_samples_total = len(stream_rows)
    if n_samples_total == 0:
        raise ValueError("stream_rows is empty.")

    warmup_size = int(n_samples_total * static_train_ratio)
    if warmup_size <= 0:
        raise ValueError("Warmup size must be > 0.")
    if warmup_size >= n_samples_total:
        raise ValueError("Warmup size must be smaller than total number of samples.")

    eval_start_idx = warmup_size

    adaptive_model = make_online_embedding_model() if mode == "adaptive" else None
    static_model = make_static_embedding_model() if mode == "static" else None

    detector = None
    if track_drifts:
        detector = drift.ADWIN(
            delta=delta,
        )

    metric_acc = metrics.Accuracy()
    metric_f1 = metrics.F1()
    metric_precision = metrics.Precision()
    metric_recall = metrics.Recall()

    drift_events = []
    metric_history = []

    if mode == "static":
        X_warm = np.array([row["vec"] for row in stream_rows[:warmup_size]])
        y_warm = np.array([row["y"] for row in stream_rows[:warmup_size]])
        static_model.fit(X_warm, y_warm)

    for global_i, row in enumerate(stream_rows):
        y = row["y"]
        ts = row["ts"]
        x_dict = row["x_dict"]
        vec = row["vec"]

        if mode == "adaptive":
            y_pred = adaptive_model.predict_one(x_dict)
            if y_pred is None:
                y_pred = 0
        else:
            if global_i < warmup_size:
                continue
            y_pred = int(static_model.predict(vec.reshape(1, -1))[0])

        if global_i >= eval_start_idx:
            metric_acc.update(y, y_pred)
            metric_f1.update(y, y_pred)
            metric_precision.update(y, y_pred)
            metric_recall.update(y, y_pred)

            err = int(y_pred != y)

            if detector is not None:
                detector.update(err)

                if detector.drift_detected:
                    drift_events.append({
                        "mode": mode,
                        "drift_at_sample": global_i,
                        "drift_date_utc": datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%d"),
                        "error_rate_proxy": err,
                        "accuracy_now": metric_acc.get(),
                        "f1_now": metric_f1.get(),
                        "precision_now": metric_precision.get(),
                        "recall_now": metric_recall.get(),
                    })

                    detector = drift.ADWIN(
                        delta=delta,
                    )

            if ((global_i - eval_start_idx) % metric_log_every) == 0:
                metric_history.append({
                    "sample_idx": global_i,
                    "mode": mode,
                    "accuracy": metric_acc.get(),
                    "f1": metric_f1.get(),
                    "precision": metric_precision.get(),
                    "recall": metric_recall.get(),
                })

        if mode == "adaptive":
            adaptive_model.learn_one(x_dict, y)

    summary_df = pd.DataFrame([{
        "model_name": "AdaptiveRF" if mode == "adaptive" else "StaticRF",
        "mode": mode,
        "delta": delta,
        "monthcap_k": MONTHCAPK,
        "static_train_ratio": static_train_ratio,
        "eval_start_idx": eval_start_idx,
        "final_accuracy": metric_acc.get(),
        "final_f1": metric_f1.get(),
        "final_precision": metric_precision.get(),
        "final_recall": metric_recall.get(),
        "n_drifts": len(drift_events),
        "n_samples_total": n_samples_total,
        "n_samples_evaluated": n_samples_total - eval_start_idx,
    }])

    return summary_df, pd.DataFrame(drift_events), pd.DataFrame(metric_history)


# In[72]:


# Spustenie experimentu pre AdaptiveRF a StaticRF s rovnakými parametrami a následné porovnanie výsledkov.
DRIFT_DELTA = 0.002
METRIC_LOG_EVERY = 500

adaptive_summary, adaptive_events, adaptive_metric_history = run_experiment(
    stream_rows=stream_rows,
    mode="adaptive",
    delta=DRIFT_DELTA,
    metric_log_every=METRIC_LOG_EVERY,
    static_train_ratio=STATIC_TRAIN_RATIO,
    track_drifts=True,
)

static_summary, static_events, static_metric_history = run_experiment(
    stream_rows=stream_rows,
    mode="static",
    delta=DRIFT_DELTA,
    metric_log_every=METRIC_LOG_EVERY,
    static_train_ratio=STATIC_TRAIN_RATIO,
    track_drifts=True,
)

summary_compare = pd.concat([adaptive_summary, static_summary], ignore_index=True)

display(summary_compare)
display(adaptive_events.head())
display(static_events.head())


# In[81]:


# Vizualizácia - Priame porovnanie vybranej metriky v čase medzi adaptívnym a statickým modelom vrátane vyznačených driftov.
# Rozpätie hodnôt ylim meníme podľa potreby
def plot_metric_comparison(adaptive_metric_df, static_metric_df,
                           adaptive_events, static_events,
                           metric_name, title):
    adf = adaptive_metric_df[adaptive_metric_df["mode"] == "adaptive"].copy()
    sdf = static_metric_df[static_metric_df["mode"] == "static"].copy()

    aev = adaptive_events[adaptive_events["mode"] == "adaptive"].copy()
    sev = static_events[static_events["mode"] == "static"].copy()

    ylabel_map = {
        "f1": "F1-score",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall"
    }

    plt.figure(figsize=(14, 6))

    plt.plot(
        adf["sample_idx"], adf[metric_name],
        label="Adaptive Random Forest",
        color="tab:blue",
        linewidth=2.2
    )
    plt.plot(
        sdf["sample_idx"], sdf[metric_name],
        label="Static Random Forest",
        color="tab:orange",
        linewidth=2.2
    )

    for i, (_, row) in enumerate(aev.iterrows()):
        plt.axvline(
            row["drift_at_sample"],
            color="tab:red",
            linestyle="--",
            alpha=0.70,
            label="Adaptive RF drift" if i == 0 else None
        )

    for i, (_, row) in enumerate(sev.iterrows()):
        plt.axvline(
            row["drift_at_sample"],
            color="tab:green",
            linestyle="--",
            alpha=0.70,
            label="Static RF drift" if i == 0 else None
        )

    plt.title(title)
    plt.xlabel("Sample index")
    plt.ylabel(ylabel_map.get(metric_name, metric_name))
    # Rozpätie hodnôt meníme podľa potreby
    plt.ylim(0.88, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# In[82]:


# Porovnanie precision a recall medzi modelmi v čase.
plot_metric_comparison(
    adaptive_metric_history,
    static_metric_history,
    adaptive_events,
    static_events,
    metric_name="precision",
    title="Porovnanie precision v čase"
)

plot_metric_comparison(
    adaptive_metric_history,
    static_metric_history,
    adaptive_events,
    static_events,
    metric_name="recall",
    title="Porovnanie recall v čase"
)


# In[78]:


# Porovnanie F1 skóre a accuracy medzi modelmi v čase.
plot_metric_comparison(
    adaptive_metric_history,
    static_metric_history,
    adaptive_events,
    static_events,
    metric_name="f1",
    title="Porovnanie F1 skóre v čase"
)

plot_metric_comparison(
    adaptive_metric_history,
    static_metric_history,
    adaptive_events,
    static_events,
    metric_name="accuracy",
    title="Porovnanie accuracy v čase"
)


# In[83]:


# Výpočet zmeny vybranej metriky pred a po detegovanom drifte.
# Slúži na identifikáciu driftu vhodného na hlbšiu analýzu.
def compute_drift_drop_table(metric_history_df, events_df, mode, metric_name="f1"):
    mh = metric_history_df[metric_history_df["mode"] == mode].sort_values("sample_idx").reset_index(drop=True)
    ev = events_df[events_df["mode"] == mode].sort_values("drift_at_sample").reset_index(drop=True)

    rows = []
    for _, event in ev.iterrows():
        drift_idx = event["drift_at_sample"]

        before = mh[mh["sample_idx"] < drift_idx]
        after = mh[mh["sample_idx"] >= drift_idx]

        if len(before) == 0 or len(after) == 0:
            continue

        before_metric = before.iloc[-1][metric_name]
        after_metric = after.iloc[0][metric_name]

        rows.append({
            "mode": mode,
            "drift_at_sample": drift_idx,
            "drift_date_utc": event["drift_date_utc"],
            "metric_before": before_metric,
            "metric_after": after_metric,
            "delta_metric": after_metric - before_metric
        })

    return pd.DataFrame(rows)


# In[84]:


# Výber validného driftu s dostatočne veľkým oknom pred aj po zmene.
def select_valid_drift(metric_history_df, events_df, mode, metric_name="f1", min_pre=10000, min_post=10000):
    drops = compute_drift_drop_table(metric_history_df, events_df, mode, metric_name)
    valid_rows = []

    drift_points = sorted(events_df[events_df["mode"] == mode]["drift_at_sample"].tolist())

    for _, row in drops.iterrows():
        drift_idx = int(row["drift_at_sample"])
        pos = drift_points.index(drift_idx)

        prev_drift = drift_points[pos - 1] if pos > 0 else int(len(stream_rows) * STATIC_TRAIN_RATIO)
        next_drift = drift_points[pos + 1] if pos < len(drift_points) - 1 else len(stream_rows)

        pre_size = drift_idx - prev_drift
        post_size = next_drift - drift_idx

        if pre_size >= min_pre and post_size >= min_post:
            row_dict = row.to_dict()
            row_dict["pre_size"] = pre_size
            row_dict["post_size"] = post_size
            valid_rows.append(row_dict)

    valid_df = pd.DataFrame(valid_rows)

    if len(valid_df) == 0:
        return pd.DataFrame()

    return valid_df.sort_values("delta_metric").head(1)


# In[85]:


# Výber konkrétneho driftu, ktorý bude použitý na tematickú analýzu.
selected_adaptive_drift = select_valid_drift(
    adaptive_metric_history, adaptive_events, mode="adaptive",
    metric_name="f1", min_pre=10000, min_post=10000
)

selected_static_drift = select_valid_drift(
    static_metric_history, static_events, mode="static",
    metric_name="f1", min_pre=10000, min_post=10000
)
display(selected_adaptive_drift)
display(selected_static_drift)


# In[86]:


# Zistenie hraníc okien pred a po zvolenom drifte.
def get_drift_boundaries(events_df, selected_drift_idx):
    drift_points = sorted(events_df["drift_at_sample"].tolist())
    pos = drift_points.index(selected_drift_idx)

    prev_drift = drift_points[pos - 1] if pos > 0 else None
    next_drift = drift_points[pos + 1] if pos < len(drift_points) - 1 else None

    return prev_drift, selected_drift_idx, next_drift


# In[87]:


# Zobrazenie indexov vybraného driftu a jeho susedných hraníc.
best_adaptive_drift_idx = int(selected_adaptive_drift.iloc[0]["drift_at_sample"])
best_static_drift_idx = int(selected_static_drift.iloc[0]["drift_at_sample"])

adaptive_prev, adaptive_sel, adaptive_next = get_drift_boundaries(adaptive_events, best_adaptive_drift_idx)
static_prev, static_sel, static_next = get_drift_boundaries(static_events, best_static_drift_idx)

print("ADAPTIVE boundaries:", adaptive_prev, adaptive_sel, adaptive_next)
print("STATIC boundaries:", static_prev, static_sel, static_next)


# In[88]:


# Zber textov z časového úseku pred driftom a po drifte pre adaptívny aj statický model.
def collect_between_drift_windows(stream_rows, start_idx, drift_idx, end_idx):
    pre_items = []
    post_items = []

    pre_start = 0 if start_idx is None else start_idx
    post_end = len(stream_rows) if end_idx is None else end_idx

    for i, row in enumerate(stream_rows):
        item = (row["text"], row["ts"], row["y"])

        if pre_start <= i < drift_idx:
            pre_items.append(item)
        elif drift_idx <= i < post_end:
            post_items.append(item)

    return {
        "pre": pre_items,
        "post": post_items
    }


# In[89]:


# Vytvorenie datasetu pred a po vybranom drifte.
adaptive_between_windows = collect_between_drift_windows(
    stream_rows=stream_rows,
    start_idx=adaptive_prev,
    drift_idx=adaptive_sel,
    end_idx=adaptive_next
)

static_between_windows = collect_between_drift_windows(
    stream_rows=stream_rows,
    start_idx=static_prev,
    drift_idx=static_sel,
    end_idx=static_next
)

print("ADAPTIVE pre/post:", len(adaptive_between_windows["pre"]), len(adaptive_between_windows["post"]))
print("STATIC pre/post:", len(static_between_windows["pre"]), len(static_between_windows["post"]))


# In[90]:


# Oddelenie textov a labelov z pripravených okien pre následné LDA modelovanie tém.
def extract_texts_and_labels(window_items):
    texts = []
    labels = []

    for text, ts, y in window_items:
        if isinstance(text, str) and text.strip():
            texts.append(text)
            labels.append(y)

    return texts, labels


# In[91]:


# Spustenie LDA modelu nad textami v danom časovom okne a výpočet dominantných tém a ich štatistík.
def run_lda_topics(texts, labels, n_topics=7, n_top_words=15, min_df=15, max_df=0.6):
    valid_pairs = [
        (t, y)
        for t, y in zip(texts, labels)
        if isinstance(t, str) and t.strip()
    ]

    if len(valid_pairs) == 0:
        return {
            "vectorizer": None,
            "lda": None,
            "topics": [],
            "doc_topic_matrix": None,
            "dominant_topics": None,
            "topic_stats": pd.DataFrame(),
            "ndocs": 0
        }

    texts_clean = [t for t, _ in valid_pairs]
    labels_clean = [y for _, y in valid_pairs]

    vectorizer = CountVectorizer(min_df=min_df, max_df=max_df)
    X = vectorizer.fit_transform(texts_clean)

    if X.shape[1] == 0:
        return {
            "vectorizer": vectorizer,
            "lda": None,
            "topics": [],
            "doc_topic_matrix": None,
            "dominant_topics": None,
            "topic_stats": pd.DataFrame(),
            "ndocs": len(texts_clean)
        }

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        learning_method="batch",
        max_iter=30
    )

    doc_topic_matrix = lda.fit_transform(X)
    dominant_topics = doc_topic_matrix.argmax(axis=1)
    feature_names = vectorizer.get_feature_names_out()

    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_indices]
        topics.append({
            "topic_id": topic_idx,
            "top_words": top_words
        })

    topic_stats = pd.DataFrame({
        "topic_id": dominant_topics,
        "y": labels_clean
    })

    topic_stats = (
        topic_stats.groupby(["topic_id", "y"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .rename(columns={0: "class_0_docs", 1: "class_1_docs"})
    )

    if "class_0_docs" not in topic_stats.columns:
        topic_stats["class_0_docs"] = 0
    if "class_1_docs" not in topic_stats.columns:
        topic_stats["class_1_docs"] = 0

    topic_stats["total_docs"] = topic_stats["class_0_docs"] + topic_stats["class_1_docs"]

    return {
        "vectorizer": vectorizer,
        "lda": lda,
        "topics": topics,
        "doc_topic_matrix": doc_topic_matrix,
        "dominant_topics": dominant_topics,
        "topic_stats": topic_stats,
        "ndocs": len(texts_clean)
    }


# In[92]:


# Doplnenie tematických štatistík a zarovnanie tém medzi oknami pomocou podobnosti topic-word distribúcií.
def enrich_topic_stats(result):
    stats = result["topic_stats"].copy()

    if len(stats) == 0:
        return stats

    total_docs_window = stats["total_docs"].sum()

    stats["topic_share"] = stats["total_docs"] / total_docs_window
    stats["class_1_ratio"] = np.where(
        stats["total_docs"] > 0,
        stats["class_1_docs"] / stats["total_docs"],
        0.0
    )
    stats["class_0_ratio"] = np.where(
        stats["total_docs"] > 0,
        stats["class_0_docs"] / stats["total_docs"],
        0.0
    )

    return stats


def get_topic_word_distributions(result):
    lda = result["lda"]
    vectorizer = result["vectorizer"]

    if lda is None or vectorizer is None:
        return None, None

    topic_word = lda.components_ / lda.components_.sum(axis=1, keepdims=True)
    vocab = vectorizer.get_feature_names_out()

    return topic_word, vocab


def align_topic_matrices(pre_result, post_result):
    pre_tw, pre_vocab = get_topic_word_distributions(pre_result)
    post_tw, post_vocab = get_topic_word_distributions(post_result)

    if pre_tw is None or post_tw is None:
        return None, None, None

    union_vocab = sorted(set(pre_vocab).union(set(post_vocab)))
    vocab_to_idx = {w: i for i, w in enumerate(union_vocab)}

    def remap(topic_word, vocab):
        M = np.zeros((topic_word.shape[0], len(union_vocab)), dtype=float)
        for j, word in enumerate(vocab):
            M[:, vocab_to_idx[word]] = topic_word[:, j]
        return M

    pre_aligned = remap(pre_tw, pre_vocab)
    post_aligned = remap(post_tw, post_vocab)

    sim = cosine_similarity(pre_aligned, post_aligned)

    return sim, pre_aligned, post_aligned


def match_topics_by_similarity(pre_result, post_result):
    sim, _, _ = align_topic_matrices(pre_result, post_result)

    if sim is None:
        return pd.DataFrame()

    used_post = set()
    matches = []

    for pre_topic in range(sim.shape[0]):
        candidates = [
            (post_topic, sim[pre_topic, post_topic])
            for post_topic in range(sim.shape[1])
            if post_topic not in used_post
        ]

        if not candidates:
            continue

        best_post, best_sim = max(candidates, key=lambda x: x[1])
        used_post.add(best_post)

        matches.append({
            "pre_topic_id": pre_topic,
            "post_topic_id": best_post,
            "topic_similarity": best_sim
        })

    return pd.DataFrame(matches).sort_values("pre_topic_id")


def merge_topic_stats_with_match(pre_result, post_result):
    pre_stats = enrich_topic_stats(pre_result)
    post_stats = enrich_topic_stats(post_result)
    matches = match_topics_by_similarity(pre_result, post_result)

    if len(matches) == 0:
        return pd.DataFrame()

    pre_stats = pre_stats.rename(columns={
        "topic_id": "pre_topic_id",
        "class_0_docs": "pre_class_0_docs",
        "class_1_docs": "pre_class_1_docs",
        "total_docs": "pre_total_docs",
        "topic_share": "pre_topic_share",
        "class_0_ratio": "pre_class_0_ratio",
        "class_1_ratio": "pre_class_1_ratio",
    })

    post_stats = post_stats.rename(columns={
        "topic_id": "post_topic_id",
        "class_0_docs": "post_class_0_docs",
        "class_1_docs": "post_class_1_docs",
        "total_docs": "post_total_docs",
        "topic_share": "post_topic_share",
        "class_0_ratio": "post_class_0_ratio",
        "class_1_ratio": "post_class_1_ratio",
    })

    merged = (
        matches
        .merge(pre_stats, on="pre_topic_id", how="left")
        .merge(post_stats, on="post_topic_id", how="left")
    )

    merged["delta_topic_share"] = merged["post_topic_share"] - merged["pre_topic_share"]
    merged["delta_class_1_ratio"] = merged["post_class_1_ratio"] - merged["pre_class_1_ratio"]

    return merged.sort_values("pre_topic_id")


# In[93]:


# Výpis top slov pre každú tému a zobrazenie tematických štatistík.
def print_lda_result(result, title):
    print("=" * 80)
    print(title)
    print("=" * 80)

    for t in result["topics"]:
        print(f"Topic {t['topic_id']}: {', '.join(t['top_words'])}")

    display(result["topic_stats"])


# In[94]:


# Príprava textov a labelov pre LDA analýzu pred a po vybranom drifte pre oba modely.
adaptive_pre_texts, adaptive_pre_labels = extract_texts_and_labels(adaptive_between_windows["pre"])
adaptive_post_texts, adaptive_post_labels = extract_texts_and_labels(adaptive_between_windows["post"])

static_pre_texts, static_pre_labels = extract_texts_and_labels(static_between_windows["pre"])
static_post_texts, static_post_labels = extract_texts_and_labels(static_between_windows["post"])


# In[95]:


# Spustenie LDA analýzy pre adaptívny a statický model a následné porovnanie výsledkov pred a po drifte.
adaptive_pre_lda = run_lda_topics(adaptive_pre_texts, adaptive_pre_labels, n_topics=7, n_top_words=15, min_df=15, max_df=0.6)
adaptive_post_lda = run_lda_topics(adaptive_post_texts, adaptive_post_labels, n_topics=7, n_top_words=15, min_df=15, max_df=0.6)

static_pre_lda = run_lda_topics(static_pre_texts, static_pre_labels, n_topics=7, n_top_words=15, min_df=15, max_df=0.6)
static_post_lda = run_lda_topics(static_post_texts, static_post_labels, n_topics=7, n_top_words=15, min_df=15, max_df=0.6)


# In[96]:


print_lda_result(adaptive_pre_lda, "ADAPTIVE MODEL - BEFORE SELECTED DRIFT")
print_lda_result(adaptive_post_lda, "ADAPTIVE MODEL - AFTER SELECTED DRIFT")

print_lda_result(static_pre_lda, "STATIC MODEL - BEFORE SELECTED DRIFT")
print_lda_result(static_post_lda, "STATIC MODEL - AFTER SELECTED DRIFT")


# In[97]:


adaptive_topic_comparison = merge_topic_stats_with_match(adaptive_pre_lda, adaptive_post_lda)
static_topic_comparison = merge_topic_stats_with_match(static_pre_lda, static_post_lda)

display(adaptive_topic_comparison)
display(static_topic_comparison)


# In[98]:


print("SELECTED ADAPTIVE DRIFT")
display(selected_adaptive_drift)

print("SELECTED STATIC DRIFT")
display(selected_static_drift)

print("ADAPTIVE TOPIC COMPARISON")
display(adaptive_topic_comparison)

print("STATIC TOPIC COMPARISON")
display(static_topic_comparison)


# In[ ]:




