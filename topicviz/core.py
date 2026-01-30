from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

try:
    import umap  # type: ignore
except Exception:  # pragma: no cover
    umap = None


def _as_list_of_str(docs: Union[Sequence[str], np.ndarray, Iterable[str]]) -> List[str]:
    if isinstance(docs, np.ndarray):
        docs = docs.tolist()
    return [str(x) for x in docs]


def _top_words_per_topic(
    lda: LatentDirichletAllocation,
    feature_names: List[str],
    n_top_words: int = 10,
) -> List[List[str]]:
    topics: List[List[str]] = []
    for topic in lda.components_:
        top_idx = np.argsort(topic)[::-1][:n_top_words]
        topics.append([feature_names[i] for i in top_idx])
    return topics


def _format_label(words: List[str], max_words: int = 6, per_line: int = 3) -> str:
    w = words[:max_words]
    chunks = [w[i:i + per_line] for i in range(0, len(w), per_line)]
    return "<br>".join([" | ".join(c) for c in chunks])


def _topic_sentence(words: List[str], max_words: int = 10) -> str:
    return " ".join(words[:max_words])


def _repel_labels(
    anchors_xy: np.ndarray,
    labels: List[str],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    font_px: int,
    iters: int = 250,
    pad_px: int = 6,
    spring: float = 0.020,
    repel: float = 0.085,
    jitter: float = 0.01,
) -> np.ndarray:
    """Simple force-based label repulsion around anchor points."""
    x0, x1 = xlim
    y0, y1 = ylim
    spanx = (x1 - x0) if (x1 - x0) != 0 else 1.0
    spany = (y1 - y0) if (y1 - y0) != 0 else 1.0

    p = (anchors_xy - np.array([x0, y0])) / np.array([spanx, spany])
    a = p.copy()
    rng = np.random.default_rng(42)
    p = p + rng.normal(0, jitter, size=p.shape)

    ref_px = 800.0
    char_w = 0.62 * font_px / ref_px
    line_h = 1.35 * font_px / ref_px
    pad = pad_px / ref_px

    line_counts = []
    max_line_lens = []
    for t in labels:
        lines = t.replace("<b>", "").replace("</b>", "").split("<br>")
        line_counts.append(len(lines))
        max_line_lens.append(max(len(s) for s in lines) if lines else 1)

    w = np.array([ml * char_w + 2 * pad for ml in max_line_lens])
    h = np.array([lc * line_h + 2 * pad for lc in line_counts])

    def boxes(pos):
        xmin = pos[:, 0] - w / 2
        xmax = pos[:, 0] + w / 2
        ymin = pos[:, 1] - h / 2
        ymax = pos[:, 1] + h / 2
        return xmin, xmax, ymin, ymax

    for _ in range(iters):
        xmin, xmax, ymin, ymax = boxes(p)
        disp = np.zeros_like(p)

        k = p.shape[0]
        for i in range(k):
            for j in range(i + 1, k):
                ox = min(xmax[i], xmax[j]) - max(xmin[i], xmin[j])
                oy = min(ymax[i], ymax[j]) - max(ymin[i], ymin[j])
                if ox > 0 and oy > 0:
                    dx = p[i, 0] - p[j, 0]
                    dy = p[i, 1] - p[j, 1]
                    if abs(dx) + abs(dy) < 1e-9:
                        dx, dy = rng.normal(), rng.normal()
                    if ox > oy:
                        push = np.sign(dx) * repel * ox
                        disp[i, 0] += push
                        disp[j, 0] -= push
                    else:
                        push = np.sign(dy) * repel * oy
                        disp[i, 1] += push
                        disp[j, 1] -= push

        disp += spring * (a - p)
        p += disp

        margin = 0.02
        p[:, 0] = np.clip(p[:, 0], margin + w / 2, 1 - margin - w / 2)
        p[:, 1] = np.clip(p[:, 1], margin + h / 2, 1 - margin - h / 2)

    return p * np.array([spanx, spany]) + np.array([x0, y0])


@dataclass
class TopicViz:
    """Topic modeling (LDA) + visualization (UMAP) with semantic projection."""

    n_topics: int = 12
    max_features: int = 5000
    stop_words: Optional[str] = "english"
    random_state: int = 42
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1

    vectorizer_: Optional[CountVectorizer] = None
    lda_: Optional[LatentDirichletAllocation] = None
    doc_topic_: Optional[np.ndarray] = None
    coords_: Optional[np.ndarray] = None
    topic_words_: Optional[List[List[str]]] = None
    docs_: Optional[List[str]] = None
    reducer_: Optional[Any] = None
    topic_coords_: Optional[np.ndarray] = None

    def fit(
        self,
        docs: Union[Sequence[str], np.ndarray, Iterable[str]],
        *,
        embedding_model: Optional[Any] = None,
        embedding_batch_size: int = 64,
        use_embeddings_for_projection: bool = True,
        topic_embedding_words: int = 10,
    ) -> "TopicViz":
        docs_list = _as_list_of_str(docs)

        self.vectorizer_ = CountVectorizer(
            max_features=self.max_features,
            stop_words=self.stop_words,
            ngram_range=(1, 2),
            min_df=2,
        )
        X = self.vectorizer_.fit_transform(docs_list)

        self.lda_ = LatentDirichletAllocation(
            n_components=self.n_topics,
            learning_method="batch",
            random_state=self.random_state,
            max_iter=25,
            evaluate_every=-1,
        )
        self.doc_topic_ = self.lda_.fit_transform(X)

        feature_names = self.vectorizer_.get_feature_names_out().tolist()
        self.topic_words_ = _top_words_per_topic(self.lda_, feature_names, n_top_words=max(10, topic_embedding_words))

        if umap is None:
            raise ImportError("umap-learn is required. Install with: pip install umap-learn")

        if embedding_model is not None and use_embeddings_for_projection:
            emb = embedding_model.encode(
                docs_list,
                batch_size=embedding_batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
            )
            emb = np.asarray(emb, dtype=float)

            self.reducer_ = umap.UMAP(
                n_components=2,
                n_neighbors=self.umap_n_neighbors,
                min_dist=self.umap_min_dist,
                metric="cosine",
                random_state=self.random_state,
            ).fit(emb)

            self.coords_ = self.reducer_.embedding_

            topic_sentences = [_topic_sentence(w, max_words=topic_embedding_words) for w in self.topic_words_]
            topic_emb = embedding_model.encode(
                topic_sentences,
                batch_size=min(32, len(topic_sentences)),
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            topic_emb = np.asarray(topic_emb, dtype=float)

            try:
                self.topic_coords_ = self.reducer_.transform(topic_emb)
            except Exception:
                self.topic_coords_ = None
        else:
            self.reducer_ = umap.UMAP(
                n_components=2,
                n_neighbors=self.umap_n_neighbors,
                min_dist=self.umap_min_dist,
                metric="hellinger",
                random_state=self.random_state,
            ).fit(self.doc_topic_)
            self.coords_ = self.reducer_.embedding_
            self.topic_coords_ = None

        self.docs_ = docs_list
        return self

    def visualize_topics(
        self,
        *,
        width: int = 800,
        height: int = 800,
        colorscale: str = "Blues",
        density: bool = True,
        show_text: bool = True,
        repel_labels: bool = True,
        topic_label_mode: str = "embedding",
        sample: Optional[int] = None,
        title: str = "Topic map (LDA + UMAP)",
        point_size: int = 6,
        point_color: str = "rgba(255, 70, 70, 0.55)",
        point_opacity: float = 0.55,
        label_size_ratio: float = 60,
        label_words: int = 6,
        label_words_per_line: int = 3,
        hover_show_topic: bool = False,
        hover_max_chars: int = 140,
    ) -> "go.Figure":
        if self.coords_ is None or self.doc_topic_ is None or self.topic_words_ is None or self.docs_ is None:
            raise ValueError("Model is not fitted. Call .fit(...) first.")

        coords = self.coords_
        doc_topic = self.doc_topic_
        docs = self.docs_
        topic_words = self.topic_words_

        n = coords.shape[0]
        if sample is not None and sample < n:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(np.arange(n), size=sample, replace=False)
            coords = coords[idx]
            doc_topic = doc_topic[idx]
            docs = [docs[i] for i in idx.tolist()]

        dominant = doc_topic.argmax(axis=1)
        strength = doc_topic.max(axis=1)

        x = coords[:, 0]
        y = coords[:, 1]

        xlim = (float(np.min(x)), float(np.max(x)))
        ylim = (float(np.min(y)), float(np.max(y)))
        padx = (xlim[1] - xlim[0]) * 0.05 if xlim[1] > xlim[0] else 1.0
        pady = (ylim[1] - ylim[0]) * 0.05 if ylim[1] > ylim[0] else 1.0
        xlim = (xlim[0] - padx, xlim[1] + padx)
        ylim = (ylim[0] - pady, ylim[1] + pady)

        fig = go.Figure()

        if density:
            fig.add_trace(
                go.Histogram2dContour(
                    x=x,
                    y=y,
                    colorscale=colorscale,
                    showscale=False,
                    ncontours=18,
                    contours=dict(showlines=True),
                    opacity=0.55,
                    hoverinfo="skip",
                )
            )

        def trunc(s: str) -> str:
            s = s.replace("\n", " ").strip()
            return (s[:hover_max_chars] + "…") if len(s) > hover_max_chars else s

        if hover_show_topic:
            hover_text = [
                f"{trunc(d)}<br><span style='color:gray'>topic={int(t)} p={float(p):.3f}</span>"
                for d, t, p in zip(docs, dominant, strength)
            ]
        else:
            hover_text = [trunc(d) for d in docs]

        fig.add_trace(
            go.Scattergl(
                x=x,
                y=y,
                mode="markers",
                marker=dict(size=point_size, color=point_color, opacity=point_opacity),
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
                name="documents",
                showlegend=False,
            )
        )

        if show_text:
            font_size = max(11, int(min(width, height) / max(label_size_ratio, 1)))
            label_html = [_format_label(topic_words[t], label_words, label_words_per_line) for t in range(self.n_topics)]

            if topic_label_mode == "embedding" and self.topic_coords_ is not None and self.topic_coords_.shape[0] == self.n_topics:
                anchors = self.topic_coords_.copy()
            else:
                centroids = []
                for t in range(self.n_topics):
                    mask = dominant == t
                    if mask.sum() == 0:
                        centroids.append((float(np.mean(x)), float(np.mean(y))))
                    else:
                        centroids.append((float(np.mean(x[mask])), float(np.mean(y[mask]))))
                anchors = np.array(centroids, dtype=float)

            placed = _repel_labels(anchors, label_html, xlim=xlim, ylim=ylim, font_px=font_size) if repel_labels else anchors

            for t in range(self.n_topics):
                fig.add_annotation(
                    x=float(placed[t, 0]),
                    y=float(placed[t, 1]),
                    text=label_html[t],
                    showarrow=False,
                    font=dict(size=font_size, color="rgb(40,60,220)"),
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="rgba(120,120,255,0.8)",
                    borderwidth=1,
                    borderpad=4,
                )

        fig.update_layout(
            title=title,
            width=width,
            height=height,
            template="plotly_white",
            xaxis=dict(visible=False, range=[xlim[0], xlim[1]]),
            yaxis=dict(visible=False, range=[ylim[0], ylim[1]]),
            margin=dict(l=10, r=10, t=50, b=10),
            showlegend=False,
        )
        return fig
