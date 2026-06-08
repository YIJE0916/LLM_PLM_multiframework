import json
import os
import re
import time

import faiss
import numpy as np
import requests
import streamlit as st
import torch
from groq import Groq
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForMaskedLM, AutoTokenizer, logging

logging.set_verbosity_error()
# ------------------------------
#  Page Configuration
# ------------------------------
st.set_page_config(page_title="Kinase Biology AI", page_icon="🧬", layout="wide")
st.title("🧬 Kinase Biology AI Chatbot")
st.markdown(
    "Ask any question about kinase biology, drug resistance, or mutation mechanisms. "
    "The system retrieves literature, scores mutations with ESM-2, and synthesises a scientific answer."
)

# ------------------------------
#  API Keys & Configuration
# ------------------------------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found. Set it in .streamlit/secrets.toml or as an environment variable.")
    st.stop()

LLM_MODEL_versatile = "llama-3.3-70b-versatile"
LLM_MODEL_instant = "llama-3.1-8b-instant"
UNIPROT_API = "https://rest.uniprot.org/uniprotkb/search"
TOP_K = 5

MODEL_NAME = "facebook/esm2_t30_150M_UR50D"
CACHE_DIR = "./cache"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

FAISS_INDEX_PATH = "kinase_faiss.index"
CHUNKS_PATH = "kinase_chunks_indexed.json"


# ------------------------------
#  Cached Resources
# ------------------------------
@st.cache_resource
def load_esm2():
    """Load the ESM-2 tokenizer and masked language model for mutation scoring.

    Returns:
        tuple: (tokenizer, model, aa_to_id) where `tokenizer` is the ESM-2 tokenizer,
            `model` is the loaded AutoModelForMaskedLM on the selected device, and
            `aa_to_id` maps single-letter amino acids to their token IDs.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR).to(
        DEVICE
    )
    model.eval()
    aa_list = list("ACDEFGHIKLMNPQRSTVWY")
    aa_to_id = {}
    for aa in aa_list:
        tokens = tokenizer.tokenize(aa)
        if len(tokens) != 1:
            raise ValueError(f"Expected single token for {aa}, got {tokens}")
        aa_to_id[aa] = tokenizer.convert_tokens_to_ids(tokens[0])
    return tokenizer, model, aa_to_id


@st.cache_resource
def load_rag():
    """Load retrieval assets for the RAG pipeline.

    Returns:
        tuple: (embedder, index, chunks) where `embedder` is the sentence transformer,
            `index` is the Faiss index, and `chunks` is the list of indexed literature chunks.
    """
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(CHUNKS_PATH, "r") as f:
        chunks = json.load(f)
    return embedder, index, chunks


# ------------------------------
#  Pipeline Functions (unchanged)
# ------------------------------


def decompose_query(user_question: str) -> dict:
    """Extract structured kinase query components from a user question.

    Args:
        user_question (str): The user's natural language question.

    Returns:
        dict: A JSON-like mapping with keys `kinase`, `mutation`, and `drug`.
    """
    client = Groq(api_key=GROQ_API_KEY)
    system_prompt = (
        "You are a computational biology expert specializing in kinase biology and cancer mutations. "
        "Given a user question, extract structured information and respond ONLY with a JSON object "
        "with keys: kinase, mutation, drug. "
        "kinase: gene name of the kinase (e.g. EGFR, BRAF, ABL1). "
        "mutation: point mutation in standard notation (e.g. T790M, V600E) or null if not mentioned. "
        "drug: drug name if mentioned (e.g. gefitinib, imatinib) or null if not mentioned."
    )
    response = client.chat.completions.create(
        model=LLM_MODEL_instant,
        max_tokens=512,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question},
        ],
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        st.warning("⚠️ Stage 1 JSON parsing failed - falling back to EGFR")
        return {"kinase": "EGFR", "mutation": None, "drug": None}


def retrieve_chunks(
    embedder: SentenceTransformer,
    index: faiss.Index,
    chunks_list: list,
    question: str,
    kinase: str | None,
) -> list[str]:
    """Retrieve the most relevant literature chunks for a query.

    Args:
        embedder: SentenceTransformer used to encode the query.
        index: Faiss index for similarity search.
        chunks_list (list): The text chunks indexed in Faiss.
        question (str): The user question to retrieve context for.
        kinase (str, optional): Optional kinase filter to prioritize gene-specific chunks.

    Returns:
        list[str]: The top-k retrieved chunk texts.
    """
    q_emb = embedder.encode([question], show_progress_bar=False)
    q_emb = np.array(q_emb, dtype="float32")
    _, indices = index.search(q_emb, TOP_K * 3)
    candidates = [chunks_list[i] for i in indices[0] if i < len(chunks_list)]
    if kinase:
        kinase_upper = kinase.upper()
        matched = [
            c for c in candidates if kinase_upper in c.get("gene_name", "").upper()
        ]
        others = [
            c for c in candidates if kinase_upper not in c.get("gene_name", "").upper()
        ]
        candidates = (matched + others)[:TOP_K]
    else:
        candidates = candidates[:TOP_K]
    return [c["text"] for c in candidates]


def fetch_wt_sequence(gene_name: str) -> tuple[str, str]:
    """Fetch the reviewed UniProt sequence for a human kinase gene.

    Args:
        gene_name (str): The gene name to look up (e.g. EGFR).

    Returns:
        tuple[str, str]: The UniProt accession and amino acid sequence.

    Raises:
        ValueError: If no reviewed UniProt entry is found for the gene.
    """
    params = {
        "query": f"gene_exact:{gene_name} AND organism_id:9606 AND reviewed:true",
        "format": "json",
        "size": 1,
        "fields": "accession,sequence",
    }
    resp = requests.get(UNIPROT_API, params=params, timeout=30)
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if not results:
        raise ValueError(f"No reviewed UniProt entry found for {gene_name}")
    acc = results[0].get("primaryAccession", gene_name)
    seq = results[0].get("sequence", {}).get("value", "")
    return acc, seq


def apply_mutation(seq: str, mutation: str) -> tuple[int, str, str, str]:
    """Apply a point mutation to a wildtype amino acid sequence.

    Args:
        seq (str): The wildtype protein sequence.
        mutation (str): Mutation in standard format, e.g. T790M.

    Returns:
        tuple[int, str, str, str]: The 1-based mutation position, the observed wildtype amino acid,
            the mutant amino acid, and the mutated sequence.

    Raises:
        ValueError: If the mutation string cannot be parsed or position is out of range.
    """
    match = re.match(r"([A-Z])(\d+)([A-Z])", mutation.upper())
    if not match:
        raise ValueError(f"Cannot parse mutation: {mutation}")
    wt_aa, pos_str, mut_aa = match.groups()
    pos = int(pos_str)
    if pos > len(seq):
        raise ValueError(f"Position {pos} out of range (seq length {len(seq)})")
    actual_wt = seq[pos - 1]
    if actual_wt != wt_aa:
        st.info(
            f"⚠️ Position {pos}: expected {wt_aa}, found {actual_wt} – applying anyway."
        )
    mut_seq = seq[: pos - 1] + mut_aa + seq[pos:]
    return pos, actual_wt, mut_aa, mut_seq


def score_mutation_esm(seq_wt: str, mutation: str, tokenizer, model, aa_to_id):
    """Score a mutation using ESM-2 masked language modeling.

    This function masks the mutation position and computes the log-probability delta
    between the mutant and wildtype amino acids.

    Args:
        seq_wt (str): Wildtype protein sequence.
        mutation (str): Mutation string like T790M.
        tokenizer: Tokenizer for the ESM-2 model.
        model: Loaded ESM-2 masked language model.
        aa_to_id (dict): Mapping from amino acid letters to token IDs.

    Returns:
        dict: Mutation scoring information including `mutation`, `position`, and `delta_ll`.
    """

    def parse_mutant(mutant):
        wt = mutant[0]
        pos = int(mutant[1:-1]) - 1
        mt = mutant[-1]
        return wt, pos, mt

    def build_masked_sequence(sequence_wt, pos):
        seq = list(sequence_wt)
        seq[pos] = "<mask>"
        return "".join(seq)

    def score_one_mutant(mutant, sequence_wt):
        wt, pos, mt = parse_mutant(mutant)
        MAX_LEN = 1020
        if len(sequence_wt) > MAX_LEN:
            start = max(0, pos - MAX_LEN // 2)
            end = min(len(sequence_wt), start + MAX_LEN)
            start = max(0, end - MAX_LEN)
            adj_pos = pos - start
            seq_window = sequence_wt[start:end]
        else:
            adj_pos = pos
            seq_window = sequence_wt
        masked = build_masked_sequence(seq_window, adj_pos)
        inputs = tokenizer(
            [masked], return_tensors="pt", padding=True, add_special_tokens=True
        ).to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits
        mask_idx = (
            (inputs["input_ids"][0] == tokenizer.mask_token_id)
            .nonzero(as_tuple=True)[0]
            .item()
        )
        log_probs = torch.log_softmax(logits[0, mask_idx, :], dim=-1)
        delta = log_probs[aa_to_id[mt]] - log_probs[aa_to_id[wt]]
        return delta.item()

    delta_ll = score_one_mutant(mutation, seq_wt)
    result = {
        "accession": None,
        "mutation": mutation,
        "position": parse_mutant(mutation)[1] + 1,
        "delta_ll": delta_ll,
    }
    return result


def cross_modal_verify(
    user_question: str,
    plm_result: dict | None,
    mutation: str,
    drug: str,
    chunks: list[str] | None,
):
    """Generate a final scientific answer integrating question, literature, and PLM evidence.

    Args:
        user_question (str): Original question from the user.
        plm_result (dict): Mutation scoring result from ESM-2.
        mutation (str): Mutation string if available.
        drug (str): Drug name if mentioned.
        chunks (list[str], optional): Retrieved literature chunks for context.

    Returns:
        str: The final answer generated by the LLM.
    """
    client = Groq(api_key=GROQ_API_KEY)
    if plm_result:
        plm_summary = (
            f"  Mutation: {plm_result.get('mutation')} at position {plm_result.get('position')}\n"
            f"  ΔLL = {plm_result.get('delta_ll', 0):.4f} "
            f"({'strongly disruptive' if plm_result.get('delta_ll', 0) < -1 else 'moderately disruptive' if plm_result.get('delta_ll', 0) < -0.5 else 'neutral'})"
        )
    else:
        plm_summary = "  No PLM results available."
    drug_context = f"The drug in question is {drug}." if drug else ""

    literature_context = ""
    if chunks:
        literature_text = "\n\n".join(chunks)
        literature_context = f"""
LITERATURE EVIDENCE (from UniProt kinase RAG index)
{literature_text}
"""

    prompt = f"""You are a computational biology expert specializing in kinase biology.
A user asked: \"{user_question}\"
{drug_context}

{literature_context}

ESM-2 MUTATION EVIDENCE (ΔLL = logp(mutant) - logp(wildtype) at the masked mutation position):
{plm_summary}

Interpretation guide for ΔLL:
- Near 0: position is not evolutionarily conserved — mutation is neutral
- Negative (< -0.5): position is conserved — mutation disrupts a functionally important residue
- Strongly negative (< -1): position is highly conserved — mutation almost certainly affects function

Task:
1. Answer the user's question using the literature evidence.
2. If mutation data is available, interpret the ΔLL score: does it confirm that this position is evolutionarily conserved?
3. Integrate structural, functional, and evolutionary evidence for a comprehensive explanation.
4. Be concise and scientifically rigorous."""

    response = client.chat.completions.create(
        model=LLM_MODEL_versatile,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


# ------------------------------
#  Streamlit UI with Improved Answer Display
# ------------------------------
embedder, faiss_index, chunks_list = load_rag()
tokenizer_esm, model_esm, aa_to_id = load_esm2()

col1, col2 = st.columns([4, 1])
with col1:
    user_question = st.text_area(
        "Enter your question:",
        height=80,
        placeholder="e.g. Why does EGFR T790M cause resistance to gefitinib?",
        key="question_input",
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    submit = st.button("Ask", type="primary", use_container_width=True)

with st.expander("💡 Example questions"):
    st.markdown(
        "- Why does EGFR T790M cause resistance to gefitinib?\n"
        "- How does BRAF V600E activate the MAPK pathway?\n"
        "- Why does BCR-ABL T315I resist imatinib?\n"
        "- What is the role of the DFG motif in kinase activation?\n"
        "- How does ALK rearrangement drive lung cancer?"
    )

if submit and user_question.strip():
    # Layout with answer at top
    answer_placeholder = st.empty()
    # Progress area (will be removed after completion)
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0, text="Starting...")

    overall_start = time.time()

    try:
        # Stage 1
        progress_bar.progress(5, text="Stage 1/4: Extracting kinase & mutation...")
        with st.status("Stage 1: Query Orchestration", expanded=False) as status1:
            query_info = decompose_query(user_question)
            kinase = query_info.get("kinase") or None
            mutation = query_info.get("mutation")
            drug = query_info.get("drug")
            st.write(f"Kinase: {kinase}  |  Mutation: {mutation}  |  Drug: {drug}")
            status1.update(label="Stage 1 complete ✅", state="complete")
        progress_bar.progress(25, text="Stage 1 complete ✅")

        # Stage 2
        progress_bar.progress(30, text="Stage 2/4: Retrieving literature...")
        with st.status("Stage 2: RAG Retrieval", expanded=False) as status2:
            chunks = retrieve_chunks(
                embedder, faiss_index, chunks_list, user_question, kinase=kinase
            )
            st.write(f"Retrieved {len(chunks)} relevant chunks.")
            if chunks:
                st.text_area(
                    "Sample chunk:",
                    chunks[0][:300] + "...",
                    height=100,
                    disabled=True,
                )
            status2.update(label="Stage 2 complete ✅", state="complete")
        progress_bar.progress(50, text="Literature retrieved ✅")

        # Stage 2b
        progress_bar.progress(55, text="Stage 2b/4: Fetching sequence...")
        with st.status("Stage 2b: Sequence Fetch & Mutation", expanded=False) as status2b:
            wt_seq = None
            mut_seq = None
            mut_pos = None
            try:
                if kinase:
                    acc, wt_seq = fetch_wt_sequence(kinase)
                    st.write(f"Sequence: {acc} ({len(wt_seq)} aa)")
                    if mutation:
                        mut_pos, mut_aa_wt, mut_aa_mut, mut_seq = apply_mutation(
                            wt_seq, mutation
                        )
                        st.write(
                            f"Applied {mutation} at position {mut_pos}: {mut_aa_wt} → {mut_aa_mut}"
                        )
                    else:
                        st.write("No mutation specified.")
                else:
                    st.write("No kinase identified - skipping sequence fetch.")
            except Exception as e:
                st.warning(f"⚠️ Sequence fetch failed: {e}")
            status2b.update(label="Stage 2b complete ✅", state="complete")
        progress_bar.progress(60, text="Sequence ready ✅")

        # Stage 3
        progress_bar.progress(65, text="Stage 3/4: Running ESM-2 mutation scoring...")
        with st.status("Stage 3: ESM-2 Mutation Scoring", expanded=False) as status3:
            plm_result = {}
            if mutation and wt_seq:
                try:
                    plm_result = score_mutation_esm(
                        wt_seq, mutation, tokenizer_esm, model_esm, aa_to_id
                    )
                    delta = plm_result["delta_ll"]
                    label = "disruptive ⚠️" if delta < -0.5 else "neutral"
                    st.metric(
                        "ΔLL (logp_mut - logp_wt)",
                        f"{delta:.4f}",
                        delta_color="inverse",
                    )
                    st.write(f"Interpretation: {label}")
                except Exception as e:
                    st.warning(f"⚠️ Mutation scoring failed: {e}")
            else:
                st.write("No mutation or sequence available - skipping scoring.")
            status3.update(label="Stage 3 complete ✅", state="complete")
        progress_bar.progress(80, text="Mutation scoring complete ✅")

        # Stage 4
        progress_bar.progress(85, text="Stage 4/4: Cross-modal verification...")
        with st.status("Stage 4: Cross-modal Verification", expanded=False) as status4:
            try:
                final_answer = cross_modal_verify(
                    user_question, plm_result, mutation, drug, chunks=chunks
                )
            except Exception as e:
                final_answer = f"Error during final verification: {e}"
            status4.update(label="Stage 4 complete ✅", state="complete")
        progress_bar.progress(95, text="Finalising answer...")

        # Complete
        total_time = time.time() - overall_start
        progress_bar.progress(100, text="Analysis complete!")
        time.sleep(0.3)
        progress_container.empty()

        with answer_placeholder.container():
            st.header("📋 Final Answer")
            st.markdown(final_answer, unsafe_allow_html=True)
            st.caption(f"Pipeline completed in {total_time:.1f} seconds")

    except Exception as e:
        progress_container.empty()
        answer_placeholder.error(f"❌ Pipeline failed: {e}")
        st.exception(e)

else:
    st.info(
        "👆 Enter a question and click **Ask** to get an AI-powered kinase biology analysis."
    )
