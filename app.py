"""
Health Insurance Analyzer
A Streamlit app that analyses Indian health insurance policy documents
and lets users chat with an AI underwriter about their specific policy.
"""

import base64
import json
from io import BytesIO

import anthropic
import pandas as pd
import pdfplumber
import streamlit as st

# ---------- Configuration ----------

# Pinned snapshot for reproducibility. Change to "claude-sonnet-4-6" or another
# model string if you want to upgrade.
MODEL = "claude-sonnet-4-6"

MAX_TOKENS_ANALYSIS = 4000
MAX_TOKENS_CHAT = 600  # 150 words ≈ 200 tokens, buffer for formatting

PARAMETERS = [
    "Sum Insured",
    "Room rent",
    "ICU",
    "Pre hospitalisation expenses",
    "Post hospitalisation expenses",
    "Co-pay",
    "Day care treatments",
    "OPD coverage",
    "PED coverage",
    "Maternity coverage",
    "Permanent exclusions",
    "Specific illness waiting period",
]

ANALYSIS_SYSTEM_PROMPT = """You are an experienced health insurance underwriter with more than 15 years of professional experience specialising in Indian health insurance policies.

You will be given a document uploaded by a user. Your task is to:

STEP 1 — VALIDATE the document on two checks:

a) is_indian_health_policy — Is this a health insurance policy document issued in India?
   Positive signals: IRDAI / IRDA references; Indian insurer names (HDFC Ergo, Star Health, Niva Bupa, ICICI Lombard, Bajaj Allianz, Care Health, Tata AIG, Aditya Birla, Manipal Cigna, Reliance General, New India Assurance, National Insurance, Oriental Insurance, United India, etc.); Indian currency (₹ / INR / Rupees); Indian addresses (PIN codes, Indian states); Indian regulatory references; UIN (Unique Identification Number).
   Negative signals: USD / $ / £ / € as primary currency; non-Indian insurers (Aetna, Cigna US, BUPA UK, AXA France, etc.); non-Indian regulators (NAIC, PRA, etc.); non-health document types (life, motor, travel, property) — return false for non-health even if Indian.

b) has_terms_and_conditions — Does the document contain the terms & conditions of the insurance contract?
   Positive signals: sections titled Definitions, Coverage / Benefits, Exclusions, Waiting Periods, Claim Procedure, General Conditions, Cancellation, Renewal.
   Negative signals: the document is only a Policy Schedule / Certificate of Insurance / Premium Receipt / KYC form with no actual clauses — return false.

STEP 2 — If BOTH checks pass, extract these 12 parameters from the document:

- Sum Insured — the total coverage amount; include any restoration / bonus if mentioned.
- Room rent — capping if any (e.g., "1% of SI per day", "Single Private AC Room", "No capping").
- ICU — capping if any (e.g., "2% of SI per day", "No capping").
- Pre hospitalisation expenses — number of days covered (e.g., "30 days", "60 days").
- Post hospitalisation expenses — number of days covered (e.g., "60 days", "90 days").
- Co-pay — percentage co-payment if applicable; mention conditions (age, zone, etc.).
- Day care treatments — covered / not covered, number of procedures if listed.
- OPD coverage — outpatient cover details; "Not covered" if absent.
- PED coverage — Pre-Existing Disease waiting period (e.g., "36 months", "48 months").
- Maternity coverage — covered / not covered, limits, waiting period.
- Permanent exclusions — concise list of key permanent exclusions.
- Specific illness waiting period — specific illnesses and their waiting period (e.g., "Cataract, Hernia, Hysterectomy — 24 months").

Rules:
- Use ONLY the uploaded document as the source of truth.
- For each parameter, provide the exact value / condition from the policy. Include sub-limits, capping, or conditions.
- If a parameter is not mentioned in the document, use exactly: "Not specified in the policy".
- Be precise and factual. Do not infer or assume beyond what is written.

Return your response as a VALID JSON object in EXACTLY this format:

{
  "is_indian_health_policy": true,
  "has_terms_and_conditions": true,
  "parameters": {
    "Sum Insured": "...",
    "Room rent": "...",
    "ICU": "...",
    "Pre hospitalisation expenses": "...",
    "Post hospitalisation expenses": "...",
    "Co-pay": "...",
    "Day care treatments": "...",
    "OPD coverage": "...",
    "PED coverage": "...",
    "Maternity coverage": "...",
    "Permanent exclusions": "...",
    "Specific illness waiting period": "..."
  }
}

If is_indian_health_policy is false OR has_terms_and_conditions is false, set "parameters" to null.

Output ONLY the JSON object. No markdown fences, no preamble, no explanation."""

CHAT_SYSTEM_PROMPT = """You are an experienced health insurance underwriter with more than 15 years of professional experience specialising in Indian health insurance policies.

You are helping a user understand their uploaded health insurance policy. The policy document has been provided as context in the conversation.

Strict rules:
- Use ONLY the uploaded policy document as the data source. Do NOT use external knowledge or assumptions.
- Answer each question in a MAXIMUM of 150 words.
- If you are not able to answer any question based on the document, respond EXACTLY with this sentence and nothing else: "I am not able to answer or understand that, regret the same"
- Be precise and factual. Reference specific clauses or sections of the policy when helpful.
- Do not speculate, generalise, or infer beyond what the document states."""

# ---------- Helpers ----------

def extract_pdf_text(file_bytes: bytes) -> tuple[str, bool]:
    """Try local text extraction. Returns (text, success). Success=False for scanned PDFs."""
    try:
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            pages = []
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    pages.append(t)
            text = "\n\n".join(pages)
            # Heuristic: a real policy doc will have well over 500 chars of text.
            # Scanned/image PDFs typically return ~0 chars.
            if len(text.strip()) >= 500:
                return text, True
            return text, False
    except Exception:
        return "", False


def build_document_content(uploaded_file) -> list | None:
    """
    Hybrid approach:
      - PDF: try pdfplumber text extraction; fall back to native PDF block
      - Images: always send as base64 image block (Claude handles OCR)
    Returns a list of content blocks for the user message, or None on failure.
    """
    file_bytes = uploaded_file.getvalue()
    mime = uploaded_file.type or ""
    name = (uploaded_file.name or "").lower()

    # PDF path
    if mime == "application/pdf" or name.endswith(".pdf"):
        text, ok = extract_pdf_text(file_bytes)
        if ok:
            return [{
                "type": "text",
                "text": f"=== UPLOADED POLICY DOCUMENT (extracted text) ===\n\n{text}\n\n=== END OF DOCUMENT ===",
            }]
        # Fall back to native PDF for scanned/image-based policies
        b64 = base64.standard_b64encode(file_bytes).decode("utf-8")
        return [{
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": b64},
        }]

    # Image path
    if mime.startswith("image/") or name.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif")):
        media_type = mime if mime.startswith("image/") else None
        if not media_type:
            if name.endswith(".png"):
                media_type = "image/png"
            elif name.endswith((".jpg", ".jpeg")):
                media_type = "image/jpeg"
            elif name.endswith(".webp"):
                media_type = "image/webp"
            elif name.endswith(".gif"):
                media_type = "image/gif"
            else:
                media_type = "image/jpeg"
        b64 = base64.standard_b64encode(file_bytes).decode("utf-8")
        return [{
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": b64},
        }]

    return None


def analyze_policy(client: anthropic.Anthropic, doc_content: list) -> dict:
    """Single call that validates + extracts parameters. Uses JSON prefill for reliability."""
    user_content = list(doc_content) + [{
        "type": "text",
        "text": "Please validate this document and extract the 12 parameters as instructed. Return only the JSON object."
    }]

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS_ANALYSIS,
        system=ANALYSIS_SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": "{"},  # JSON prefill
        ],
    )

    raw = response.content[0].text.strip()
    # Reconstruct: prefill "{" + model continuation
    json_text = "{" + raw
    # Safety: strip any trailing markdown fences if model added them
    if json_text.endswith("```"):
        json_text = json_text.rsplit("```", 1)[0].strip()
    return json.loads(json_text)


def chat_with_policy(
    client: anthropic.Anthropic,
    doc_content: list,
    history: list[dict],
    new_question: str,
) -> str:
    """
    Chat turn. The policy is re-injected in the first user message every call —
    this preserves memory semantics (the model sees full context each turn) while
    keeping history serialisable in session state.
    """
    first_user_content = list(doc_content) + [{
        "type": "text",
        "text": "The above is my health insurance policy document. I will now ask you questions about it.",
    }]

    messages = [
        {"role": "user", "content": first_user_content},
        {"role": "assistant", "content": "Understood. I have reviewed your policy and will answer your questions based only on its contents."},
    ]
    # Replay prior Q&A from this session
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    # Current question
    messages.append({"role": "user", "content": new_question})

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS_CHAT,
        system=CHAT_SYSTEM_PROMPT,
        messages=messages,
    )
    return response.content[0].text.strip()


def reset_session():
    for k in ("analysis", "doc_content", "chat_history", "policy_filename"):
        if k in st.session_state:
            del st.session_state[k]


# ---------- UI ----------

st.set_page_config(
    page_title="Health Insurance Analyzer",
    page_icon="🏥",
    layout="wide",
)

# Initialise session state
st.session_state.setdefault("analysis", None)
st.session_state.setdefault("doc_content", None)
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("policy_filename", None)

# Header
st.title("🏥 Health Insurance Analyzer")
st.caption("Understand your Indian health insurance policy at a glance — powered by Claude")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        help="Your key is kept only for this browser session and is never saved.",
        placeholder="sk-ant-...",
    )
    st.caption("Don't have one? Get it at [console.anthropic.com](https://console.anthropic.com/)")

    st.divider()
    if st.button("🔄 Reset session", use_container_width=True):
        reset_session()
        st.rerun()

    st.divider()
    st.markdown("### About")
    st.markdown(
        "Extracts 12 key parameters from any Indian health insurance policy document "
        "and lets you chat with an AI underwriter about the specifics of your policy."
    )
    st.markdown("**Model:** Claude Sonnet 4.6")
    st.markdown("**Scope:** Indian health insurance policies only")

# --- Step 1: Upload ---
st.subheader("Step 1 · Upload your policy document")
uploaded_file = st.file_uploader(
    "Drop a PDF or an image (PNG / JPG / WEBP) of your health insurance policy",
    type=["pdf", "png", "jpg", "jpeg", "webp"],
    help="Large scanned PDFs may take longer as they are processed with vision.",
)

# --- Step 2: Disclaimer ---
st.subheader("Step 2 · Accept the disclaimer")
disclaimer_accepted = st.checkbox(
    "I agree that the analysis is for informational purpose only and should be cross referenced for any practical purpose.",
    key="disclaimer",
)

# --- Step 3: Analyse button ---
st.subheader("Step 3 · Analyse")
can_analyse = bool(uploaded_file) and disclaimer_accepted and bool(api_key)

if not api_key:
    st.info("👈 Enter your Anthropic API key in the sidebar to begin.")
elif not uploaded_file:
    st.info("Upload a policy document above.")
elif not disclaimer_accepted:
    st.warning("Tick the disclaimer checkbox to proceed.")

analyse_clicked = st.button(
    "🔍 Analyse policy",
    type="primary",
    disabled=not can_analyse,
)

# --- Run analysis ---
if analyse_clicked:
    try:
        client = anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialise Anthropic client: {e}")
        st.stop()

    with st.spinner("Analysing your policy… this typically takes 20–60 seconds."):
        doc_content = build_document_content(uploaded_file)
        if doc_content is None:
            st.error("Unsupported file type. Please upload a PDF or image.")
            st.stop()

        try:
            result = analyze_policy(client, doc_content)
        except anthropic.AuthenticationError:
            st.error("Invalid API key. Please check the key in the sidebar and try again.")
            st.stop()
        except anthropic.BadRequestError as e:
            st.error(f"The document could not be processed: {e}")
            st.stop()
        except json.JSONDecodeError:
            st.error("Could not parse the model's response. Please try again.")
            st.stop()
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.stop()

        # --- Validation gates ---
        if not result.get("is_indian_health_policy"):
            st.error("Incorrect document uploaded, please upload the correct document")
            st.stop()
        if not result.get("has_terms_and_conditions"):
            st.error("Incomplete document uploaded without the terms and conditions, please upload the correct document")
            st.stop()

        params = result.get("parameters") or {}
        # Persist in session
        st.session_state.analysis = params
        st.session_state.doc_content = doc_content
        st.session_state.policy_filename = uploaded_file.name
        st.session_state.chat_history = []

    st.success(f"Analysis complete for **{uploaded_file.name}**")

# --- Display results ---
if st.session_state.analysis:
    st.divider()
    st.subheader("📋 Policy Analysis")
    st.caption(f"Analysed document: *{st.session_state.policy_filename}*")

    rows = []
    for p in PARAMETERS:
        value = st.session_state.analysis.get(p) or "Not specified in the policy"
        rows.append({"Health insurance parameter": p, "Value": value})

    df = pd.DataFrame(rows).set_index("Health insurance parameter")
    st.table(df)

    # --- Chat section ---
    st.divider()
    st.subheader("💬 Chat about your policy")
    st.caption(
        "Ask anything about coverage, exclusions, waiting periods, claim procedure, "
        "or any specific clause in your policy. Answers are capped at 150 words."
    )

    # Replay history
    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.write(turn["content"])

    user_q = st.chat_input("Type your question here…")
    if user_q:
        # Show user message immediately
        with st.chat_message("user"):
            st.write(user_q)

        try:
            client = anthropic.Anthropic(api_key=api_key)
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    answer = chat_with_policy(
                        client,
                        st.session_state.doc_content,
                        st.session_state.chat_history,
                        user_q,
                    )
                st.write(answer)

            st.session_state.chat_history.append({"role": "user", "content": user_q})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
        except anthropic.AuthenticationError:
            st.error("Invalid API key. Please check the key in the sidebar.")
        except Exception as e:
            st.error(f"Error: {e}")

# Footer
st.divider()
st.caption(
    "⚠️ This tool provides an AI-generated analysis for informational purposes only. "
    "Always cross-reference with your policy document and / or a qualified insurance advisor "
    "before making any decisions."
)
