"""N-dine-lang-math: 4-Stage Diné Math & Quantum Demo App."""

import ast
import operator
from pathlib import Path
from flask import Flask, jsonify, render_template, request
from llama_cpp import Llama

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILENAME = "gemma3-1b-ndine-Q4_K_M.gguf"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / MODEL_FILENAME
MODEL_RELEASES_PAGE = "https://github.com/krmorgan-git/N-dine-lang/releases"
MODEL_RELEASE_URL = f"{MODEL_RELEASES_PAGE}/download/v1.0/{MODEL_FILENAME}"


def ensure_model() -> Path:
    """Ensure the local GGUF model exists, downloading from Releases if missing."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        return MODEL_PATH
    import urllib.request

    print(f"Downloading {MODEL_FILENAME} from {MODEL_RELEASE_URL}...")
    try:
        urllib.request.urlretrieve(MODEL_RELEASE_URL, str(MODEL_PATH))
    except Exception as exc:
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
        raise FileNotFoundError(
            f"Could not download {MODEL_FILENAME} from {MODEL_RELEASE_URL} ({exc}).\n"
            f"Download it manually from {MODEL_RELEASES_PAGE} into {MODEL_DIR}/ and restart app.py."
        ) from exc
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
        raise FileNotFoundError(f"Downloaded model is missing or empty: {MODEL_PATH}")
    return MODEL_PATH


ensure_model()

DINE_NUMBERS = {
    0: ("Názbąs", "Nazbas"),
    1: ("Tʼááłáʼí", "TaalaiLaii"),
    2: ("naaki", "naaki"),
    3: ("tááʼ", "taa"),
    4: ("dį́į́ʼ", "dii"),
    5: ("ashdlaʼ", "ashdla"),
    6: ("hastą́ą́", "hastaa"),
    7: ("tsostsʼid", "tsostsid"),
    8: ("tseebíí", "tseebii"),
    9: ("náhástʼéí", "nahastei"),
    10: ("neeznáá", "neeznaa"),
    20: ("naadiin", "naadiin"),
    100: ("neeznádiin", "neeznadiin"),
    1000: ("mííl", "miil"),
}

SHAPES = [
    {"name": "Circle", "dine_raw": "názbąs", "dine_ascii": "nazbas", "ndine": "T-prs S-this V-be O-circle.", "sentence": "Dii nazbas ate.", "svg": "circle"},
    {"name": "Triangle", "dine_raw": "táaʼgo deezʼá", "dine_ascii": "taago deeza", "ndine": "T-prs S-this V-be O-triangle.", "sentence": "Dii taago deeza ate.", "svg": "triangle"},
    {"name": "Square", "dine_raw": "dikʼą́", "dine_ascii": "dika", "ndine": "T-prs S-this V-be O-square.", "sentence": "Dii dika ate.", "svg": "square"},
]

OPS = {
    ast.Add: ("add", "+", operator.add),
    ast.Sub: ("subtract", "-", operator.sub),
    ast.Mult: ("multiply", "*", operator.mul),
    ast.Div: ("divide", "/", operator.truediv),
}

_llm = None


def get_llm():
    global _llm
    if _llm is None and MODEL_PATH.exists():
        _llm = Llama(model_path=str(MODEL_PATH), n_ctx=2048, n_threads=4, verbose=False)
    return _llm


def translate_ndine_with_gemma(ndine_text: str) -> str:
    llm = get_llm()
    if llm is None:
        return "(GGUF model not found in models/)"
    prompt = (
        "Translate the N-dine-lang T S V O L sequence into Diné (ASCII).\n"
        "Output ONLY the Diné (ASCII) words with no English definitions, no apostrophes, and no diacritics.\n\n"
        f"N-dine-lang: {ndine_text}\nDine ASCII:"
    )
    resp = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=96,
        temperature=0.0,
    )
    return resp["choices"][0]["message"]["content"].strip()


def num_to_dine(val: float) -> str:
    if isinstance(val, float) and val.is_integer():
        val = int(val)
    if val in DINE_NUMBERS:
        return f"{val} ({DINE_NUMBERS[val][1]})"
    return str(val)


def eval_ast_steps(node, steps: list[dict]) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -eval_ast_steps(node.operand, steps)
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        left = eval_ast_steps(node.left, steps)
        right = eval_ast_steps(node.right, steps)
        op_verb, op_sym, op_fn = OPS[type(node.op)]
        if op_verb == "divide" and right == 0:
            raise ValueError("Cannot divide by zero.")
        res = op_fn(left, right)
        l_disp = int(left) if left.is_integer() else round(left, 4)
        r_disp = int(right) if right.is_integer() else round(right, 4)
        res_disp = int(res) if float(res).is_integer() else round(res, 4)
        steps.append({
            "english": f"{op_verb.capitalize()} {l_disp} and {r_disp} to get {res_disp}.",
            "ndine": f"M-now S-1pl V-{op_verb} O-{l_disp}-and-{r_disp}.",
            "expr": f"{l_disp} {op_sym} {r_disp} = {res_disp}",
        })
        return res
    raise ValueError("Unsupported expression. Use numbers and +, -, *, /.")


def solve_4_stage(expr_text: str) -> dict:
    tree = ast.parse(expr_text.strip(), mode="eval")
    raw_nums = sorted({
        int(n.value) if isinstance(n.value, float) and n.value.is_integer() else n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
    })

    impl_steps: list[dict] = []
    final_val = eval_ast_steps(tree.body, impl_steps)
    final_disp = int(final_val) if float(final_val).is_integer() else round(final_val, 4)

    # Translate key N-dine-lang lines with Gemma 3 1B
    think_ndine = f"T-prs S-inputs V-be O-{'-and-'.join(str(x) for x in raw_nums)}."
    plan_ndine = " -> ".join(s["ndine"].replace("M-now", "T-fut") for s in impl_steps) or "T-fut S-1pl V-evaluate O-expression."
    reflect_ndine = f"T-prs S-result V-equal O-{final_disp}."

    for s in impl_steps:
        s["dine"] = f"{s['expr']} (Diné: {num_to_dine(float(s['expr'].split('=')[-1].strip()))})"

    return {
        "expression": expr_text,
        "answer": final_disp,
        "answer_dine": num_to_dine(final_val),
        "stages": [
            {
                "name": "1. Thinking (Nitsáhákees)",
                "english": f"Identified given values: {', '.join(num_to_dine(n) for n in raw_nums)}.",
                "ndine": think_ndine,
                "dine": ", ".join(num_to_dine(n) for n in raw_nums),
            },
            {
                "name": "2. Planning (Nahatʼá)",
                "english": f"Break into {len(impl_steps)} ordered step(s).",
                "ndine": plan_ndine,
                "dine": " · ".join(s["expr"] for s in impl_steps),
            },
            {
                "name": "3. Implementing (Iiná)",
                "steps": impl_steps,
            },
            {
                "name": "4. Reflecting (Siihasin)",
                "english": f"Verified result: {expr_text} = {num_to_dine(final_val)}.",
                "ndine": reflect_ndine,
                "dine": translate_ndine_with_gemma(reflect_ndine),
            },
        ],
    }


@app.route("/")
def index():
    return render_template("index.html", numbers=DINE_NUMBERS, shapes=SHAPES)


@app.route("/api/solve", methods=["POST"])
def api_solve():
    data = request.get_json(silent=True) or {}
    expr = (data.get("expression") or "").strip()
    if not expr:
        return jsonify({"error": "Please enter a math expression like (2 + 3) * 2."}), 400
    try:
        return jsonify(solve_4_stage(expr))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/translate_shape", methods=["POST"])
def api_translate_shape():
    data = request.get_json(silent=True) or {}
    ndine = (data.get("ndine") or "").strip()
    if not ndine:
        return jsonify({"error": "Missing N-dine-lang input."}), 400
    return jsonify({"ndine": ndine, "dine": translate_ndine_with_gemma(ndine)})


if __name__ == "__main__":
    get_llm()
    app.run(debug=True, use_reloader=False, host="127.0.0.1", port=5055)
