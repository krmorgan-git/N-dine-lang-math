# N-dine-lang-math

**Author:** Kelly R. Morgan, Assistant Professor of Information Technology  
**Affiliation:** Navajo Technical University (Crownpoint, NM & Chinle, AZ)  
**Related Spec & Studio:** [`N-dine-lang`](https://github.com/krmorgan-git/N-dine-lang) · `N-dine-lang-studio`  

---

## Overview

**`N-dine-lang-math`** is a standalone, offline Flask demonstration app showing how **`N-dine-lang`** (`T S V O L`) and the **4-Stage Diné Problem-Solving Cycle** work together with a fine-tuned **Gemma 3 1B** model (`gemma3-1b-ndine-Q4_K_M.gguf`) for math, geometry, and quantum computing concepts—with **no SQLite database or CSV files required at runtime**.

---

## Guide for AI Models: How Math Works in `N-dine-lang`

1. **Deterministic Math + Neural Translation:**
   - Python evaluates arithmetic expressions safely and deterministically via AST parsing (so the 1B model never hallucinates calculations).
   - Each reasoning step is expressed in canonical **`N-dine-lang`** (`[T/Q/C/M] S- V- O- L-`) and translated into plain-ASCII **Diné Bizaad** using the fine-tuned `.gguf` model.

2. **The 4-Stage Diné Cycle:**
   - **1. Thinking (*Nitsáhákees*):** Identify the given values (`T-prs S-inputs V-be O-2-and-3.`).
   - **2. Planning (*Nahatʼá*):** Order the operations (`T-fut S-1pl V-add O-2-and-3. -> T-fut S-1pl V-multiply O-5-and-2.`).
   - **3. Implementing (*Iiná*):** Execute one operation per line (`M-now S-1pl V-add O-2-and-3.`).
   - **4. Reflecting (*Siihasin*):** Verify and state the final result (`T-prs S-result V-equal O-10.`).

3. **Diné Numbers & Geometry Included:**
   - **Numbers:** `Názbąs` (`Nazbas`, 0), `Tʼááłáʼí` (`TaalaiLaii`, 1), `naaki` (2), `tááʼ` (`taa`, 3), `dį́į́ʼ` (`dii`, 4), `ashdlaʼ` (`ashdla`, 5), `hastą́ą́` (`hastaa`, 6), `tsostsʼid` (`tsostsid`, 7), `tseebíí` (`tseebii`, 8), `náhástʼéí` (`nahastei`, 9), `neeznáá` (`neeznaa`, 10), `naadiin` (20), `neeznádiin` (100), `mííl` (`miil`, 1000).
   - **Geometry:** Circle (`názbąs`), Triangle (`táaʼgo deezʼá`), Square (`dikʼą́`).
   - **Quantum Math Preview:** Maps qubit states (`|0⟩` *Názbąs*, `|1⟩` *Tʼááłáʼí*), Hadamard gates, and measurement onto the same 4-stage `N-dine-lang` cycle.

---

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5055` in your browser. *(Place `gemma3-1b-ndine-Q4_K_M.gguf` in `models/`.)*
