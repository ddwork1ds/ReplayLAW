import re
import json
import pickle
import difflib
import traceback
import html as html_esc
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import numpy as np
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# =============================================================================
# 1. CẤU HÌNH MÔ HÌNH & TỰ ĐỘNG DÒ ĐƯỜNG DẪN TRỌNG SỐ
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = "VietAI/vit5-base"

# Tự động nhận diện vị trí adapter_config.json (dù app.py nằm trong hay ngoài vit5_lora_final)
if (BASE_DIR / "adapter_config.json").exists():
    MODEL_OUT = str(BASE_DIR)
elif (BASE_DIR / "vit5_lora_final" / "adapter_config.json").exists():
    MODEL_OUT = str(BASE_DIR / "vit5_lora_final")
elif (BASE_DIR / "processed_data" / "vit5_lora_final" / "adapter_config.json").exists():
    MODEL_OUT = str(BASE_DIR / "processed_data" / "vit5_lora_final")
else:
    MODEL_OUT = str(BASE_DIR)

MAX_INPUT_LEN = 512
MAX_TARGET_LEN = 384
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[INFO] Device: {DEVICE} | Adapter Path: {MODEL_OUT}")

# =============================================================================
# 2. BỘ PHÂN TÍCH CẤU TRÚC CÂY AST & RENDER VĂN BẢN
# =============================================================================
@dataclass
class LegalNode:
    id: str
    level: str
    title: str
    content: str = ""
    children: List["LegalNode"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LegalTree:
    doc_id: str
    preamble: str = ""
    root: LegalNode = field(default_factory=lambda: LegalNode("ROOT", "root", ""))
    epilogue: str = ""

    def clone(self):
        return pickle.loads(pickle.dumps(self))


class LegalStructureParser:
    RE_CHUONG = re.compile(r"^\s*(?:Chương|CHƯƠNG)\s+([IVXLCDM]+|\d+)\s*[.:]?\s*(.*)$")
    RE_MUC = re.compile(r"^\s*(?:Mục|MỤC)\s+(\d+)\s*[.:]?\s*(.*)$")
    RE_DIEU = re.compile(r"^\s*(?:Điều|ĐIỀU|Ðiều|ÐIỀU)\s+(\d+[a-zđ]?)\s*[.:]?\s*(.*)$")
    RE_KHOAN = re.compile(r"^\s*(\d+)\.\s+(.*)$")
    RE_DIEM = re.compile(r"^\s*([a-zđ])\)\s+(.*)$")
    RE_PHULUC = re.compile(r"^\s*(?:PHỤ LỤC|Phụ lục)\s*(?:số\s*)?([IVXLCDM\d]*)\s*[.:]?\s*(.*)$")
    RE_XREF = re.compile(r"(?:tại|theo|quy định tại)\s+Điều\s+(\d+[a-zđ]?)(?:\s+[Kk]hoản\s+(\d+))?", re.I)
    RE_EPILOGUE = re.compile(r"^(?:Nơi\s+nhận:|TM\.|KT\.|CHỦ\s+TỊCH|BỘ\s+TRƯỞNG|THỦ\s+TƯỚNG)", re.I)
    RE_STRUCT_START = re.compile(
        r"^(?:Chương|CHƯƠNG|Mục|MỤC|Điều|ĐIỀU|Ðiều|ÐIỀU|PHỤ LỤC|Phụ lục|\d+\.\s+|[a-zđ]\)\s+|Nơi\s+nhận:|TM\.|KT\.)"
    )

    @staticmethod
    def _is_quoted(line: str) -> bool:
        return line.lstrip().startswith(('"', '“', '‘'))

    def parse(self, doc_id: str, text: str) -> Optional[LegalTree]:
        if not text or len(text) < 30:
            return None
        lines = self._join_lines(text.split("\n"))
        tree = LegalTree(doc_id=doc_id)
        preamble_buf, epilogue_buf = [], []
        in_body = in_epilogue = False
        current_phuluc = current_chuong = current_dieu = current_khoan = current_diem = None
        seen_ids, pl_counter = set(), 0

        def register_id(base_id):
            if base_id not in seen_ids:
                seen_ids.add(base_id)
                return base_id
            idx = 2
            while f"{base_id}_dup{idx}" in seen_ids:
                idx += 1
            uid = f"{base_id}_dup{idx}"
            seen_ids.add(uid)
            return uid

        for line in lines:
            if in_epilogue:
                epilogue_buf.append(line)
                continue
            if in_body and current_phuluc is None and self.RE_EPILOGUE.match(line):
                in_epilogue = True
                epilogue_buf.append(line)
                continue
            if self._is_quoted(line):
                tgt = current_diem or current_khoan or current_dieu or current_chuong or current_phuluc
                if tgt is not None:
                    tgt.content = (tgt.content + "\n" + line).strip()
                else:
                    preamble_buf.append(line)
                continue

            m = self.RE_PHULUC.match(line)
            if m and len(line) < 120:
                in_body = True
                pl_counter += 1
                pl_id = register_id(f"PL{pl_counter}")
                node = LegalNode(id=pl_id, level="phuluc", title=line)
                tree.root.children.append(node)
                current_phuluc = node
                current_chuong = current_dieu = current_khoan = current_diem = None
                continue

            m = self.RE_CHUONG.match(line)
            if m and len(line) < 150:
                in_body = True
                prefix = f"{current_phuluc.id}." if current_phuluc else ""
                node = LegalNode(id=register_id(f"{prefix}C{m.group(1)}"), level="chuong", title=line)
                (current_phuluc or tree.root).children.append(node)
                current_chuong = node
                current_dieu = current_khoan = current_diem = None
                continue

            m = self.RE_MUC.match(line)
            if m:
                prefix = current_chuong.id if current_chuong else "C0"
                node = LegalNode(id=register_id(f"{prefix}.M{m.group(1)}"), level="muc", title=line)
                (current_chuong or current_phuluc or tree.root).children.append(node)
                current_dieu = current_khoan = current_diem = None
                continue

            m = self.RE_DIEU.match(line)
            if m:
                in_body = True
                prefix = f"{current_phuluc.id}." if current_phuluc else ""
                node = LegalNode(id=register_id(f"{prefix}D{m.group(1).lower()}"), level="dieu", title=line)
                (current_chuong or current_phuluc or tree.root).children.append(node)
                current_dieu = node
                current_khoan = current_diem = None
                refs = self.RE_XREF.findall(m.group(2))
                if refs:
                    node.metadata["references"] = refs
                continue

            m = self.RE_KHOAN.match(line)
            if m and current_dieu is not None:
                node = LegalNode(
                    id=register_id(f"{current_dieu.id}.K{m.group(1).lower()}"),
                    level="khoan",
                    title=f"{m.group(1)}.",
                    content=m.group(2),
                )
                current_dieu.children.append(node)
                current_khoan = node
                current_diem = None
                continue

            m = self.RE_DIEM.match(line)
            if m and current_khoan is not None:
                node = LegalNode(
                    id=register_id(f"{current_khoan.id}.P{m.group(1).lower()}"),
                    level="diem",
                    title=f"{m.group(1)})",
                    content=m.group(2),
                )
                current_khoan.children.append(node)
                current_diem = node
                continue

            if not in_body:
                preamble_buf.append(line)
            else:
                tgt = current_diem or current_khoan or current_dieu or current_chuong or current_phuluc
                if tgt is not None:
                    tgt.content = (tgt.content + "\n" + line).strip()
                else:
                    preamble_buf.append(line)

        tree.preamble = "\n".join(preamble_buf).strip()
        tree.epilogue = "\n".join(epilogue_buf).strip()
        if not self._has_dieu(tree.root):
            return None
        return tree

    @classmethod
    def _join_lines(cls, lines):
        result, buffer = [], ""
        for ln in lines:
            ln = ln.strip()
            if not ln:
                if buffer:
                    result.append(buffer)
                    buffer = ""
                continue
            if buffer and cls.RE_STRUCT_START.match(ln):
                result.append(buffer)
                buffer = ""
            buffer = (buffer + " " + ln) if buffer else ln
            if buffer.endswith((".", ":", ";", "!", "?")):
                result.append(buffer)
                buffer = ""
        if buffer:
            result.append(buffer)
        return result

    @staticmethod
    def _has_dieu(node):
        if node.level == "dieu":
            return True
        return any(LegalStructureParser._has_dieu(c) for c in node.children)


class TreeRenderer:
    @staticmethod
    def _render(node: LegalNode, out: List[str]):
        if node.level in ("chuong", "muc", "phuluc", "dieu"):
            if node.title:
                out.append(node.title)
            if node.content:
                out.append(node.content)
        elif node.level in ("khoan", "diem"):
            combined = f"{node.title} {node.content}".strip()
            if combined:
                out.append(combined)
        for c in node.children:
            TreeRenderer._render(c, out)

    @classmethod
    def render(cls, tree: LegalTree) -> str:
        parts = []
        if tree.preamble:
            parts.append(tree.preamble)
        for c in tree.root.children:
            cls._render(c, parts)
        if tree.epilogue:
            parts.append(tree.epilogue)
        return "\n".join(parts).strip()


# =============================================================================
# 3. TÁCH CÂU LỆNH SỬA ĐỔI
# =============================================================================
class InstructionSegmenter:
    AMEND_VERBS = re.compile(r"(?:sửa\s+đổi|bổ\s+sung|bãi\s+bỏ|thay\s+thế|hủy\s+bỏ|căn\s+cứ)", re.I)
    RE_ITEM_START = re.compile(r"^(?:Điều\s+\d+[a-zđ]?[.:]?|\d+[a-zđ]?\.|[a-zđ]\))\s+", re.I)

    @classmethod
    def split_multi_operations(cls, block: str) -> List[str]:
        ops, buf, i, n = [], [], 0, len(block)
        in_sq, curly = False, 0
        while i < n:
            ch = block[i]
            if ch == '"':
                in_sq = not in_sq
                buf.append(ch)
            elif ch == '“':
                curly += 1
                buf.append(ch)
            elif ch == '”':
                curly = max(0, curly - 1)
                buf.append(ch)
            elif ch == ';' and not in_sq and curly == 0:
                la = block[i + 1 : i + 80].strip()
                if cls.AMEND_VERBS.match(la) or cls.RE_ITEM_START.match(la):
                    seg = "".join(buf).strip()
                    if seg:
                        ops.append(seg)
                    buf = []
                else:
                    buf.append(ch)
            else:
                buf.append(ch)
            i += 1
        tail = "".join(buf).strip()
        if tail:
            ops.append(tail)
        return ops

    @classmethod
    def extract_candidate_clauses(cls, raw_text: str) -> List[str]:
        lines = raw_text.split("\n") if "\n" in raw_text else [raw_text]
        lines = [ln.strip() for ln in lines if ln.strip()]
        blocks, cur, in_q = [], [], False
        for ln in lines:
            if not in_q and cls.RE_ITEM_START.match(ln) and cur:
                blocks.append("\n".join(cur))
                cur = [ln]
            else:
                cur.append(ln)
            for ch in ln:
                if ch == '"':
                    in_q = not in_q
                elif ch == '“':
                    in_q = True
                elif ch == '”':
                    in_q = False
        if cur:
            blocks.append("\n".join(cur))
        cands = []
        for b in blocks:
            if len(b) < 10:
                continue
            for op in cls.split_multi_operations(b):
                if cls.AMEND_VERBS.search(op):
                    cands.append(op.strip())
        return cands


# =============================================================================
# 4. BỘ KIỂM TRA 5 LUẬT LOGIC & THỰC THI TRÊN CÂY AST
# =============================================================================
@dataclass
class DSLCommand:
    op: str
    target_id: str
    new_content: str = ""
    old_phrase: str = ""


class DSLParser:
    RE_NOOP = re.compile(r"^NO_OP\(\s*\)$")
    RE_REPEAL = re.compile(r"^REPEAL\(\s*([A-Za-z0-9_.]+)\s*\)$")
    RE_REPLACE = re.compile(r'^REPLACE\(\s*([A-Za-z0-9_.]+)\s*,\s*"((?:[^"\\]|\\.)*)"\s*\)$', re.DOTALL)
    RE_INSERT = re.compile(r'^INSERT_AFTER\(\s*([A-Za-z0-9_.]+)\s*,\s*"((?:[^"\\]|\\.)*)"\s*\)$', re.DOTALL)
    RE_PHRASE = re.compile(
        r'^PHRASE_REPLACE\(\s*([A-Za-z0-9_.]+)\s*,\s*"((?:[^"\\]|\\.)*)"\s*,\s*"((?:[^"\\]|\\.)*)"\s*\)$',
        re.DOTALL,
    )

    @staticmethod
    def _u(s):
        return s.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")

    @classmethod
    def parse(cls, s):
        s = (s or "").strip()
        if cls.RE_NOOP.match(s):
            return DSLCommand("NO_OP", "NONE")
        m = cls.RE_REPEAL.match(s)
        if m:
            return DSLCommand("REPEAL", m.group(1))
        m = cls.RE_REPLACE.match(s)
        if m:
            return DSLCommand("REPLACE", m.group(1), new_content=cls._u(m.group(2)))
        m = cls.RE_INSERT.match(s)
        if m:
            return DSLCommand("INSERT_AFTER", m.group(1), new_content=cls._u(m.group(2)))
        m = cls.RE_PHRASE.match(s)
        if m:
            return DSLCommand(
                "PHRASE_REPLACE", m.group(1), old_phrase=cls._u(m.group(2)), new_content=cls._u(m.group(3))
            )
        return None


class ConsistencyChecker:
    @staticmethod
    def find_node(root, tid):
        if root.id == tid:
            return root
        for c in root.children:
            r = ConsistencyChecker.find_node(c, tid)
            if r:
                return r
        return None

    @staticmethod
    def all_ids(root):
        ids = []
        if root.id != "ROOT":
            ids.append(root.id)
        for c in root.children:
            ids.extend(ConsistencyChecker.all_ids(c))
        return ids

    @staticmethod
    def subtree_text(node):
        p = [node.title, node.content]
        for c in node.children:
            p.append(ConsistencyChecker.subtree_text(c))
        return "\n".join(x for x in p if x)

    @classmethod
    def verify(cls, tree, cmd):
        if cmd.op == "NO_OP":
            return True, "PASS_NO_OP"
        if cmd.op == "PHRASE_REPLACE" and cmd.target_id == "SELF":
            tgt = tree.root
        else:
            tgt = cls.find_node(tree.root, cmd.target_id)
            if tgt is None:
                return False, f"R1_TARGET_NOT_FOUND:{cmd.target_id}"
        ids = cls.all_ids(tree.root)
        if len(ids) != len(set(ids)):
            return False, "R2_DUPLICATE_ID"
        if cmd.op in ("REPLACE", "INSERT_AFTER") and tgt.level not in ("dieu", "khoan", "diem"):
            return False, f"R3_HIERARCHY:{tgt.level}"
        if cmd.op in ("REPLACE", "INSERT_AFTER") and len(cmd.new_content.strip()) < 2:
            return False, "R4_EMPTY_CONTENT"
        if cmd.op == "PHRASE_REPLACE":
            if not cmd.old_phrase:
                return False, "R4_EMPTY_PHRASE"
            if cmd.old_phrase.lower() not in cls.subtree_text(tgt).lower():
                return False, "R4_PHRASE_NOT_IN_SCOPE"
        if cmd.op == "REPEAL":
            dieus = [i for i in ids if re.match(r"^D\d+[a-zđ]?$", i)]
            if len(dieus) <= 1 and cmd.target_id in dieus:
                return False, "R5_LAST_ARTICLE"
        return True, "PASS"


class TreeExecutor:
    @staticmethod
    def _find_parent_idx(parent, tid):
        for i, c in enumerate(parent.children):
            if c.id == tid:
                return parent, i
            p, idx = TreeExecutor._find_parent_idx(c, tid)
            if p is not None:
                return p, idx
        return None, -1

    @staticmethod
    def _replace_phrase(node, old, new):
        cnt = 0
        if old in node.title:
            cnt += node.title.count(old)
            node.title = node.title.replace(old, new)
        if old in node.content:
            cnt += node.content.count(old)
            node.content = node.content.replace(old, new)
        for c in node.children:
            cnt += TreeExecutor._replace_phrase(c, old, new)
        return cnt

    @staticmethod
    def _parse_new_nodes(level, content):
        if level == "dieu":
            t = parser.parse("TMP", content)
            return t.root.children if t else []
        if level == "khoan":
            t = parser.parse("TMP", f"Điều 0. Khung\n{content}")
            if t and t.root.children and t.root.children[0].children:
                return t.root.children[0].children
        if level == "diem":
            t = parser.parse("TMP", f"Điều 0. Khung\n1. Khung\n{content}")
            if t and t.root.children and t.root.children[0].children and t.root.children[0].children[0].children:
                return t.root.children[0].children[0].children
        return []

    @classmethod
    def execute(cls, tree, cmd):
        ok, status = ConsistencyChecker.verify(tree, cmd)
        if not ok:
            return False, status
        if cmd.op == "NO_OP":
            return True, "EXECUTED_NO_OP"
        if cmd.op == "PHRASE_REPLACE":
            scope = tree.root if cmd.target_id == "SELF" else ConsistencyChecker.find_node(tree.root, cmd.target_id)
            n = cls._replace_phrase(scope, cmd.old_phrase, cmd.new_content)
            return True, f"EXECUTED_PHRASE({n})"
        parent, idx = cls._find_parent_idx(tree.root, cmd.target_id)
        if parent is None:
            return False, f"R1_TARGET_NOT_FOUND:{cmd.target_id}"
        if cmd.op == "REPEAL":
            parent.children.pop(idx)
            return True, "EXECUTED_REPEAL"
        target_level = parent.children[idx].level
        new_nodes = cls._parse_new_nodes(target_level, cmd.new_content)
        if cmd.op == "REPLACE":
            if new_nodes:
                new_nodes[0].id = cmd.target_id
                parent.children[idx] = new_nodes[0]
                for off, nn in enumerate(new_nodes[1:], start=1):
                    nn.id = f"{cmd.target_id}_ext{off}"
                    parent.children.insert(idx + off, nn)
            else:
                old_title = parent.children[idx].title
                if target_level == "dieu" and not re.match(r"^(?:Điều|ĐIỀU)\s+\d+", cmd.new_content.strip()):
                    m_head = re.match(r"^((?:Điều|ĐIỀU)\s+\d+[a-zđ]?[.:]?)", old_title)
                    prefix = (m_head.group(1) + " ") if m_head else ""
                    parent.children[idx].title = prefix + cmd.new_content
                else:
                    parent.children[idx].title = cmd.new_content
                parent.children[idx].content = ""
                parent.children[idx].children = []
            return True, "EXECUTED_REPLACE"
        if cmd.op == "INSERT_AFTER":
            if new_nodes:
                for off, nn in enumerate(new_nodes, start=1):
                    nn.id = f"{cmd.target_id}_ins{off}"
                    parent.children.insert(idx + off, nn)
            else:
                parent.children.insert(
                    idx + 1, LegalNode(id=f"{cmd.target_id}_ins1", level=target_level, title=cmd.new_content, content="")
                )
            return True, "EXECUTED_INSERT"
        return False, f"UNSUPPORTED:{cmd.op}"


parser = LegalStructureParser()
seg = InstructionSegmenter()

# =============================================================================
# 5. NẠP MÔ HÌNH ViT5-LoRA & SUY LUẬN
# =============================================================================
_MODEL_CACHE = {}


def _load_model():
    if "m" in _MODEL_CACHE:
        return _MODEL_CACHE["m"], _MODEL_CACHE["t"]

    cfg_path = Path(MODEL_OUT) / "adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy adapter_config.json tại '{MODEL_OUT}'. "
            f"Các file hiện có: {[p.name for p in Path(MODEL_OUT).iterdir()]}"
        )

    import inspect
    from peft import LoraConfig
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw_cfg = json.load(f)
    valid_keys = inspect.signature(LoraConfig.__init__).parameters.keys()
    clean_cfg = {k: v for k, v in raw_cfg.items() if k in valid_keys}
    lora_config = LoraConfig(**clean_cfg)

    print("[INFO] Đang tải mô hình nền VietAI/vit5-base...")
    base = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32
    )

    # Vá tương thích giữa transformers v5 và peft 0.13.2
    if not hasattr(base, "_prepare_encoder_decoder_kwargs_for_generation") and hasattr(
        base, "_prepare_text_encoder_decoder_kwargs_for_generation"
    ):
        base._prepare_encoder_decoder_kwargs_for_generation = base._prepare_text_encoder_decoder_kwargs_for_generation

    print(f"[INFO] Đang gắn trọng số LoRA từ {MODEL_OUT}...")
    m = PeftModel.from_pretrained(base, MODEL_OUT, config=lora_config)
    m.to(DEVICE).eval()

    # Đọc trực tiếp spiece.model qua sentencepiece để tránh bug T5Tokenizer của transformers v5
    sp_path = Path(MODEL_OUT) / "spiece.model"
    if not sp_path.exists():
        from huggingface_hub import hf_hub_download
        sp_path = Path(hf_hub_download(repo_id=MODEL_NAME, filename="spiece.model"))

    try:
        import sentencepiece as spm
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "sentencepiece"])
        import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(sp_path))

    class _SPModel:
        pad_token_id = 0
        eos_token_id = 1
        unk_token_id = 2

    class _SimpleSPtok:
        def __init__(self, processor):
            self.sp = processor
            self.pad_token_id = 0
            self.eos_token_id = 1
            self.unk_token_id = 2

        def __call__(self, texts, max_length=512, truncation=True, padding=True, return_tensors="pt"):
            if isinstance(texts, str):
                texts = [texts]
            seqs = []
            for txt in texts:
                ids = self.sp.encode(txt, out_type=int)
                if truncation and len(ids) > max_length - 1:
                    ids = ids[: max_length - 1]
                ids = ids + [self.eos_token_id]
                seqs.append(ids)

            max_len = max((len(s) for s in seqs), default=1) if padding else max(len(s) for s in seqs)
            input_ids, attention_mask = [], []
            for s in seqs:
                pad_len = max_len - len(s)
                input_ids.append(s + [self.pad_token_id] * pad_len)
                attention_mask.append([1] * len(s) + [0] * pad_len)

            t_ids = torch.tensor(input_ids, dtype=torch.long)
            t_mask = torch.tensor(attention_mask, dtype=torch.long)

            class _Enc(dict):
                def to(self, device):
                    return _Enc({k: v.to(device) if hasattr(v, "to") else v for k, v in self.items()})

            if return_tensors == "pt":
                return _Enc({"input_ids": t_ids, "attention_mask": t_mask})
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        def batch_decode(self, token_ids, skip_special_tokens=True):
            out = []
            if hasattr(token_ids, "tolist"):
                token_ids = token_ids.tolist()
            special_ids = {self.pad_token_id, self.eos_token_id, self.unk_token_id}
            for row in token_ids:
                clean = [i for i in row if not (skip_special_tokens and i in special_ids)]
                out.append(self.sp.decode(clean))
            return out

    if getattr(base.config, "pad_token_id", None) is None:
        base.config.pad_token_id = 0
    if getattr(base.config, "eos_token_id", None) is None:
        base.config.eos_token_id = 1
    if getattr(base.config, "decoder_start_token_id", None) is None:
        base.config.decoder_start_token_id = 0

    t = _SimpleSPtok(sp)

    _MODEL_CACHE["m"], _MODEL_CACHE["t"] = m, t
    print("[INFO] Nạp mô hình thành công!")
    return m, t


def predict_batch(inputs: List[str], batch_size: int = 8) -> List[str]:
    model, tok = _load_model()
    order = np.argsort([-len(x) for x in inputs])
    sorted_inputs = [inputs[idx] for idx in order]
    sorted_preds = [""] * len(inputs)

    for i in range(0, len(sorted_inputs), batch_size):
        batch = sorted_inputs[i : i + batch_size]
        enc = tok(batch, max_length=MAX_INPUT_LEN, truncation=True, padding=True, return_tensors="pt").to(DEVICE)
        with torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=MAX_TARGET_LEN, num_beams=1)
        decoded = [d.strip() for d in tok.batch_decode(gen, skip_special_tokens=True)]
        sorted_preds[i : i + len(decoded)] = decoded

    out = [""] * len(inputs)
    for orig_idx, p in zip(order, sorted_preds):
        out[orig_idx] = p
    return out


# =============================================================================
# 6. GIAO DIỆN WEB GRADIO & HIỂN THỊ DIFF
# =============================================================================
def render_diff_html(old_text: str, new_text: str) -> str:
    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()
    sm = difflib.SequenceMatcher(None, old_lines, new_lines)
    html_parts = [
        '<div style="font-family: Segoe UI, sans-serif; line-height: 1.6; '
        'padding: 14px; border: 1px solid #d0d7de; border-radius: 8px; background: #ffffff; color: #1f2328;">'
    ]
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for ln in new_lines[j1:j2]:
                html_parts.append(f'<div style="margin: 3px 0; color: #1f2328;">{html_esc.escape(ln)}</div>')
        elif tag in ("replace", "delete"):
            for ln in old_lines[i1:i2]:
                html_parts.append(
                    f'<div style="background:#ffebe9; color:#82071e; text-decoration:line-through; '
                    f'padding:4px 8px; border-left:4px solid #cf222e; margin:4px 0; border-radius:4px;">'
                    f"- {html_esc.escape(ln)}</div>"
                )
            if tag == "replace":
                for ln in new_lines[j1:j2]:
                    html_parts.append(
                        f'<div style="background:#dafbe1; color:#116329; font-weight:500; '
                        f'padding:4px 8px; border-left:4px solid #1a7f37; margin:4px 0; border-radius:4px;">'
                        f"+ {html_esc.escape(ln)}</div>"
                    )
        elif tag == "insert":
            for ln in new_lines[j1:j2]:
                html_parts.append(
                    f'<div style="background:#dafbe1; color:#116329; font-weight:500; '
                    f'padding:4px 8px; border-left:4px solid #1a7f37; margin:4px 0; border-radius:4px;">'
                    f"+ {html_esc.escape(ln)}</div>"
                )
    html_parts.append("</div>")
    return "".join(html_parts)


def interactive_consolidate(base_text: str, amend_text: str):
    try:
        tree = parser.parse("DEMO", base_text)
        if tree is None:
            return "❌ Văn bản gốc chưa đúng định dạng Điều/Khoản.", "", ""

        before_rendered = TreeRenderer.render(tree)
        clauses = seg.extract_candidate_clauses(amend_text)
        if not clauses:
            clauses = [amend_text.strip()]

        dsl_preds = predict_batch([f"Chuyển đổi câu lệnh pháp luật sang DSL: {c}" for c in clauses])

        logs = []
        for idx, (cl, dsl) in enumerate(zip(clauses, dsl_preds), 1):
            cmd = DSLParser.parse(dsl)
            if cmd is None:
                logs.append(f"🔴 Lệnh {idx}: Lỗi cú pháp DSL\n   • Input : {cl}\n   • Output: {dsl}")
                continue
            ok, status = TreeExecutor.execute(tree, cmd)
            icon = "🟢" if ok else "🔴"
            logs.append(f"{icon} Lệnh {idx} [{status}]\n   • Câu lệnh : {cl}\n   • Mã DSL   : {dsl}")

        after_rendered = TreeRenderer.render(tree)
        diff_html = render_diff_html(before_rendered, after_rendered)
        ast_nodes = ", ".join(ConsistencyChecker.all_ids(tree.root))
        return "\n\n".join(logs), diff_html, f"Cây AST nhận diện được: [{ast_nodes}]"
    except Exception:
        err_msg = traceback.format_exc()
        print(err_msg)
        return f"❌ Chi tiết lỗi hệ thống:\n{err_msg}", "", ""


SAMPLE_BASE = """Điều 1. Đối tượng nộp thuế trong thương mại điện tử
1. Tổ chức, cá nhân kinh doanh hàng hóa, dịch vụ trên sàn giao dịch thương mại điện tử.
2. Nhà cung cấp nước ngoài không có cơ sở thường trú tại Việt Nam có hoạt động kinh doanh thương mại điện tử.
Điều 2. Trách nhiệm cung cấp thông tin
1. Sàn giao dịch thương mại điện tử có trách nhiệm cung cấp thông tin của thương nhân định kỳ hàng quý.
Điều 3. Hiệu lực thi hành
1. Nghị định này có hiệu lực thi hành kể từ ngày 01 tháng 01 năm 2025."""

SAMPLE_AMEND_OK = """1. Sửa đổi, bổ sung khoản 1 Điều 2 như sau: "1. Sàn giao dịch thương mại điện tử có trách nhiệm khấu trừ, nộp thuế thay và cung cấp thông tin người bán định kỳ hàng tháng."
2. Bãi bỏ khoản 2 Điều 1."""

SAMPLE_AMEND_TRAP = """Bãi bỏ Điều 99."""

# CSS tinh chỉnh bố cục: nút và kết quả chiếm toàn chiều ngang;
# các vùng văn bản dài được cố định chiều cao và cuộn bên trong.
LAYOUT_CSS = """
.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    padding: 24px clamp(14px, 2.5vw, 42px) !important;
    font-family: "Segoe UI", Arial, sans-serif !important;
}
.gradio-container textarea {
    font-size: 16px !important;
    line-height: 1.65 !important;
    overflow-wrap: anywhere !important;
}
.gradio-container label {
    font-size: 16px !important;
    font-weight: 700 !important;
}
#base-document textarea {
    min-height: 175px !important;
}
#amendment-document textarea {
    min-height: 150px !important;
}
/* Mục 3 có chiều cao tối đa; log dài sẽ cuộn thay vì kéo giãn trang. */
#model-output textarea {
    height: 360px !important;
    min-height: 220px !important;
    max-height: 420px !important;
    overflow-y: auto !important;
    resize: vertical !important;
    white-space: pre-wrap !important;
    overflow-wrap: anywhere !important;
}
/* AST dùng toàn bộ chiều rộng cột phải, thẳng hàng với mục 3. */
#ast-structure textarea {
    width: 100% !important;
    min-height: 100px !important;
    max-height: 150px !important;
    overflow-y: auto !important;
    resize: vertical !important;
    white-space: pre-wrap !important;
    overflow-wrap: anywhere !important;
}
/* Nút hợp nhất và phần văn bản sau hợp nhất trải hết khung trình duyệt. */
#run-button,
#run-button button {
    width: 100% !important;
}
#run-button button {
    min-height: 48px !important;
}
#diff-result {
    width: 100% !important;
    min-height: 300px !important;
}
#diff-result > div {
    width: 100% !important;
    box-sizing: border-box !important;
    overflow-wrap: anywhere !important;
}
@media (max-width: 900px) {
    .gradio-container { padding: 14px !important; }
    #base-document textarea { min-height: 150px !important; }
    #model-output textarea {
        height: 300px !important;
        min-height: 200px !important;
        max-height: 360px !important;
    }
}
"""

with gr.Blocks(
    title="REPLAYLAW-VN — LegalTech AI Demo",
    theme=gr.themes.Soft(),
    css=LAYOUT_CSS
) as demo:
    gr.Markdown(
        "# REPLAYLAW-VN: Hệ thống AI Hợp nhất Văn bản Pháp luật Tự động"
    )
    gr.Markdown(
        "Biên dịch câu lệnh sửa đổi tiếng Việt sang **DSL** bằng **ViT5-LoRA** "
        "và thực thi trên cây **AST** có kiểm soát 5 luật logic."
    )

    # Mục 1-2 ở bên trái; AST và mục 3 ở bên phải.
    with gr.Row(equal_height=False):
        with gr.Column(scale=5, min_width=360):
            in_base = gr.Textbox(
                label="1. Văn bản pháp luật gốc (Base Document)",
                value=SAMPLE_BASE,
                lines=6,
                elem_id="base-document"
            )
            in_amend = gr.Textbox(
                label="2. Nội dung văn bản sửa đổi, bổ sung (Amendment)",
                value=SAMPLE_AMEND_OK,
                lines=5,
                elem_id="amendment-document"
            )

        with gr.Column(scale=7, min_width=440):
            # AST được mở rộng toàn chiều rộng cột, không còn ô trống bên cạnh.
            ast_info = gr.Textbox(
                label="Cấu trúc cây AST phân tích được",
                interactive=False,
                lines=3,
                elem_id="ast-structure"
            )

            # Mục 3 giữ chiều cao có giới hạn; nội dung dài có thể cuộn xem.
            out_log = gr.Textbox(
                label="3. Tầng Neural (ViT5 -> DSL) và kiểm tra 5 luật (R1-R5)",
                lines=12,
                interactive=False,
                elem_id="model-output"
            )

    # Nút được đặt dưới cả mục 2 và mục 3, trải hết chiều ngang.
    btn_run = gr.Button(
        "HỢP NHẤT VĂN BẢN TỰ ĐỘNG",
        variant="primary",
        size="lg",
        elem_id="run-button"
    )

    # Kết quả so sánh được đưa ra khỏi hai cột để dùng toàn bộ khung hình.
    out_diff = gr.HTML(
        label="4. Văn bản sau hợp nhất (Visual Diff)",
        elem_id="diff-result"
    )

    btn_run.click(
        interactive_consolidate,
        inputs=[in_base, in_amend],
        outputs=[out_log, out_diff, ast_info]
    )

if __name__ == "__main__":
    _load_model()  # Nạp trước mô hình để kiểm tra ngay khi khởi động
    demo.launch(share=True)