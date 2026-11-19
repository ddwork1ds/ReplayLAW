import os
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
import streamlit as st
import sentencepiece as spm
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSeq2SeqLM
from peft import PeftModel

st.set_page_config(
    page_title="REPLAYLAW-VN | LegalTech AI",
    page_icon="⚖️",
    layout="wide",
)

# =============================================================================
# 1. CẤU HÌNH MODEL HUGGING FACE
# =============================================================================
BASE_MODEL = "VietAI/vit5-base"
MAX_INPUT_LEN = 512
MAX_TARGET_LEN = 384

# Cấu hình trong Streamlit Cloud > App > Settings > Secrets:
# HF_LORA_REPO = "DatLocPhuHo/ReplayLaw"
# HF_LORA_SUBFOLDER = "vit5_lora_final"
# HF_TOKEN = "hf_..."  # Chỉ cần nếu Model Repository đặt Private

def _read_secret(name: str, default: str = "") -> str:
    """Đọc Streamlit Secrets trước, sau đó fallback sang environment variables."""
    try:
        value = st.secrets.get(name, default)
    except Exception:
        value = default
    return str(value or os.getenv(name, default)).strip()


LORA_REPO = _read_secret("HF_LORA_REPO")
LORA_SUBFOLDER = _read_secret("HF_LORA_SUBFOLDER").strip("/")
HF_TOKEN = _read_secret("HF_TOKEN") or None

# Tương thích cấu hình cũ dạng "namespace/repo/subfolder":
# Hugging Face yêu cầu repo_id chỉ có "namespace/repo";
# phần còn lại phải truyền riêng qua tham số subfolder.
_repo_parts = LORA_REPO.strip("/").split("/") if LORA_REPO else []
if len(_repo_parts) > 2:
    LORA_REPO = "/".join(_repo_parts[:2])
    if not LORA_SUBFOLDER:
        LORA_SUBFOLDER = "/".join(_repo_parts[2:])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {DEVICE} | Base model: {BASE_MODEL} | LoRA repo: {LORA_REPO or '[NOT CONFIGURED]'}")

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
# 5. NẠP ViT5 + LoRA TRỰC TIẾP TỪ HUGGING FACE HUB
# =============================================================================
_MODEL_CACHE = {}


class SimpleSPTokenizer:
    """Tokenizer SentencePiece tương thích với pipeline hiện tại của dự án."""
    pad_token_id = 0
    eos_token_id = 1
    unk_token_id = 2

    def __init__(self, processor):
        self.sp = processor

    def __call__(self, texts, max_length=512, truncation=True, padding=True, return_tensors="pt"):
        if isinstance(texts, str):
            texts = [texts]
        seqs = []
        for txt in texts:
            ids = self.sp.encode(txt, out_type=int)
            if truncation and len(ids) > max_length - 1:
                ids = ids[: max_length - 1]
            seqs.append(ids + [self.eos_token_id])

        max_len = max((len(s) for s in seqs), default=1) if padding else max(len(s) for s in seqs)
        input_ids, attention_mask = [], []
        for seq in seqs:
            pad_len = max_len - len(seq)
            input_ids.append(seq + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(seq) + [0] * pad_len)

        class Enc(dict):
            def to(self, device):
                return Enc({k: v.to(device) if hasattr(v, "to") else v for k, v in self.items()})

        encoded = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
        return Enc(encoded) if return_tensors == "pt" else encoded

    def batch_decode(self, token_ids, skip_special_tokens=True):
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        special_ids = {self.pad_token_id, self.eos_token_id, self.unk_token_id}
        results = []
        for row in token_ids:
            ids = [int(i) for i in row if not (skip_special_tokens and int(i) in special_ids)]
            results.append(self.sp.decode(ids))
        return results


@st.cache_resource(show_spinner="Đang tải VietAI/ViT5 và LoRA Adapter từ Hugging Face...")
def _load_model():
    if not LORA_REPO or LORA_REPO == "USERNAME/REPOSITORY_NAME":
        raise ValueError(
            "Chưa cấu hình HF_LORA_REPO. Mở Streamlit Cloud > App settings > Secrets "
            'và thêm HF_LORA_REPO = "username/model-repository".'
        )

    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    base = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        token=HF_TOKEN,
        low_cpu_mem_usage=True,
    )

    # Adapter được tải trực tiếp từ Model Repository, không cần copy weights vào GitHub.
    model = PeftModel.from_pretrained(
        base,
        LORA_REPO,
        subfolder=LORA_SUBFOLDER or "",
        token=HF_TOKEN,
        is_trainable=False,
    )

    model.to(DEVICE).eval()

    # SentencePiece model lấy từ model nền VietAI/vit5-base.
    sp_path = hf_hub_download(
        repo_id=BASE_MODEL,
        filename="spiece.model",
        token=HF_TOKEN,
    )
    processor = spm.SentencePieceProcessor(model_file=sp_path)
    tokenizer = SimpleSPTokenizer(processor)

    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model.config, "eos_token_id", None) is None:
        model.config.eos_token_id = tokenizer.eos_token_id
    if getattr(model.config, "decoder_start_token_id", None) is None:
        model.config.decoder_start_token_id = tokenizer.pad_token_id

    return tokenizer, model, DEVICE


def predict_batch(inputs: List[str], batch_size: int = 1) -> List[str]:
    tokenizer, model, device = _load_model()
    order = np.argsort([-len(x) for x in inputs])
    sorted_inputs = [inputs[int(idx)] for idx in order]
    sorted_preds = [""] * len(inputs)

    for start in range(0, len(sorted_inputs), batch_size):
        batch = sorted_inputs[start : start + batch_size]
        encoded = tokenizer(
            batch,
            max_length=MAX_INPUT_LEN,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=MAX_TARGET_LEN,
                num_beams=1,
            )
        decoded = [item.strip() for item in tokenizer.batch_decode(generated, skip_special_tokens=True)]
        sorted_preds[start : start + len(decoded)] = decoded

    predictions = [""] * len(inputs)
    for original_idx, prediction in zip(order, sorted_preds):
        predictions[int(original_idx)] = prediction
    return predictions


def render_diff_html(old_text: str, new_text: str) -> str:
    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()
    sm = difflib.SequenceMatcher(None, old_lines, new_lines)
    html_parts = [
        '<div style="font-family: Segoe UI, sans-serif; line-height: 1.7; '
        'padding: 18px; border: 1px solid #294064; border-radius: 14px; '
        'background: rgba(12, 25, 48, 0.94); color: #e8efff;">'
    ]
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for ln in new_lines[j1:j2]:
                html_parts.append(f'<div style="margin: 3px 0; color: #e8efff;">{html_esc.escape(ln)}</div>')
        elif tag in ("replace", "delete"):
            for ln in old_lines[i1:i2]:
                html_parts.append(
                    f'<div style="background:#482333; color:#ffd5df; text-decoration:line-through; '
                    f'padding:4px 8px; border-left:4px solid #fb7185; margin:4px 0; border-radius:4px;">'
                    f"- {html_esc.escape(ln)}</div>"
                )
            if tag == "replace":
                for ln in new_lines[j1:j2]:
                    html_parts.append(
                        f'<div style="background:#123b3a; color:#bbf7e7; font-weight:500; '
                        f'padding:4px 8px; border-left:4px solid #34d399; margin:4px 0; border-radius:4px;">'
                        f"+ {html_esc.escape(ln)}</div>"
                    )
        elif tag == "insert":
            for ln in new_lines[j1:j2]:
                html_parts.append(
                    f'<div style="background:#123b3a; color:#bbf7e7; font-weight:500; '
                    f'padding:4px 8px; border-left:4px solid #34d399; margin:4px 0; border-radius:4px;">'
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

# =============================================================================
# 7. GIAO DIỆN STREAMLIT – PHONG CÁCH LEGALTECH CỦA REPLAYLAW-VN
# =============================================================================
APP_CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --rl-ink: #eaf0ff;
    --rl-muted: #aab9d5;
    --rl-border: rgba(124, 160, 211, 0.24);
}

html, body, [class*="css"] { font-family: 'Be Vietnam Pro', sans-serif; }
.stApp { color: var(--rl-ink); background: transparent; }

[data-testid="stAppViewContainer"] {
    background-color: #071326;
    background-image:
        radial-gradient(ellipse at 12% 3%, rgba(37, 99, 235, 0.25), transparent 34%),
        radial-gradient(ellipse at 88% 14%, rgba(14, 165, 233, 0.16), transparent 30%),
        radial-gradient(ellipse at 70% 85%, rgba(99, 102, 241, 0.13), transparent 38%),
        linear-gradient(135deg, #071326 0%, #0a1931 48%, #0d1b36 100%);
    background-attachment: fixed;
}

[data-testid="stHeader"] { background: rgba(7, 19, 38, 0.72); backdrop-filter: blur(14px); }
[data-testid="stMainBlockContainer"] { max-width: 1500px; padding-top: 2.2rem; padding-bottom: 3rem; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0a1830 0%, #0c1d38 100%); border-right: 1px solid var(--rl-border); }

h1, h2, h3, h4, p, label, [data-testid="stMarkdownContainer"] { color: var(--rl-ink); }
h2, h3 { font-family: 'Space Grotesk', 'Be Vietnam Pro', sans-serif; letter-spacing: -0.02em; }

.hero-shell {
    position: relative; overflow: hidden; padding: 30px 34px 28px; margin: 0 0 26px 0;
    border: 1px solid rgba(125, 170, 230, 0.30); border-radius: 22px;
    background:
        radial-gradient(circle at 86% 18%, rgba(34, 211, 238, 0.16), transparent 25%),
        radial-gradient(circle at 14% 100%, rgba(59, 130, 246, 0.19), transparent 36%),
        linear-gradient(115deg, rgba(15, 35, 66, 0.96), rgba(13, 31, 59, 0.90));
    box-shadow: 0 18px 55px rgba(0, 0, 0, 0.20), inset 0 1px 0 rgba(255,255,255,0.05);
}
.hero-shell:after {
    content: ""; position: absolute; width: 310px; height: 310px; right: -100px; top: -180px;
    border: 1px solid rgba(103, 232, 249, 0.18); border-radius: 50%;
    box-shadow: 0 0 0 28px rgba(103,232,249,0.025), 0 0 0 58px rgba(103,232,249,0.018);
    pointer-events: none;
}
.hero-kicker {
    display: inline-flex; align-items: center; gap: 8px; padding: 7px 12px; margin-bottom: 15px;
    border: 1px solid rgba(103, 232, 249, 0.28); border-radius: 999px;
    background: rgba(8, 27, 51, 0.62); color: #8be9f5;
    font: 700 11px 'Space Grotesk', sans-serif; letter-spacing: 0.16em; text-transform: uppercase;
}
.hero-title {
    margin: 0; color: #f5f8ff;
    font: 800 clamp(28px, 4vw, 44px)/1.15 'Space Grotesk', 'Be Vietnam Pro', sans-serif;
    letter-spacing: -0.045em;
}
.hero-title span {
    display: block; margin-top: 6px;
    background: linear-gradient(90deg, #93c5fd 0%, #67e8f9 52%, #a5b4fc 100%);
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-description { max-width: 920px; margin: 15px 0 0 0; color: #b8c8e2; font-size: 14px; line-height: 1.8; }
.hero-tags { display: flex; flex-wrap: wrap; gap: 9px; margin-top: 19px; }
.hero-tag {
    padding: 7px 11px; border: 1px solid rgba(135, 169, 216, 0.22); border-radius: 8px;
    background: rgba(5, 17, 35, 0.48); color: #ccdcf5; font-size: 11px; font-weight: 600;
}

[data-testid="stTextArea"] textarea {
    color: #edf4ff !important; background: rgba(8, 22, 43, 0.88) !important;
    border: 1px solid rgba(121, 160, 215, 0.34) !important; border-radius: 12px !important;
    line-height: 1.65 !important;
}
[data-testid="stTextArea"] textarea:focus { border-color: #67c9f2 !important; box-shadow: 0 0 0 1px rgba(103, 201, 242, 0.22) !important; }
[data-testid="stTextArea"] label, [data-testid="stWidgetLabel"] p { color: #dce8fb !important; font-weight: 600 !important; }
.stButton > button[kind="primary"] {
    min-height: 52px; border: 1px solid rgba(147, 197, 253, 0.5); border-radius: 12px;
    background: linear-gradient(100deg, #2563eb 0%, #0ea5e9 54%, #6366f1 100%);
    color: white; font-weight: 800; letter-spacing: 0.045em;
    box-shadow: 0 10px 28px rgba(37, 99, 235, 0.25); transition: transform 160ms ease, box-shadow 160ms ease;
}
.stButton > button[kind="primary"]:hover { transform: translateY(-1px); box-shadow: 0 14px 34px rgba(37, 99, 235, 0.34); border-color: #93c5fd; }
[data-testid="stAlert"] { border-radius: 12px; }
[data-testid="stCodeBlock"] { border: 1px solid var(--rl-border); border-radius: 12px; }
hr { border-color: rgba(124, 160, 211, 0.22) !important; }
small, .stCaption, [data-testid="stCaptionContainer"] { color: var(--rl-muted) !important; }

@media (max-width: 760px) {
    [data-testid="stMainBlockContainer"] { padding-left: 1rem; padding-right: 1rem; }
    .hero-shell { padding: 23px 20px; border-radius: 17px; }
}
"""
st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)

st.markdown(
    """
    <section class="hero-shell">
      <div class="hero-kicker">✦ Artificial Intelligence &amp; Law</div>
      <h1 class="hero-title">REPLAYLAW-VN
        <span>CÔNG NGHỆ AI KIẾN TẠO TƯƠNG LAI PHÁP LUẬT</span>
      </h1>
      <p class="hero-description">
        Nền tảng AI hỗ trợ hợp nhất, phân tích và xử lý văn bản pháp luật Việt Nam.
        Kết hợp mô hình ngôn ngữ ViT5-LoRA với bộ máy AST và hệ thống 5 quy tắc logic,
        hướng đến quy trình xử lý minh bạch và nhất quán.
      </p>
      <div class="hero-tags">
        <span class="hero-tag">AI LANGUAGE MODEL · ViT5-LoRA</span>
        <span class="hero-tag">AST RULE ENGINE · R1–R5</span>
        <span class="hero-tag">LEGAL DOCUMENT AI</span>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)
st.caption("REPLAYLAW-VN WORKSPACE · Hệ thống xử lý và hợp nhất văn bản pháp luật")

with st.sidebar:
    st.subheader("Trạng thái hệ thống")
    st.write(f"Thiết bị: **{DEVICE.upper()}**")
    if LORA_REPO:
        st.success(f"LoRA Repository: {LORA_REPO}")
        if LORA_SUBFOLDER:
            st.caption(f"Thư mục adapter: {LORA_SUBFOLDER}")
    else:
        st.warning("Chưa cấu hình HF_LORA_REPO trong Streamlit Secrets.")
    st.caption("Model nền: VietAI/vit5-base")

left_col, right_col = st.columns([5, 7], gap="large")
with left_col:
    st.subheader("1. Văn bản pháp luật gốc (Base Document)")
    base_text = st.text_area("Văn bản gốc", value=SAMPLE_BASE, height=230, label_visibility="collapsed")
    st.subheader("2. Nội dung văn bản sửa đổi, bổ sung (Amendment)")
    amend_text = st.text_area("Nội dung sửa đổi", value=SAMPLE_AMEND_OK, height=190, label_visibility="collapsed")

with right_col:
    st.subheader("Cấu trúc cây AST phân tích được")
    ast_placeholder = st.empty()
    st.subheader("3. Tầng Neural (ViT5 → DSL) và kiểm tra 5 luật (R1–R5)")
    log_placeholder = st.empty()

run_button = st.button("HỢP NHẤT VĂN BẢN TỰ ĐỘNG", type="primary", use_container_width=True)

if run_button:
    if not base_text.strip() or not amend_text.strip():
        st.warning("Vui lòng nhập cả văn bản gốc và nội dung sửa đổi.")
    elif not LORA_REPO:
        st.error(
            "Chưa kết nối Model Repository. Vào Streamlit Cloud → Settings → Secrets và thêm:\n\n"
            'HF_LORA_REPO = "username/model-repository"\n'
            'HF_TOKEN = "hf_..."  # chỉ thêm nếu repository Private'
        )
    else:
        try:
            with st.spinner("Đang tải mô hình và xử lý các câu lệnh sửa đổi..."):
                logs, diff_html, ast_text = interactive_consolidate(base_text, amend_text)
            ast_placeholder.text_area("AST", value=ast_text, height=110, label_visibility="collapsed")
            log_placeholder.text_area("Nhật ký xử lý", value=logs, height=350, label_visibility="collapsed")
            st.subheader("4. Văn bản sau hợp nhất (Visual Diff)")
            st.markdown(diff_html, unsafe_allow_html=True)
        except Exception as exc:
            st.exception(exc)

st.divider()
st.caption("REPLAYLAW-VN | ViT5-LoRA · DSL · AST · R1–R5")
