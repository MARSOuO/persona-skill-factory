from pathlib import Path
try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None

try:
    from tokenizers import Tokenizer as RawTokenizer
except Exception:
    RawTokenizer = None


class TokenCounter:
    def __init__(self, tokenizer_name_or_path: str):
        self.backend = None
        self.tokenizer = None

        p = Path(tokenizer_name_or_path)

        # 1) 直接传 tokenizer.json 文件
        if p.is_file() and p.name == "tokenizer.json":
            if RawTokenizer is None:
                raise RuntimeError("tokenizers not installed. Run: pip install tokenizers")
            self.backend = "raw_tokenizer_json"
            self.tokenizer = RawTokenizer.from_file(str(p))
            return

        # 2) 传的是目录，优先找目录里的 tokenizer.json
        if p.is_dir():
            tokenizer_json = p / "tokenizer.json"
            if tokenizer_json.exists():
                if RawTokenizer is None:
                    raise RuntimeError("tokenizers not installed. Run: pip install tokenizers")
                self.backend = "raw_tokenizer_json"
                self.tokenizer = RawTokenizer.from_file(str(tokenizer_json))
                return

            # 目录里没有 tokenizer.json，再尝试 AutoTokenizer 本地加载
            if AutoTokenizer is None:
                raise RuntimeError("transformers not installed. Run: pip install transformers")
            self.backend = "hf_local_dir"
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(p),
                trust_remote_code=True,
                local_files_only=True,
            )
            return

        # 3) 其他情况，当作 HF repo name
        if AutoTokenizer is None:
            raise RuntimeError("transformers not installed. Run: pip install transformers")
        self.backend = "hf_repo"
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            trust_remote_code=True,
        )

    def count(self, text: str) -> int:
        if not text:
            return 0

        if self.backend == "raw_tokenizer_json":
            return len(self.tokenizer.encode(text).ids)

        return len(self.tokenizer.encode(text, add_special_tokens=False))