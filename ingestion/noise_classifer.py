import os
import re
import json
import math
import time
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Callable, Optional, Dict, Union
from torch.cuda import is_available as is_cuda_available

# ---------------------------
# Logging Setup
# ---------------------------
try:
    from utils.logger import get_ingestion_logger
    logger = get_ingestion_logger(Path(__file__).stem, max_files=5)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.warning("Could not import 'utils.logger'. Using default logging configuration.")

# ---------------------------
# Configuration & Constants
# ---------------------------
# HF Cache Setup
HF_CACHE_DIR = os.environ.get('HF_HOME', os.environ.get('TRANSFORMERS_CACHE', None))
if HF_CACHE_DIR:
    os.environ['HF_HOME'] = HF_CACHE_DIR
    os.environ['HF_DATASETS_CACHE'] = HF_CACHE_DIR
    os.environ['HUGGINGFACE_HUB_CACHE'] = HF_CACHE_DIR
    os.environ['TORCH_HOME'] = HF_CACHE_DIR
else:
    logger.warning("No HF_HOME or TRANSFORMERS_CACHE set. Using default cache location.")

# Performance Config
BUCKET_SIZES = [256, 512, 1024, 2048]
BATCH_MAX_PROMPTS = 8
PAD_TOKEN_TEXT = ""

# Regex Patterns
PARA_SPLIT_RE = re.compile(r'\n{2,}')

PROMPT_TEMPLATE = '''You are classifying whether a markdown paragraph belongs to the main academic paper body.

Return ONLY a valid JSON object with this exact schema:
{{
  "decision": "KEEP" or "REMOVE",
  "reason": "brief explanation"
}}

Classification Rules - Be CONSERVATIVE (when in doubt, KEEP):

KEEP (almost everything):
- ALL section headings (# Title, ## Abstract, ### Methods, etc.)
- Research content, discussions, methodology, results
- Data analysis, statistical tables, equations
- Figure captions, figure legends, table captions
- Citations in text (e.g., "Smith et al. (2020)")
- Abstract sections
- Acknowledgments sections
- ANY content that appears to be part of the paper structure

REMOVE (only obvious non-content):
- Individual reference list entries formatted like "[1] Author, A. et al. (2020). Title. Journal."
- Solitary page numbers (just "1" or "Page 23" on their own line)
- Repeated headers/footers that appear on every page (e.g., "Journal of Science | Volume 10")
- Copyright notices (e.g., "© 2020 Publisher")
- Email addresses appearing alone (not in author lists)
- URL footers (e.g., "http://journal.com/issue")

CRITICAL: Titles, headings (# Title), and section headers (## Abstract, ### Methods) should ALWAYS be KEPT.
When uncertain, default to KEEP.

Paragraph to classify:
{CONTENT}

JSON Response:'''

# ---------------------------
# Imports (Conditional)
# ---------------------------
try:
    from vllm import LLM, SamplingParams
except Exception as e:
    logger.error(f"Failed to import vLLM: {e}")
    LLM = None
    SamplingParams = None

try:
    import tiktoken
except Exception:
    tiktoken = None


class ContentClassifier:
    def __init__(self, model_name: str, device: str = "cuda:0", sampling_params=None):
        if LLM is None:
            raise RuntimeError("vLLM library is not installed or failed to load.")
        
        self.sampling_params = sampling_params or SamplingParams(temperature=0, max_tokens=128)
        self.estimate_tokens = self._get_token_counter()
        
        logger.info(f"Loading vLLM model {model_name} on {device}...")
        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
            max_model_len=2048,
            download_dir=HF_CACHE_DIR,
            enforce_eager=False,  # CUDAGraph optimization
            max_num_seqs=16,
            disable_custom_all_reduce=True,
            tensor_parallel_size=1,
            dtype="bfloat16",
            enable_prefix_caching=True,
            disable_sliding_window=True,
        )

    def _get_token_counter(self, encoding_name: str = "cl100k_base") -> Callable[[str], int]:
        """Returns a token counting function, preferring tiktoken."""
        if tiktoken:
            try:
                enc = tiktoken.get_encoding(encoding_name)
                return lambda text: len(enc.encode(text))
            except Exception as e:
                logger.warning(f"tiktoken encoding load failed: {e}. Falling back to heuristic.")
        return lambda text: max(1, len(text) // 4)

    # ---------------------------
    # Text Pre-processing
    # ---------------------------
    @staticmethod
    def _looks_like_table(text: str) -> bool:
        """Detect if text contains markdown table structure."""
        lines = text.splitlines()
        pipe_lines = [l for l in lines if '|' in l]
        
        if len(pipe_lines) < 3:
            return False
        
        # Check for table separator line (e.g., |---|---|)
        has_separator = any(re.match(r'^\s*\|[\s\-:|]+\|\s*$', l) for l in lines)
        
        if has_separator and len(pipe_lines) >= 3:
            return True
            
        # Alternative: multiple lines with same number of pipes
        if len(pipe_lines) >= 3:
            pipe_counts = [l.count('|') for l in pipe_lines]
            if len(set(pipe_counts)) == 1 and pipe_counts[0] >= 2:
                return True
        return False

    def _remove_pagebreaks_preserve_tables(self, text: str) -> str:
        """Conservative pagebreak removal that preserves table structures."""
        lines = text.splitlines()
        out_lines = []
        
        for i, L in enumerate(lines):
            s = L.strip()
            if not s:
                out_lines.append(L)
                continue
            
            # Check context for tables
            prev = lines[i-1] if i-1 >= 0 else ''
            nxt = lines[i+1] if i+1 < len(lines) else ''
            
            if '|' in s or '|' in prev or '|' in nxt:
                out_lines.append(L)
                continue
            
            # Remove pagebreak patterns
            if re.match(r'^[-–—]{3,}$', s):
                continue
            elif re.match(r'^\d+\s*$', s):
                continue
            elif re.match(r'^\s*\{{0,2}\d+\}{0,2}\s*[-–—]{2,}\s*$', L):
                continue
            else:
                out_lines.append(L)
        
        return "\n".join(out_lines)

    def chunk_paragraphs(self, md_text: str) -> List[str]:
        """Splits markdown text into classify-able chunks."""
        md_text = self._remove_pagebreaks_preserve_tables(md_text)
        md_text = md_text.replace('\r\n', '\n').replace('\r', '\n')
        raw_chunks = [p.strip() for p in PARA_SPLIT_RE.split(md_text) if p.strip() != ""]
        
        chunks = []
        for p in raw_chunks:
            # Reattach orphaned equation/figure lines to the previous chunk
            if len(p.splitlines()) == 1 and len(p) < 80 and chunks:
                if re.match(r'^\(?Eq\.?\s*\d+\)?$|^\(?\d+\)?$|^\[?\d+\]?$|^Fig\.?\s*\d+$', p, re.IGNORECASE):
                    chunks[-1] = chunks[-1] + "\n" + p
                    continue
            chunks.append(p)
        return chunks

    # ---------------------------
    # Optimization Helpers
    # ---------------------------
    def _pad_to_bucket(self, text: str, bucket_tokens: int) -> str:
        """Strict padding to EXACTLY hit bucket size for CUDAGraph efficiency."""
        current = self.estimate_tokens(text)
        if current >= bucket_tokens:
            return text
        
        needed_tokens = bucket_tokens - current
        pad_unit = PAD_TOKEN_TEXT + "\n"
        pad_unit_tokens = self.estimate_tokens(pad_unit)
        
        num_pads = max(1, needed_tokens // max(1, pad_unit_tokens))
        pad = "\n\n" + (pad_unit * num_pads)
        result = text + pad
        
        # Micro-adjustment
        final_tokens = self.estimate_tokens(result)
        if abs(final_tokens - bucket_tokens) > 10:
            diff = bucket_tokens - final_tokens
            if diff > 0:
                result += pad_unit * max(0, diff // max(1, pad_unit_tokens))
        
        return result

    def _extract_json_from_text(self, text: str) -> Optional[Dict]:
        """Robust JSON extraction handling markdown blocks and nested structures."""
        # Clean markdown
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        # Regex strategies
        json_patterns = [
            r'\{[^{}]*"decision"[^{}]*"reason"[^{}]*\}', 
            r'\{(?:[^{}]|\{[^{}]*\})*\}'
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    parsed = json.loads(match)
                    if 'decision' in parsed and 'reason' in parsed:
                        return parsed
                except json.JSONDecodeError:
                    continue
        
        # Fallback regex
        decision_match = re.search(r'"decision"\s*:\s*"(\w+)"', text, re.IGNORECASE)
        reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
        
        if decision_match:
            return {
                'decision': decision_match.group(1).upper(),
                'reason': reason_match.group(1) if reason_match else 'extracted via regex'
            }
        return None

    # ---------------------------
    # Core Logic
    # ---------------------------
    def classify_batch(self, paragraphs: List[str]) -> List[Dict]:
        """Classifies a list of paragraphs using batch inference."""
        N = len(paragraphs)
        out = [None] * N
        
        # Group by token buckets
        buckets_map = defaultdict(list)
        for idx, para in enumerate(paragraphs):
            tok = self.estimate_tokens(para)
            # Find smallest bucket that fits
            bucket = next((b for b in BUCKET_SIZES if tok <= b), BUCKET_SIZES[-1])
            buckets_map[bucket].append((idx, tok))
        
        logger.info(f"Classifying {N} paragraphs across {len(buckets_map)} size buckets")
        
        fallback_count = 0
        
        for bucket_tok, idx_tok_list in buckets_map.items():
            groups = [idx for idx, _ in idx_tok_list]
            
            for i0 in range(0, len(groups), BATCH_MAX_PROMPTS):
                group = groups[i0:i0 + BATCH_MAX_PROMPTS]
                prompts = []
                mapping = []
                
                for idx in group:
                    orig = paragraphs[idx]
                    para_tokens = self.estimate_tokens(orig)
                    prompt_overhead = self.estimate_tokens(PROMPT_TEMPLATE) - self.estimate_tokens('{CONTENT}')
                    
                    # Handle over-length paragraphs
                    if para_tokens + prompt_overhead > 2048:
                         # Split logic for very long paragraphs
                        logger.warning(f"Paragraph {idx} too long ({para_tokens} tokens), splitting...")
                        lines = orig.split('\n')
                        if len(lines) < 2:
                            out[idx] = {'idx': idx, 'paragraph': orig, 'flag': 'IMPORTANT', 'reason': 'too long (auto-kept)'}
                            continue
                        
                        mid = len(lines) // 2
                        sub1, sub2 = '\n'.join(lines[:mid]).strip(), '\n'.join(lines[mid:]).strip()
                        
                        for sub in [sub1, sub2]:
                            if not sub: continue
                            sub_prompt = PROMPT_TEMPLATE.replace('{CONTENT}', sub)
                            prompts.append(sub_prompt)
                            mapping.append((idx, sub, True)) # True = is_split
                        continue
                    
                    # Standard case: Pad and Add
                    padded = self._pad_to_bucket(orig, bucket_tok)
                    final_prompt = PROMPT_TEMPLATE.replace('{CONTENT}', padded)
                    
                    # Safety check on final length
                    if self.estimate_tokens(final_prompt) > 2048:
                        final_prompt = PROMPT_TEMPLATE.replace('{CONTENT}', orig) # Try unpadded
                    
                    prompts.append(final_prompt)
                    mapping.append((idx, orig, False))

                if not prompts:
                    continue

                # Inference
                try:
                    outputs = self.llm.generate(prompts=prompts, sampling_params=self.sampling_params)
                    gen_texts = [o.outputs[0].text for o in outputs]
                except Exception as e:
                    logger.error(f"vLLM Batch generation failed: {e}")
                    continue

                # Processing Results
                split_results = defaultdict(list)
                
                for (idx, content, is_split), gen in zip(mapping, gen_texts):
                    parsed = self._extract_json_from_text(gen)
                    
                    decision = 'KEEP' # Default
                    reason = 'fallback'
                    
                    if parsed:
                        decision = parsed.get('decision', '').upper()
                        reason = parsed.get('reason', '')[:300]
                    else:
                        logger.warning(f"JSON parse failed for para {idx}, defaulting to KEEP.")
                        fallback_count += 1
                    
                    if is_split:
                        split_results[idx].append((decision, reason))
                    else:
                        flag = 'OUT' if decision == 'REMOVE' else 'IMPORTANT'
                        out[idx] = {'idx': idx, 'paragraph': content, 'flag': flag, 'reason': reason}

                # Resolve splits
                for idx, decisions in split_results.items():
                    # Conservative: If ANY part is KEEP, keep the whole original
                    if any(d == 'KEEP' for d, _ in decisions):
                        reasons = [r for d, r in decisions if d == 'KEEP']
                        out[idx] = {'idx': idx, 'paragraph': paragraphs[idx], 'flag': 'IMPORTANT', 
                                   'reason': f'split-kept: {reasons[0] if reasons else "LLM_KEEP"}'}
                    else:
                        reasons = [r for d, r in decisions if d == 'REMOVE']
                        out[idx] = {'idx': idx, 'paragraph': paragraphs[idx], 'flag': 'OUT', 
                                   'reason': f'split-removed: {reasons[0] if reasons else "LLM_REMOVE"}'}

        # Final cleanup for unhandled items
        for i in range(N):
            if out[i] is None:
                out[i] = {'idx': i, 'paragraph': paragraphs[i], 'flag': 'IMPORTANT', 'reason': 'final fallback KEEP'}
        
        if fallback_count > 0:
             logger.info(f"Fallback applied to {fallback_count} items due to parse failures.")
             
        return out

    def process_file(self, md_path: Path, out_dir: Path) -> Dict:
        logger.info(f"Processing: {md_path.name}")
        try:
            text = md_path.read_text(encoding='utf-8', errors='ignore')
            paragraphs = self.chunk_paragraphs(text)
            logger.info(f"  => {len(paragraphs)} chunks")
            
            classified = self.classify_batch(paragraphs)
            
            kept = [c['paragraph'] for c in classified if c['flag'] == 'IMPORTANT']
            
            meta = {
                'file': md_path.name,
                'total_paragraphs': len(paragraphs),
                'kept': len(kept),
                'removed': len(paragraphs) - len(kept),
                'details': classified
            }
            
            out_dir.mkdir(parents=True, exist_ok=True)
            filtered_path = out_dir / f"{md_path.stem}_filtered.md"
            meta_path = out_dir / f"{md_path.stem}_meta.json"
            
            filtered_path.write_text("\n\n".join(kept), encoding='utf-8')
            meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
            
            logger.info(f"Completed {md_path.name}: Kept {meta['kept']}/{meta['total_paragraphs']}")
            return meta
            
        except Exception as e:
            logger.error(f"Error processing file {md_path}: {e}", exc_info=True)
            return {'file': md_path.name, 'error': str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing *.md files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save cleaned md files.")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    model = "microsoft/Phi-4-mini-instruct"
    device = "cuda:0" if is_cuda_available() else "cpu"
    
    if not input_dir.exists():
        logger.error(f"Input directory {input_dir} does not exist.")
        return

    classifier = ContentClassifier(model_name=model, device=device)
    
    md_files = sorted(input_dir.glob("*.md"))
    logger.info(f"Found {len(md_files)} markdown files to process.")
    
    results = []
    for md in md_files:
        results.append(classifier.process_file(md, output_dir))
    
    total = sum(r.get('total_paragraphs', 0) for r in results)
    kept = sum(r.get('kept', 0) for r in results)
    logger.info(f"Batch Complete. Total Paragraphs: {total}, Kept: {kept}")

if __name__ == "__main__":
    main()