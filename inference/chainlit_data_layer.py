"""
JSONL Data Layer for Chainlit
==============================

Custom data layer that uses the existing JSONL session log format
for persistence.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import chainlit.data as cl_data
from chainlit.element import ElementDict
from chainlit.step import StepDict
from chainlit.types import (
    Feedback,
    PageInfo,
    PaginatedResponse,
    Pagination,
    ThreadDict,
    ThreadFilter,
)
from chainlit.user import PersistedUser, User

# Try to import logger, fall back to print if not available
try:
    from utils.logger import get_chat_logger
    logger = get_chat_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class JSONLDataLayer(cl_data.BaseDataLayer):
    """
    Custom Chainlit data layer that uses JSONL session files.
    """
    
    def __init__(self, session_logs_dir: str):
        self.session_logs_dir = Path(session_logs_dir)
        self.session_logs_dir.mkdir(parents=True, exist_ok=True)
        
        self.users_file = self.session_logs_dir / "chainlit_users.json"
        self.threads_index_file = self.session_logs_dir / "chainlit_threads_index.json"
        
        self._init_index_files()
        self._threads_cache: Dict[str, ThreadDict] = {}
        
        logger.info(f"JSONL Data Layer initialized at {self.session_logs_dir}")
    
    # =========================================================================
    # Required Abstract Methods (new in recent Chainlit versions)
    # =========================================================================
    
    async def build_debug_url(self) -> str:
        """Build a debug URL for the current session."""
        return ""
    
    async def close(self) -> None:
        """Close the data layer connection."""
        pass
    
    async def get_thread_author(self, thread_id: str) -> str:
        """Get the author of a thread."""
        index = self._read_json(self.threads_index_file)
        if thread_id in index:
            return index[thread_id].get('userId', 'anonymous')
        return 'anonymous'
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _init_index_files(self):
        if not self.users_file.exists():
            self._write_json(self.users_file, {})
        if not self.threads_index_file.exists():
            self._rebuild_threads_index()
    
    def _read_json(self, path: Path) -> Dict:
        try:
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error reading {path}: {e}")
        return {}
    
    def _write_json(self, path: Path, data: Dict):
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error(f"Error writing {path}: {e}")
    
    def _read_jsonl(self, path: Path) -> List[Dict]:
        entries = []
        try:
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            entries.append(json.loads(line))
        except Exception as e:
            logger.error(f"Error reading JSONL {path}: {e}")
        return entries
    
    def _append_jsonl(self, path: Path, entry: Dict):
        try:
            with open(path, 'a', encoding='utf-8') as f:
                json.dump(entry, f, ensure_ascii=False, default=str)
                f.write('\n')
        except Exception as e:
            logger.error(f"Error appending to JSONL {path}: {e}")
    
    def _write_jsonl(self, path: Path, entries: List[Dict]):
        try:
            with open(path, 'w', encoding='utf-8') as f:
                for entry in entries:
                    json.dump(entry, f, ensure_ascii=False, default=str)
                    f.write('\n')
        except Exception as e:
            logger.error(f"Error writing JSONL {path}: {e}")
    
    def _get_thread_file(self, thread_id: str) -> Path:
        return self.session_logs_dir / f"thread_{thread_id}.jsonl"
    
    def _rebuild_threads_index(self):
        index = {}
        
        for jsonl_file in self.session_logs_dir.glob("thread_*.jsonl"):
            thread_id = jsonl_file.stem.replace("thread_", "")
            entries = self._read_jsonl(jsonl_file)
            
            if entries:
                first_entry = entries[0]
                last_entry = entries[-1]
                
                name = first_entry.get('query', 'Untitled')[:50]
                if len(first_entry.get('query', '')) > 50:
                    name += "..."
                
                index[thread_id] = {
                    'id': thread_id,
                    'name': name,
                    'createdAt': first_entry.get('timestamp', datetime.now().isoformat()),
                    'updatedAt': last_entry.get('timestamp', datetime.now().isoformat()),
                    'file': str(jsonl_file),
                    'entry_count': len(entries)
                }
        
        for jsonl_file in self.session_logs_dir.glob("session_*.jsonl"):
            session_id = jsonl_file.stem.replace("session_", "")
            entries = self._read_jsonl(jsonl_file)
            
            if entries:
                first_entry = entries[0]
                last_entry = entries[-1]
                
                name = first_entry.get('query', 'Untitled')[:50]
                if len(first_entry.get('query', '')) > 50:
                    name += "..."
                
                index[session_id] = {
                    'id': session_id,
                    'name': name,
                    'createdAt': first_entry.get('timestamp', datetime.now().isoformat()),
                    'updatedAt': last_entry.get('timestamp', datetime.now().isoformat()),
                    'file': str(jsonl_file),
                    'entry_count': len(entries),
                    'legacy': True
                }
        
        self._write_json(self.threads_index_file, index)
        logger.info(f"Rebuilt threads index: {len(index)} threads found")
        return index
    
    def _get_timestamp(self) -> str:
        return datetime.utcnow().isoformat() + "Z"
    
    # =========================================================================
    # User Management
    # =========================================================================
    
    async def get_user(self, identifier: str) -> Optional[PersistedUser]:
        users = self._read_json(self.users_file)
        user_data = users.get(identifier)
        
        if user_data:
            return PersistedUser(
                id=user_data.get('id', identifier),
                identifier=identifier,
                metadata=user_data.get('metadata', {}),
                createdAt=user_data.get('createdAt', self._get_timestamp())
            )
        return None
    
    async def create_user(self, user: User) -> Optional[PersistedUser]:
        users = self._read_json(self.users_file)
        
        user_id = users.get(user.identifier, {}).get('id', str(uuid.uuid4()))
        created_at = users.get(user.identifier, {}).get('createdAt', self._get_timestamp())
        
        users[user.identifier] = {
            'id': user_id,
            'identifier': user.identifier,
            'metadata': user.metadata,
            'createdAt': created_at,
            'updatedAt': self._get_timestamp()
        }
        
        self._write_json(self.users_file, users)
        
        return PersistedUser(
            id=user_id,
            identifier=user.identifier,
            metadata=user.metadata,
            createdAt=created_at
        )
    
    # =========================================================================
    # Thread Management
    # =========================================================================
    
    async def list_threads(
        self,
        pagination: Pagination,
        filters: ThreadFilter
    ) -> PaginatedResponse[ThreadDict]:
        index = self._read_json(self.threads_index_file)
        
        threads_list = []
        for thread_id, thread_meta in index.items():
            if filters.userId:
                thread_user = thread_meta.get('userId')
                if thread_user and thread_user != filters.userId:
                    continue
            
            if filters.search:
                name = thread_meta.get('name', '').lower()
                if filters.search.lower() not in name:
                    continue
            
            threads_list.append({
                'id': thread_id,
                'name': thread_meta.get('name', 'Untitled'),
                'createdAt': thread_meta.get('createdAt'),
                'updatedAt': thread_meta.get('updatedAt'),
                'metadata': thread_meta.get('metadata', {}),
                'steps': [],
                'elements': [],
                'userId': thread_meta.get('userId')
            })
        
        threads_list.sort(key=lambda x: x.get('updatedAt', ''), reverse=True)
        
        total = len(threads_list)
        start = 0
        end = pagination.first if pagination.first else total
        
        if pagination.cursor:
            for i, t in enumerate(threads_list):
                if t['id'] == pagination.cursor:
                    start = i + 1
                    break
            end = start + (pagination.first or total)
        
        paginated = threads_list[start:end]
        has_next = end < total
        end_cursor = paginated[-1]['id'] if paginated else None
        
        return PaginatedResponse(
            data=paginated,
            pageInfo=PageInfo(
                hasNextPage=has_next,
                endCursor=end_cursor
            )
        )
    
    async def get_thread(self, thread_id: str) -> Optional[ThreadDict]:
        index = self._read_json(self.threads_index_file)
        
        if thread_id not in index:
            logger.warning(f"Thread {thread_id} not found in index")
            return None
        
        thread_meta = index[thread_id]
        thread_file = Path(thread_meta.get('file', self._get_thread_file(thread_id)))
        
        if not thread_file.exists():
            logger.warning(f"Thread file not found: {thread_file}")
            return None
        
        entries = self._read_jsonl(thread_file)
        
        steps = []
        for entry in entries:
            user_step = {
                'id': entry.get('entry_id', str(uuid.uuid4())) + '_user',
                'name': 'User',
                'type': 'user_message',
                'output': entry.get('query', ''),
                'createdAt': entry.get('timestamp', self._get_timestamp()),
                'threadId': thread_id,
                'parentId': None
            }
            steps.append(user_step)
            
            assistant_step = {
                'id': entry.get('entry_id', str(uuid.uuid4())) + '_assistant',
                'name': 'Assistant',
                'type': 'assistant_message',
                'output': entry.get('answer', ''),
                'createdAt': entry.get('timestamp', self._get_timestamp()),
                'threadId': thread_id,
                'parentId': None,
                'metadata': {
                    'entry_id': entry.get('entry_id'),
                    'config': entry.get('config', {}),
                    'timings': entry.get('timings', {}),
                    'hallucination_check': entry.get('hallucination_check'),
                    'final_sources': entry.get('final_sources', [])
                }
            }
            steps.append(assistant_step)
        
        thread_dict: ThreadDict = {
            'id': thread_id,
            'name': thread_meta.get('name', 'Untitled'),
            'createdAt': thread_meta.get('createdAt'),
            'updatedAt': thread_meta.get('updatedAt'),
            'metadata': thread_meta.get('metadata', {}),
            'steps': steps,
            'elements': [],
            'userId': thread_meta.get('userId')
        }
        
        return thread_dict
    
    async def update_thread(
        self,
        thread_id: str,
        name: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None
    ):
        index = self._read_json(self.threads_index_file)
        
        if thread_id not in index:
            index[thread_id] = {
                'id': thread_id,
                'name': name or 'New Conversation',
                'createdAt': self._get_timestamp(),
                'updatedAt': self._get_timestamp(),
                'file': str(self._get_thread_file(thread_id)),
                'entry_count': 0
            }
        
        if name is not None:
            index[thread_id]['name'] = name
        if user_id is not None:
            index[thread_id]['userId'] = user_id
        if metadata is not None:
            index[thread_id]['metadata'] = metadata
        if tags is not None:
            index[thread_id]['tags'] = tags
        
        index[thread_id]['updatedAt'] = self._get_timestamp()
        
        self._write_json(self.threads_index_file, index)
    
    async def delete_thread(self, thread_id: str):
        index = self._read_json(self.threads_index_file)
        
        if thread_id in index:
            thread_file = Path(index[thread_id].get('file', self._get_thread_file(thread_id)))
            if thread_file.exists():
                thread_file.unlink()
            
            del index[thread_id]
            self._write_json(self.threads_index_file, index)
            logger.info(f"Deleted thread: {thread_id}")
    
    # =========================================================================
    # Step Management
    # =========================================================================
    
    async def create_step(self, step_dict: StepDict):
        thread_id = step_dict.get('threadId')
        if not thread_id:
            return
        
        index = self._read_json(self.threads_index_file)
        if thread_id in index:
            index[thread_id]['updatedAt'] = self._get_timestamp()
            
            if index[thread_id].get('name') in ['New Conversation', 'Untitled']:
                if step_dict.get('type') == 'user_message':
                    content = step_dict.get('output', '')[:50]
                    if len(step_dict.get('output', '')) > 50:
                        content += "..."
                    index[thread_id]['name'] = content
            
            self._write_json(self.threads_index_file, index)
    
    async def update_step(self, step_dict: StepDict):
        pass
    
    async def delete_step(self, step_id: str):
        pass
    
    # =========================================================================
    # RAG Pipeline Integration
    # =========================================================================
    
    async def add_rag_entry(self, thread_id: str, entry: Dict):
        thread_file = self._get_thread_file(thread_id)
        self._append_jsonl(thread_file, entry)
        
        index = self._read_json(self.threads_index_file)
        if thread_id in index:
            index[thread_id]['updatedAt'] = entry.get('timestamp', self._get_timestamp())
            index[thread_id]['entry_count'] = index[thread_id].get('entry_count', 0) + 1
            
            if index[thread_id].get('name') in ['New Conversation', 'Untitled']:
                query = entry.get('query', '')[:50]
                if len(entry.get('query', '')) > 50:
                    query += "..."
                index[thread_id]['name'] = query
            
            self._write_json(self.threads_index_file, index)
    
    async def get_rag_entry(self, thread_id: str, entry_id: str) -> Optional[Dict]:
        thread_file = self._get_thread_file(thread_id)
        if not thread_file.exists():
            return None
        
        entries = self._read_jsonl(thread_file)
        for entry in entries:
            if entry.get('entry_id') == entry_id:
                return entry
        return None
    
    # =========================================================================
    # Feedback
    # =========================================================================
    
    async def upsert_feedback(self, feedback: Feedback) -> str:
        step_id = feedback.forId
        index = self._read_json(self.threads_index_file)
        
        for thread_id, thread_meta in index.items():
            thread_file = Path(thread_meta.get('file', self._get_thread_file(thread_id)))
            if thread_file.exists():
                entries = self._read_jsonl(thread_file)
                for entry in entries:
                    entry_id = entry.get('entry_id', '')
                    if step_id and step_id.startswith(entry_id):
                        if 'feedback' not in entry:
                            entry['feedback'] = []
                        
                        entry['feedback'].append({
                            'id': feedback.id or str(uuid.uuid4()),
                            'value': feedback.value,
                            'comment': feedback.comment,
                            'createdAt': self._get_timestamp()
                        })
                        
                        self._write_jsonl(thread_file, entries)
                        return feedback.id or str(uuid.uuid4())
        
        return str(uuid.uuid4())
    
    async def delete_feedback(self, feedback_id: str) -> bool:
        return True
    
    # =========================================================================
    # Elements
    # =========================================================================
    
    async def create_element(self, element: ElementDict):
        pass
    
    async def get_element(self, thread_id: str, element_id: str) -> Optional[ElementDict]:
        return None
    
    async def delete_element(self, element_id: str, thread_id: Optional[str] = None):
        pass