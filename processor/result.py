from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ProcessingResult:
    job_id: str
    output_path: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def success(self):
        return self.error is None and self.output_path is not None
