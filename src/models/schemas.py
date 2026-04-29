"""Pydantic models for all inter-agent data contracts."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    """Types of data sources used in due diligence."""
    SEC_FILING = "sec_filing"
    NEWS_ARTICLE = "news_article"
    JOB_POSTING = "job_posting"
    COMPANY_WEBSITE = "company_website"
    SCRAPED_PAGE = "scraped_page"
    OTHER = "other"


class Source(BaseModel):
    """A traceable data source backing one or more claims."""
    id: str = Field(..., description="Unique identifier for this source")
    url: str = Field(..., description="URL where the source was retrieved")
    title: str = Field(..., description="Title or headline of the source")
    retrieved_at: datetime = Field(default_factory=datetime.utcnow, description="When the source was fetched")
    snippet: str = Field(..., description="Relevant excerpt from the source")
    source_type: SourceType = Field(..., description="Category of this source")


class AgentClaim(BaseModel):
    """A single factual claim made by an agent, linked to supporting sources."""
    text: str = Field(..., description="The claim text")
    source_ids: list[str] = Field(default_factory=list, description="IDs of Source objects that support this claim")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Agent's confidence in this claim (0-1)")


class ConflictingClaim(BaseModel):
    """A pair of claims that contradict each other."""
    claim_a: AgentClaim = Field(..., description="First conflicting claim")
    claim_b: AgentClaim = Field(..., description="Second conflicting claim")
    description: str = Field(..., description="Explanation of the conflict")


class AgentSubReport(BaseModel):
    """Structured output from a specialist agent."""
    agent_name: str = Field(..., description="Name of the agent that produced this report")
    findings: list[AgentClaim] = Field(default_factory=list, description="List of claims discovered")
    sources: list[Source] = Field(default_factory=list, description="All sources referenced in findings")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence in this sub-report")
    conflicts: list[ConflictingClaim] = Field(default_factory=list, description="Internal contradictions detected")
    raw_data_summary: str = Field(default="", description="Brief summary of raw data reviewed")


class SynthesizedSection(BaseModel):
    """A section of the final investment memo produced by the synthesis agent."""
    title: str = Field(..., description="Section heading")
    content: str = Field(..., description="Narrative content for this section")
    claims: list[AgentClaim] = Field(default_factory=list, description="Claims supporting this section")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in this section's conclusions")
    conflicting_claims: list[ConflictingClaim] = Field(
        default_factory=list, description="Cross-agent conflicts surfaced in this section"
    )


class InvestmentMemo(BaseModel):
    """The final structured investment memo output."""
    company_name: str = Field(..., description="Name of the company being evaluated")
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="When the memo was generated")
    executive_summary: str = Field(..., description="High-level investment summary")
    sections: list[SynthesizedSection] = Field(default_factory=list, description="Detailed analysis sections")
    overall_confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence in the memo")
    metadata: dict = Field(default_factory=dict, description="Additional metadata (timing, model versions, etc.)")
